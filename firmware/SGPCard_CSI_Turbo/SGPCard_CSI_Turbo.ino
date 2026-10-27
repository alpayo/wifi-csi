/*
 * ═══════════════════════════════════════════════════════════════════════════
 *   SGP CSI TURBO  —  single-device, maximum-rate WiFi-CSI sensor
 *   LilyGO T-Display (classic ESP32) · creates its OWN WiFi network and
 *   streams CSI as fast as it can
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   This is the HIGH-PERFORMANCE single-card firmware. It is a pure sensor:
 *   no DSP, no mic, no web server — every cycle goes into capturing the WiFi
 *   channel and shipping it out, so the PC gets the densest, fastest CSI stream
 *   possible and can detect the tiniest movement.
 *
 *   How it works:
 *     1. The card is its own access point  "SGP_TURBO_xxxx"  (channel 6).
 *     2. The PC joins it and FLOODS tiny UDP pings to 192.168.4.1:5566.
 *     3. Every packet the card receives makes the radio produce one CSI
 *        snapshot; a lock-free ring buffer + a tight drain loop send each one
 *        straight back. CSI rate = ping rate the radio can keep up with
 *        (hundreds to ~1k frames/s — the PC's "Turbo" ping drives it).
 *
 *   The frame is the SAME 26-byte "SG" header + CSI as SGPCard_CSI_Node, with
 *   the acoustic fields left at 0, so the PC program (SGP CSI WIFI) reads this
 *   firmware with no changes. It also answers the "SD" discovery probe, so the
 *   app still auto-detects it.
 *
 *   Hardware: LilyGO T-Display (ESP32-D0WDQ6-V3, 16MB flash, CH340K USB-serial).
 *   Ported from the upstream SGP Card Mini (ESP32-S3) build, which drove a
 *   NeoPixel on pin 5 and a sensor rail on pin 6; the T-Display has neither
 *   (its pin 5 is the display's chip select), so both are gone and the status
 *   they showed is printed on the serial port once a second.
 *   Build: board esp32:esp32:esp32 (ESP32 Dev Module) · Partition "Huge APP"
 *          · Flash 4MB+ · arduino-esp32 core 3.3.x · no extra libraries.
 * ═══════════════════════════════════════════════════════════════════════════
 */
#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"

#define FW_VERSION    "SGP CSI TURBO v1.0"

// ───────── network ─────────
#define AP_CHANNEL    6          // fixed channel — CSI is per-channel
#define UDP_PORT      5566
#define PC_TIMEOUT_MS 2500

// ───────── CSI frame ─────────
#define CSI_MAX       128        // 64 sub-carriers * (I,Q) from the L-LTF
#define HDR_LEN       26
#define RING_N        512        // CSI frames buffered (MUST be a power of two)

struct Frame {
  uint8_t  buf[CSI_MAX];
  uint16_t len;
  uint32_t t_ms;
  int8_t   rssi, noise;
  uint8_t  channel, flags;
};
static Frame    ring[RING_N];
// 1 producer (CSI cb, WiFi task on core 0), 1 consumer (loop, core 1): the
// release/acquire pair makes a frame's bytes visible before its index is.
static uint16_t rHead = 0, rTail = 0;

WiFiUDP      udp;
IPAddress    pcIP;
uint16_t     pcPort = 0;
unsigned long pcLastSeen = 0;
uint32_t     seq = 0, framesSent = 0;
// written only by the CSI callback and never reset: loop() reads per-second
// deltas, so no increment is lost to a cross-core read-modify-write
static uint32_t csiRx = 0, dropped = 0;
String       apName;
bool         csiOk = false;
uint8_t      staMac[6]; bool staKnown = false;

// ───────── CSI capture (WiFi task — keep it tiny: copy into the ring) ─────────
void csiCallback(void* ctx, wifi_csi_info_t* info) {
  if (!info || !info->buf || info->len == 0) return;
  if (staKnown && memcmp(info->mac, staMac, 6) != 0) return;   // only the PC link
  uint16_t head = rHead;                                       // only we write it
  uint16_t next = (uint16_t)((head + 1) & (RING_N - 1));
  if (next == __atomic_load_n(&rTail, __ATOMIC_ACQUIRE)) { __atomic_fetch_add(&dropped, 1, __ATOMIC_RELAXED); return; }   // full → drop newest
  Frame& f = ring[head];
  f.len     = info->len > CSI_MAX ? CSI_MAX : info->len;
  memcpy(f.buf, info->buf, f.len);
  f.t_ms    = millis();
  f.rssi    = info->rx_ctrl.rssi;
  f.noise   = info->rx_ctrl.noise_floor;
  f.channel = info->rx_ctrl.channel;
  f.flags   = info->first_word_invalid ? 1 : 0;
  __atomic_store_n(&rHead, next, __ATOMIC_RELEASE);
  __atomic_fetch_add(&csiRx, 1, __ATOMIC_RELAXED);
}

bool csiStart() {
  wifi_csi_config_t cfg = {};
  cfg.lltf_en           = true;    // legacy LTF: in every packet → 64 sub-carriers
  cfg.htltf_en          = false;
  cfg.stbc_htltf2_en    = false;
  cfg.ltf_merge_en      = false;
  cfg.channel_filter_en = false;   // independent sub-carriers (better for sensing)
  cfg.manu_scale        = false;
  cfg.shift             = 0;
  cfg.dump_ack_en       = false;
  // a core built without CSI support fails here, not later — say so loudly
  esp_err_t e = esp_wifi_set_csi_config(&cfg);
  if (e == ESP_OK) e = esp_wifi_set_csi_rx_cb(csiCallback, NULL);
  if (e == ESP_OK) e = esp_wifi_set_csi(true);
  if (e != ESP_OK) Serial.printf("[CSI] start FAILED: %s\n", esp_err_to_name(e));
  return e == ESP_OK;
}

void onWifiEvent(WiFiEvent_t ev, WiFiEventInfo_t info) {
  if (ev == ARDUINO_EVENT_WIFI_AP_STACONNECTED) {
    memcpy(staMac, info.wifi_ap_staconnected.mac, 6); staKnown = true;
    Serial.println("[AP] PC joined");
  } else if (ev == ARDUINO_EVENT_WIFI_AP_STADISCONNECTED) {
    staKnown = false; Serial.println("[AP] PC left");
  }
}

// ───────── send one CSI frame (same "SG" layout; mic fields = 0) ─────────
inline void sendFrame(const Frame& f) {
  uint8_t pkt[HDR_LEN + CSI_MAX];
  pkt[0] = 'S'; pkt[1] = 'G'; pkt[2] = 1; pkt[3] = f.flags;
  uint32_t s = seq++;
  memcpy(pkt + 4, &s, 4);
  memcpy(pkt + 8, &f.t_ms, 4);
  pkt[12] = (uint8_t)f.rssi; pkt[13] = (uint8_t)f.noise; pkt[14] = f.channel; pkt[15] = 0;
  memset(pkt + 16, 0, 8);                 // acoustic fields unused in turbo
  memcpy(pkt + 24, &f.len, 2);
  memcpy(pkt + HDR_LEN, f.buf, f.len);
  udp.beginPacket(pcIP, pcPort);
  udp.write(pkt, HDR_LEN + f.len);
  udp.endPacket();
  framesSent++;
}

// reply to a discovery probe so the PC app auto-detects this card
void sendHello(IPAddress ip, uint16_t port) {
  uint8_t b[64]; int o = 0;
  b[o++] = 'S'; b[o++] = 'G'; b[o++] = 'H'; b[o++] = 1;
  uint8_t L = (uint8_t)min((int)apName.length(), 32);
  b[o++] = L; memcpy(b + o, apName.c_str(), L); o += L;
  uint64_t mac = ESP.getEfuseMac();
  for (int i = 0; i < 6; i++) b[o++] = (uint8_t)((mac >> (8 * i)) & 0xFF);
  udp.beginPacket(ip, port); udp.write(b, o); udp.endPacket();
}

void setup() {
  Serial.begin(115200); delay(150);
  Serial.println("\n" FW_VERSION);

  apName = "SGP_TURBO_" + String((uint16_t)(ESP.getEfuseMac() & 0xFFFF), HEX);
  WiFi.onEvent(onWifiEvent);
  WiFi.mode(WIFI_AP);
  WiFi.softAP(apName.c_str(), nullptr, AP_CHANNEL);
  WiFi.setSleep(WIFI_PS_NONE);                 // never sleep — maximum throughput
  esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_BW_HT20);
  esp_wifi_set_max_tx_power(80);               // strong link = cleaner CSI
  csiOk = csiStart();
  udp.begin(UDP_PORT);

  Serial.printf("AP \"%s\" ch%d  ->  join it, run SGP CSI WIFI (Turbo ping)\n",
                apName.c_str(), AP_CHANNEL);
}

void loop() {
  // 1) intake PC packets: "SD" = discovery (reply hello), else a ping (stream target)
  int sz = udp.parsePacket();
  while (sz > 0) {
    uint8_t tmp[64]; int n = udp.read(tmp, sizeof(tmp));
    if (n >= 2 && tmp[0] == 'S' && tmp[1] == 'D') {
      sendHello(udp.remoteIP(), udp.remotePort());
    } else {
      pcIP = udp.remoteIP(); pcPort = udp.remotePort(); pcLastSeen = millis();
    }
    sz = udp.parsePacket();
  }

  // 2) drain the CSI ring as fast as possible → PC
  bool live = pcPort && (millis() - pcLastSeen < PC_TIMEOUT_MS);
  int burst = 0;
  uint16_t tail = rTail;                                       // only we write it
  while (tail != __atomic_load_n(&rHead, __ATOMIC_ACQUIRE)) {
    if (live) sendFrame(ring[tail]);
    tail = (uint16_t)((tail + 1) & (RING_N - 1));
    __atomic_store_n(&rTail, tail, __ATOMIC_RELEASE);
    if (++burst >= 64) break;          // yield to WiFi after a burst, then come back
  }

  // 3) status LED + rate print (once a second; cheap)
  static unsigned long tStat = 0;
  static uint32_t lastSent = 0, lastRx = 0, lastDropped = 0;
  unsigned long now = millis();
  if (now - tStat >= 1000) {
    uint32_t rx = __atomic_load_n(&csiRx, __ATOMIC_RELAXED);
    uint32_t dr = __atomic_load_n(&dropped, __ATOMIC_RELAXED);
    uint32_t hz = framesSent - lastSent, rxHz = rx - lastRx, drHz = dr - lastDropped;
    lastSent = framesSent; lastRx = rx; lastDropped = dr; tStat = now;
    Serial.printf("%s  clients %d  CSI %lu/s  sent %lu/s  dropped %lu\n",
                  live ? "STREAMING" : "WAITING  ", WiFi.softAPgetStationNum(),
                  (unsigned long)rxHz, (unsigned long)hz, (unsigned long)drHz);
  }
}
