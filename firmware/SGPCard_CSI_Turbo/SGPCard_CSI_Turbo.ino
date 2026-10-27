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
 *   Hardware: LilyGO T-Display (ESP32-D0WDQ6-V3, 16MB flash, CH340K USB-serial,
 *   1.14" 135x240 ST7789). Ported from the upstream SGP Card Mini (ESP32-S3)
 *   build, which drove a NeoPixel on pin 5 and a sensor rail on pin 6; the
 *   T-Display has neither (its pin 5 is the screen's chip select). The link
 *   state the NeoPixel showed is now on the screen, readable across the room,
 *   and on the serial port once a second.
 *
 *   Screen: SSID + channel, link state (NO CLIENT / JOINED / STREAMING, or
 *   NO CSI / CSI ERROR / AP DOWN in red), CSI frames per second, clients,
 *   UDP frames sent per second and drops.
 *   Send "shot" + newline on the serial port to get the current frame back as
 *   "SHOT 240 135\n" + 240*135 RGB565 pixels (little-endian) + "\nEND\n".
 *   (a WiFi event line printed mid-frame corrupts it: just ask again)
 *
 *   Build: arduino-esp32 core 3.3.x · board "LilyGo T-Display"
 *          (esp32:esp32:lilygo_t_display) · Flash Size 16MB · Partition
 *          "Huge APP" · Lib: Adafruit ST7735 and ST7789 Library (+ Adafruit GFX).
 * ═══════════════════════════════════════════════════════════════════════════
 */
#include <WiFi.h>
#include <WiFiUdp.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include "esp_wifi.h"

#define FW_VERSION    "SGP CSI TURBO v1.0"

// ───────── T-Display screen (ST7789, 135x240) ─────────
#define TFT_MOSI      19
#define TFT_SCLK      18
#define TFT_CS        5
#define TFT_DC        16
#define TFT_RST       23
#define TFT_BL        4
#define TFT_ROTATION  1          // landscape; 3 = the same, turned 180°
#define TFT_W         240
#define TFT_H         135

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
bool         apOk = false, csiOk = false;
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

// ───────── status screen (drawn off-screen, blitted once a second) ─────────
Adafruit_ST7789 tft(&SPI, TFT_CS, TFT_DC, TFT_RST);
GFXcanvas16*    screen = nullptr;      // null → no RAM for it; serial status only
size_t          shotPos = SIZE_MAX;    // bytes of a "shot" already sent; MAX = idle

#define C_BG      0x0000
#define C_BAR     0x2124
#define C_TEXT    0xFFFF
#define C_DIM     0x8C51
#define C_GREEN   0x07E8
#define C_AMBER   0xFD20
#define C_BLUE    0x3D9F
#define C_RED     0xF8A2

void textAt(int x, int y, uint8_t size, uint16_t col, const char* s) {
  screen->setTextSize(size); screen->setTextColor(col); screen->setCursor(x, y);
  screen->print(s);
}
void textRight(int xr, int y, uint8_t size, uint16_t col, const char* s) {
  textAt(xr - (int)strlen(s) * 6 * size, y, size, col, s);   // built-in font: 6 px/char
}

void screenBegin() {
  pinMode(TFT_BL, OUTPUT); digitalWrite(TFT_BL, HIGH);
  SPI.begin(TFT_SCLK, -1, TFT_MOSI, TFT_CS);
  tft.init(TFT_H, TFT_W);              // panel is 135x240 portrait natively
  tft.setSPISpeed(40000000);
  tft.setRotation(TFT_ROTATION);
  tft.fillScreen(C_BG);
  screen = new GFXcanvas16(TFT_W, TFT_H);
  if (!screen->getBuffer()) {
    delete screen; screen = nullptr;
    Serial.println("[TFT] no RAM for the frame buffer: screen stays blank");
    return;
  }
  uint32_t t0 = micros();             // the blit is what loop() pays once a second
  tft.drawRGBBitmap(0, 0, screen->getBuffer(), TFT_W, TFT_H);
  Serial.printf("[TFT] ready, frame blit %lu us\n", (unsigned long)(micros() - t0));
}

void drawStatus(const char* state, uint16_t col, int clients,
                uint32_t rxHz, uint32_t txHz, uint32_t drHz) {
  if (!screen || shotPos != SIZE_MAX) return;   // a "shot" is reading this frame
  char b[24];
  screen->fillScreen(C_BG);
  // header: which network to join
  screen->fillRect(0, 0, TFT_W, 22, C_BAR);
  textAt(6, 4, 2, C_TEXT, apName.c_str());
  snprintf(b, sizeof(b), "ch%d", AP_CHANNEL);
  textRight(TFT_W - 6, 4, 2, C_DIM, b);
  // link state
  screen->fillCircle(14, 40, 7, col);
  textAt(30, 29, 3, col, state);
  // CSI frames per second, big
  snprintf(b, sizeof(b), "%lu", (unsigned long)min(rxHz, (uint32_t)9999));
  textRight(170, 60, 6, rxHz ? C_TEXT : C_DIM, b);
  textAt(178, 64, 2, C_DIM, "CSI");
  textAt(178, 86, 2, C_DIM, "fr/s");
  // clients joined · frames shipped to the PC · ring overflows
  snprintf(b, sizeof(b), "PC %d", clients);
  textAt(6, 115, 2, clients ? C_TEXT : C_DIM, b);
  if (drHz) { snprintf(b, sizeof(b), "DROP %lu/s", (unsigned long)drHz); textRight(TFT_W - 6, 115, 2, C_RED, b); }
  else      { snprintf(b, sizeof(b), "UDP %lu/s",  (unsigned long)txHz); textRight(TFT_W - 6, 115, 2, txHz ? C_TEXT : C_DIM, b); }
  tft.drawRGBBitmap(0, 0, screen->getBuffer(), TFT_W, TFT_H);
}

// "shot\n" on the serial port → stream the frame buffer back, a FIFO's worth
// per loop() pass so the CSI drain never stalls behind a 64 KB serial write
void serviceShot() {
  static char cmd[8]; static uint8_t cl = 0;
  if (shotPos == SIZE_MAX) {
    while (Serial.available()) {                 // never wait for the rest of a line
      char c = (char)Serial.read();
      if (c != '\n' && c != '\r') { if (cl < sizeof(cmd) - 1) cmd[cl++] = c; continue; }
      cmd[cl] = 0; bool shot = cl && !strcmp(cmd, "shot"); cl = 0;
      if (shot && screen) { Serial.printf("SHOT %d %d\n", TFT_W, TFT_H); shotPos = 0; break; }
    }
    if (shotPos == SIZE_MAX) return;
  }
  const size_t total = (size_t)TFT_W * TFT_H * 2;
  size_t n = min((size_t)Serial.availableForWrite(), total - shotPos);
  Serial.write((const uint8_t*)screen->getBuffer() + shotPos, n);
  shotPos += n;
  if (shotPos == total) { Serial.print("\nEND\n"); shotPos = SIZE_MAX; }
}

void setup() {
  Serial.begin(115200); delay(150);
  Serial.println("\n" FW_VERSION);
  screenBegin();                       // before WiFi: the frame buffer needs 64 KB in one piece

  apName = "SGP_TURBO_" + String((uint16_t)(ESP.getEfuseMac() & 0xFFFF), HEX);
  WiFi.onEvent(onWifiEvent);
  WiFi.mode(WIFI_AP);
  apOk = WiFi.softAP(apName.c_str(), nullptr, AP_CHANNEL);
  if (!apOk) Serial.println("[AP] softAP FAILED");
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

  // 3) status screen + rate print (once a second; one ~19 ms blit)
  static unsigned long tStat = 0;
  static uint32_t lastSent = 0, lastRx = 0, lastDropped = 0;
  unsigned long now = millis();
  if (now - tStat >= 1000) {
    uint32_t rx = __atomic_load_n(&csiRx, __ATOMIC_RELAXED);
    uint32_t dr = __atomic_load_n(&dropped, __ATOMIC_RELAXED);
    uint32_t hz = framesSent - lastSent, rxHz = rx - lastRx, drHz = dr - lastDropped;
    lastSent = framesSent; lastRx = rx; lastDropped = dr; tStat = now;
    int clients = WiFi.softAPgetStationNum();
    const char* state; uint16_t col;
    if (!apOk)              { state = "AP DOWN";   col = C_RED;   }
    else if (!csiOk)        { state = "CSI ERROR"; col = C_RED;   }
    else if (live && !rxHz) { state = "NO CSI";    col = C_RED;   }   // pinged, radio silent
    else if (live)          { state = "STREAMING"; col = C_GREEN; }
    else if (clients)       { state = "JOINED";    col = C_AMBER; }   // waiting for pings
    else                    { state = "NO CLIENT"; col = C_BLUE;  }
    drawStatus(state, col, clients, rxHz, hz, drHz);
    if (shotPos == SIZE_MAX)                       // don't splice text into a shot
      Serial.printf("%-9s  clients %d  CSI %lu/s  sent %lu/s  dropped %lu\n", state, clients,
                    (unsigned long)rxHz, (unsigned long)hz, (unsigned long)drHz);
  }

  // 4) serial screenshot, a chunk per pass
  serviceShot();
}
