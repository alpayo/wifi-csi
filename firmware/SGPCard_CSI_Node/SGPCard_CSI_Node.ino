/*
 * ═══════════════════════════════════════════════════════════════════════════
 *   SGP SGPCARD NODE  —  WiFi-CSI sensing node for the SGP Card Mini
 *   ESP32-S3 · streams raw RF (WiFi CSI) + acoustic features to the PC
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   This firmware is a PURE SENSOR. It does no heavy maths: it captures the
 *   WiFi Channel State Information (how the 2.4 GHz radio channel is distorted
 *   by people moving / breathing) plus microphone energy, and streams it raw
 *   to the PC program (SGP SGPCard Command), which runs all the DSP + AI and
 *   draws the person in 3D.
 *
 *   How the data flows:
 *     1. The card creates a WiFi access point  "SGP_SGPCARD_xxxx"  (channel 6).
 *     2. The PC joins it and sends small UDP "pings" to 192.168.4.1:5566
 *        (~100 per second). Every ping the card RECEIVES produces one CSI
 *        snapshot of the radio channel between the PC and the card.
 *     3. The card sends each CSI snapshot straight back to the PC over UDP.
 *        A person moving near that radio path distorts the CSI.
 *
 *   Frame sent to the PC (little-endian, 26-byte header + CSI):
 *     0  'S''G'   magic         12 rssi (int8)     16 mic_rms  (u16)
 *     2  ver=1                  13 noise (int8)    18 mic_low  (u16)  <300 Hz
 *     3  flags (b0=first_word_invalid)             20 mic_mid  (u16)  0.3-2 kHz
 *     4  seq  (u32)             14 channel (u8)    22 mic_high (u16)  >2 kHz
 *     8  t_ms (u32)             15 reserved        24 csi_len  (u16)
 *     26.. csi int8[csi_len]  — I,Q interleaved, 64 sub-carriers (L-LTF)
 *
 *   Hardware (SGP Card Mini PCB v3.7): mic ICS-43434 I2S  WS=2 SD=3 SCK=4,
 *   RGB WS2812=5, buzzer=1, sensor 3V3 rail CE=6.
 *   Build: ESP32-S3, Partition "Huge APP", Flash 8MB.  Libs: Adafruit NeoPixel.
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>
#include <Adafruit_NeoPixel.h>
#include <ESP_I2S.h>
#include "esp_wifi.h"

#define FW_VERSION "SGP SGPCARD NODE v1.0"

// ───────── pins ─────────
#define PIN_RGB_LED    5
#define PIN_BUZZER     1
#define PIN_SENSOR_CE  6
#define PIN_MIC_WS     2
#define PIN_MIC_SD     3
#define PIN_MIC_SCK    4

// ───────── network ─────────
#define AP_CHANNEL     6          // fixed channel — CSI is per-channel
#define UDP_PORT       5566       // PC pings arrive here; CSI goes back to the sender
#define PC_TIMEOUT_MS  2500       // stop streaming if the PC goes quiet

// ───────── multi-card mode (optional) ──────────────────────────────────────
// Default: each card is its own access point and the PC joins it — simplest,
// perfect for ONE card.
//
// For SEVERAL cards at once (real x,y trilateration): the PC can only be on one
// WiFi at a time, so every card AND the PC must share ONE network. Set
// STATION_MODE to 1 and fill in your home WiFi below; the card then JOINS that
// WiFi (as a station) instead of making its own AP, and the router gives it an
// IP. Flash every card the same — each gets its own IP — then add each card by
// that IP in the PC program's "Nodes" panel. If the join fails it falls back to
// AP mode so you are never locked out.
#define STATION_MODE   0                  // 1 = join the WiFi below (multi-card)
#define STA_SSID       "YOUR_WIFI"
#define STA_PASS       "YOUR_PASSWORD"

// ───────── CSI frame ─────────
#define CSI_MAX        128        // 64 sub-carriers * (I,Q) from the L-LTF
#define HDR_LEN        26
struct CsiFrame {
  uint32_t t_ms;
  int8_t   rssi, noise;
  uint8_t  channel, flags;
  uint16_t len;
  int8_t   buf[CSI_MAX];
};

// ───────── objects / state ─────────
WiFiUDP udp;
WebServer web(80);
Adafruit_NeoPixel led(1, PIN_RGB_LED, NEO_GRB + NEO_KHZ800);
I2SClass mic;

QueueHandle_t csiQ;
volatile bool  staKnown = false;
uint8_t        staMac[6];
IPAddress      pcIP;
uint16_t       pcPort = 0;
unsigned long  pcLastSeen = 0;
uint32_t       seq = 0, framesSent = 0, framesDropped = 0, pingsRx = 0;
float          csiRateHz = 0;
String         apName;

// acoustic features (written by mic task, read by loop)
volatile uint16_t micRms = 0, micLow = 0, micMid = 0, micHigh = 0;
volatile bool     micOk = false;

// ═══════════════════════════════════════════════════════════════════════════
// LED + buzzer
// ═══════════════════════════════════════════════════════════════════════════
void ledColor(uint8_t r, uint8_t g, uint8_t b) {
  led.setPixelColor(0, led.Color(r, g, b));
  WiFi.setSleep(WIFI_PS_MAX_MODEM); led.show(); WiFi.setSleep(WIFI_PS_NONE);
}
void beep(uint16_t f, uint16_t ms) {
  ledcAttach(PIN_BUZZER, f, 8); ledcWrite(PIN_BUZZER, 128); delay(ms);
  ledcWrite(PIN_BUZZER, 0); ledcDetach(PIN_BUZZER);
  pinMode(PIN_BUZZER, OUTPUT); digitalWrite(PIN_BUZZER, LOW);
}

// ═══════════════════════════════════════════════════════════════════════════
// CSI capture (runs in the WiFi task — keep it short: copy + enqueue)
// ═══════════════════════════════════════════════════════════════════════════
void csiCallback(void* ctx, wifi_csi_info_t* info) {
  if (!info || !info->buf || info->len == 0) return;
  // only keep CSI of packets sent by the PC (the associated station)
  if (staKnown && memcmp(info->mac, staMac, 6) != 0) return;
  CsiFrame f;
  f.t_ms    = millis();
  f.rssi    = info->rx_ctrl.rssi;
  f.noise   = info->rx_ctrl.noise_floor;
  f.channel = info->rx_ctrl.channel;
  f.flags   = info->first_word_invalid ? 1 : 0;
  f.len     = info->len > CSI_MAX ? CSI_MAX : info->len;
  memcpy(f.buf, info->buf, f.len);
  if (xQueueSend(csiQ, &f, 0) != pdTRUE) framesDropped++;
}

void csiStart() {
  wifi_csi_config_t cfg = {};
  cfg.lltf_en           = true;    // legacy LTF: present in every packet → 64 sub-carriers
  cfg.htltf_en          = false;   // keep a fixed 128-byte layout
  cfg.stbc_htltf2_en    = false;
  cfg.ltf_merge_en      = false;
  cfg.channel_filter_en = false;   // keep sub-carriers independent (better for sensing)
  cfg.manu_scale        = false;
  cfg.shift             = 0;
  cfg.dump_ack_en       = false;
  esp_wifi_set_csi_config(&cfg);
  esp_wifi_set_csi_rx_cb(csiCallback, NULL);
  esp_wifi_set_csi(true);
}

void onWifiEvent(WiFiEvent_t ev, WiFiEventInfo_t info) {
  if (ev == ARDUINO_EVENT_WIFI_AP_STACONNECTED) {
    memcpy(staMac, info.wifi_ap_staconnected.mac, 6);
    staKnown = true;
    Serial.printf("[AP] PC joined %02x:%02x:%02x:%02x:%02x:%02x\n",
                  staMac[0], staMac[1], staMac[2], staMac[3], staMac[4], staMac[5]);
  } else if (ev == ARDUINO_EVENT_WIFI_AP_STADISCONNECTED) {
    staKnown = false;
    Serial.println("[AP] PC left");
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Microphone task — 16 kHz, 32 ms windows → RMS + 3 band energies
// ═══════════════════════════════════════════════════════════════════════════
void micTask(void*) {
  mic.setPins(PIN_MIC_SCK, PIN_MIC_WS, -1, PIN_MIC_SD);
  if (!mic.begin(I2S_MODE_STD, 16000, I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("[MIC] init FAILED — acoustic channel disabled");
    vTaskDelete(NULL);
  }
  micOk = true;
  Serial.println("[MIC] ICS-43434 OK");
  static int32_t raw[512];
  float lp = 0, hpPrev = 0, hpOut = 0, dc = 0;
  const float aLow = 0.111f;   // 1-pole low-pass  ~300 Hz @16 kHz
  const float aHigh = 0.56f;   // 1-pole high-pass ~2 kHz  @16 kHz
  while (true) {
    size_t n = mic.readBytes((char*)raw, sizeof(raw)) / 4;
    if (n == 0) { vTaskDelay(2); continue; }
    double eAll = 0, eLow = 0, eHigh = 0;
    for (size_t i = 0; i < n; i++) {
      float s = (float)(raw[i] >> 8);           // 24-bit sample
      dc += 0.001f * (s - dc); s -= dc;          // remove DC offset
      lp += aLow * (s - lp);                     // low band
      hpOut = aHigh * (hpOut + s - hpPrev);      // high band
      hpPrev = s;
      eAll += s * s; eLow += lp * lp; eHigh += hpOut * hpOut;
    }
    double eMid = eAll - eLow - eHigh; if (eMid < 0) eMid = 0;
    auto sc = [](double e, size_t n) -> uint16_t {
      double v = sqrt(e / n) / 8.0; return v > 65535 ? 65535 : (uint16_t)v;
    };
    micRms = sc(eAll, n); micLow = sc(eLow, n); micMid = sc(eMid, n); micHigh = sc(eHigh, n);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Streaming to the PC
// ═══════════════════════════════════════════════════════════════════════════
void sendFrame(const CsiFrame& f) {
  uint8_t pkt[HDR_LEN + CSI_MAX];
  pkt[0] = 'S'; pkt[1] = 'G'; pkt[2] = 1; pkt[3] = f.flags;
  uint32_t s = seq++;
  memcpy(pkt + 4, &s, 4);
  memcpy(pkt + 8, &f.t_ms, 4);
  pkt[12] = (uint8_t)f.rssi; pkt[13] = (uint8_t)f.noise; pkt[14] = f.channel; pkt[15] = micOk ? 1 : 0;
  uint16_t m0 = micRms, m1 = micLow, m2 = micMid, m3 = micHigh;
  memcpy(pkt + 16, &m0, 2); memcpy(pkt + 18, &m1, 2); memcpy(pkt + 20, &m2, 2); memcpy(pkt + 22, &m3, 2);
  memcpy(pkt + 24, &f.len, 2);
  memcpy(pkt + HDR_LEN, f.buf, f.len);
  udp.beginPacket(pcIP, pcPort);
  udp.write(pkt, HDR_LEN + f.len);
  udp.endPacket();
  framesSent++;
}

// Reply to a discovery probe so the PC can auto-detect this card on the network.
// Frame: 'S''G''H'  ver=1  namelen  name[namelen]  mac[6]
void sendHello(IPAddress ip, uint16_t port) {
  uint8_t b[64]; int o = 0;
  b[o++] = 'S'; b[o++] = 'G'; b[o++] = 'H'; b[o++] = 1;
  uint8_t L = (uint8_t)min((int)apName.length(), 32);
  b[o++] = L;
  memcpy(b + o, apName.c_str(), L); o += L;
  uint64_t mac = ESP.getEfuseMac();
  for (int i = 0; i < 6; i++) b[o++] = (uint8_t)((mac >> (8 * i)) & 0xFF);
  udp.beginPacket(ip, port);
  udp.write(b, o);
  udp.endPacket();
}

// ═══════════════════════════════════════════════════════════════════════════
// Tiny status page (the real UI lives on the PC)
// ═══════════════════════════════════════════════════════════════════════════
void handleRoot() {
  bool live = pcPort && (millis() - pcLastSeen < PC_TIMEOUT_MS);
  String h = F("<!DOCTYPE html><html><head><meta name=viewport content='width=device-width,initial-scale=1'>"
               "<meta http-equiv=refresh content=2><title>SGP SGPCARD NODE</title><style>"
               "body{background:#05080a;color:#9fe8b4;font-family:ui-monospace,Menlo,monospace;padding:20px}"
               "h1{font-size:16px;letter-spacing:.2em;color:#39ff88}td{padding:4px 14px 4px 0}"
               ".k{color:#4f6b57}.ok{color:#39ff88}.bad{color:#ff5a5a}</style></head><body>"
               "<h1>&#9673; SGP SGPCARD NODE</h1><table>");
  h += "<tr><td class=k>STATUS</td><td class=" + String(live ? "ok>STREAMING" : "bad>WAITING FOR PC") + "</td></tr>";
  h += "<tr><td class=k>PC</td><td>" + (pcPort ? pcIP.toString() + ":" + pcPort : String("—")) + "</td></tr>";
  h += "<tr><td class=k>CSI RATE</td><td>" + String(csiRateHz, 1) + " Hz</td></tr>";
  h += "<tr><td class=k>FRAMES</td><td>" + String(framesSent) + " sent · " + String(framesDropped) + " dropped</td></tr>";
  h += "<tr><td class=k>MIC</td><td>" + String(micOk ? "OK · rms " + String(micRms) : "off") + "</td></tr>";
  h += "<tr><td class=k>CHANNEL</td><td>" + String(AP_CHANNEL) + "</td></tr>";
  h += "<tr><td class=k>FIRMWARE</td><td>" FW_VERSION "</td></tr></table>";
  h += F("<p class=k>Run <b>SGP SGPCard Command</b> on your PC — it does all the processing and the 3D view.</p></body></html>");
  web.send(200, "text/html", h);
}

// ═══════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n" FW_VERSION);
  pinMode(PIN_SENSOR_CE, OUTPUT); digitalWrite(PIN_SENSOR_CE, HIGH); delay(80);  // power the sensor rail
  led.begin(); led.setBrightness(120); led.clear(); led.show();
  ledColor(0, 0, 60);

  csiQ = xQueueCreate(48, sizeof(CsiFrame));

  apName = "SGP_SGPCARD_" + String((uint16_t)(ESP.getEfuseMac() & 0xFFFF), HEX);
  WiFi.onEvent(onWifiEvent);
#if STATION_MODE
  // multi-card: join a shared WiFi so the PC can reach every card by IP
  WiFi.mode(WIFI_STA);
  WiFi.begin(STA_SSID, STA_PASS);
  Serial.printf("Joining \"%s\" …\n", STA_SSID);
  for (int i = 0; i < 80 && WiFi.status() != WL_CONNECTED; i++) {
    ledColor(40, 20, 0); delay(150); ledColor(0, 0, 40); delay(150);
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("STA ip %s  ->  add this IP in the PC \"Nodes\" panel\n", WiFi.localIP().toString().c_str());
    esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);
  } else {
    Serial.println("STA join failed — falling back to AP mode");
    WiFi.mode(WIFI_AP);
    WiFi.softAP(apName.c_str(), nullptr, AP_CHANNEL);
    esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_BW_HT20);
  }
#else
  // single card: be our own access point and let the PC join us
  WiFi.mode(WIFI_AP);
  WiFi.softAP(apName.c_str(), nullptr, AP_CHANNEL);
  esp_wifi_set_bandwidth(WIFI_IF_AP, WIFI_BW_HT20);
#endif
  WiFi.setSleep(WIFI_PS_NONE);
  csiStart();

  udp.begin(UDP_PORT);
  web.on("/", handleRoot);
  web.onNotFound(handleRoot);
  web.begin();

  xTaskCreatePinnedToCore(micTask, "mic", 6144, NULL, 1, NULL, 0);

  Serial.printf("AP \"%s\" ch%d  ->  PC: join it, then run SGP SGPCard Command\n", apName.c_str(), AP_CHANNEL);
  beep(2000, 60); beep(2800, 80);
}

void loop() {
  web.handleClient();

  // 1) PC packets: "SD" = discovery probe (reply hello, don't stream); anything
  //    else = a normal ping, so remember who to stream CSI to.
  int sz = udp.parsePacket();
  while (sz > 0) {
    uint8_t tmp[64]; int n = udp.read(tmp, sizeof(tmp));
    if (n >= 2 && tmp[0] == 'S' && tmp[1] == 'D') {
      sendHello(udp.remoteIP(), udp.remotePort());       // "I'm an SGP Card Mini"
    } else {
      pcIP = udp.remoteIP(); pcPort = udp.remotePort();
      if (millis() - pcLastSeen > PC_TIMEOUT_MS) { beep(2600, 40); Serial.printf("[PC] streaming to %s:%u\n", pcIP.toString().c_str(), pcPort); }
      pcLastSeen = millis(); pingsRx++;
    }
    sz = udp.parsePacket();
  }

  // 2) drain CSI queue → PC
  bool live = pcPort && (millis() - pcLastSeen < PC_TIMEOUT_MS);
  CsiFrame f;
  while (xQueueReceive(csiQ, &f, 0) == pdTRUE) {
    if (live) sendFrame(f);
  }

  // 3) stats + status LED (blue = waiting, green pulse = streaming)
  static unsigned long tStat = 0; static uint32_t lastSent = 0;
  if (millis() - tStat >= 1000) {
    csiRateHz = (framesSent - lastSent) * 1000.0f / (millis() - tStat);
    lastSent = framesSent; tStat = millis();
    if (live) ledColor(0, 40 + (millis() / 1000 % 2) * 60, 10); else ledColor(0, 0, 50);
  }
}
