// xiao_camera_tcp.ino
// Streams MJPEG frames over raw TCP to a Python server.
//
// Duty cycle
// ──────────
//   1. Wake from deep sleep (or power-on)
//   2. Init camera + connect WiFi + connect to server  (~2-3 s)
//   3. Stream frames continuously for SEND_DURATION_MS  (10 s)
//   4. Disconnect, deep-sleep for SLEEP_DURATION_S      (50 s)
//   5. Repeat — total cycle ≈ 60 s
//
// Deep sleep resets all RAM; setup() runs fresh on every wake.
//
// ── SETUP INSTRUCTIONS ────────────────────────────────────────────────────────
// 1. Set SSID / PASSWORD to your WiFi network.
// 2. Set SERVER_HOST to your server's IP or hostname.
//    For local LAN use your laptop's IP (e.g. "192.168.1.42").
//    For remote access use the public hostname (parkingCuFinal.gleeze.com).
// 3. Run parking_server.py before powering the board.
// 4. Only port 5001 (TCP) needs to be forwarded for the camera stream.
//
// Frame format sent to server (per frame):
//   [4 bytes big-endian uint32] = JPEG byte length N
//   [N bytes]                   = raw JPEG data
//
// Arduino IDE settings:
//   Tools -> Board            -> ESP32S3 Dev Module
//   Tools -> PSRAM            -> OPI PSRAM
//   Tools -> Flash Size       -> 8MB
//   Tools -> Partition Scheme -> Huge APP (3MB No OTA)
//   Tools -> USB CDC On Boot  -> Enabled
// ─────────────────────────────────────────────────────────────────────────────

#include "esp_camera.h"
#include "esp_sleep.h"
#include <WiFi.h>

// ── WiFi credentials ──────────────────────────────────────────────────────────
const char* SSID     = "CMU PSK";      // ← change to your network name
const char* PASSWORD = "AbqhzHkR965";  // ← change to your WiFi password

// ── Server address ────────────────────────────────────────────────────────────
const char*    SERVER_HOST = "parkingCuFinal.gleeze.com";
const uint16_t SERVER_PORT = 5001;

// ── Duty-cycle timing ─────────────────────────────────────────────────────────
const unsigned long SEND_DURATION_MS = 10000;   // stream for 10 s each wake
const uint64_t      SLEEP_DURATION_S = 50;      // then sleep for 50 s

// ── WiFi timeout ──────────────────────────────────────────────────────────────
const unsigned long WIFI_TIMEOUT_MS  = 15000;

// ── XIAO ESP32S3 Sense camera pins — do not change ───────────────────────────
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ── Globals ───────────────────────────────────────────────────────────────────
WiFiClient tcpClient;

// ─────────────────────────────────────────────────────────────────────────────
// Camera init
// ─────────────────────────────────────────────────────────────────────────────
bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        config.frame_size   = FRAMESIZE_VGA;   // 640×480
        config.jpeg_quality = 10;
        config.fb_count     = 2;
        config.fb_location  = CAMERA_FB_IN_PSRAM;
        Serial.println("PSRAM found — VGA mode");
    } else {
        config.frame_size   = FRAMESIZE_QVGA;  // 320×240
        config.jpeg_quality = 12;
        config.fb_count     = 1;
        config.fb_location  = CAMERA_FB_IN_DRAM;
        Serial.println("No PSRAM — QVGA mode");
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }

    sensor_t* s = esp_camera_sensor_get();
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 0);
    s->set_ae_level(s, 0);
    s->set_gain_ctrl(s, 1);
    s->set_agc_gain(s, 0);
    s->set_gainceiling(s, (gainceiling_t)0);
    s->set_bpc(s, 0);
    s->set_wpc(s, 1);
    s->set_raw_gma(s, 1);
    s->set_lenc(s, 1);
    s->set_hmirror(s, 0);
    s->set_vflip(s, 0);
    s->set_dcw(s, 1);
    s->set_colorbar(s, 0);

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Send one frame: [4-byte big-endian length][jpeg bytes]
// Returns true on success, false if the connection dropped.
// ─────────────────────────────────────────────────────────────────────────────
bool sendFrame() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Frame capture failed — skipping");
        return true;   // not a connection error, keep trying
    }

    uint32_t len = (uint32_t)fb->len;

    // 4-byte big-endian length header
    uint8_t header[4] = {
        (uint8_t)((len >> 24) & 0xFF),
        (uint8_t)((len >> 16) & 0xFF),
        (uint8_t)((len >>  8) & 0xFF),
        (uint8_t)((len      ) & 0xFF)
    };

    bool ok = (tcpClient.write(header, 4) == 4);

    if (ok) {
        size_t   remaining = fb->len;
        uint8_t* ptr       = fb->buf;
        const size_t CHUNK = 1024;

        while (remaining > 0 && ok) {
            size_t toSend = (remaining > CHUNK) ? CHUNK : remaining;
            if (tcpClient.write(ptr, toSend) != toSend) {
                ok = false;
            }
            ptr       += toSend;
            remaining -= toSend;
        }
    }

    esp_camera_fb_return(fb);

    if (!ok) {
        Serial.println("Send failed — server disconnected");
        tcpClient.stop();
    }

    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Go to deep sleep for SLEEP_DURATION_S seconds.
// Execution will resume from setup() after wakeup.
// ─────────────────────────────────────────────────────────────────────────────
void goToSleep() {
    Serial.printf("Entering deep sleep for %llu s...\n", SLEEP_DURATION_S);
    Serial.flush();
    tcpClient.stop();
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
    esp_sleep_enable_timer_wakeup(SLEEP_DURATION_S * 1000000ULL);
    esp_deep_sleep_start();
    // never reached — deep sleep resets the chip
}

// ─────────────────────────────────────────────────────────────────────────────
// setup() — runs on every power-on AND every deep-sleep wakeup
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    unsigned long t = millis();
    while (!Serial && millis() - t < 3000);

    Serial.println("\nXIAO ESP32S3 — TCP Camera Stream");
    Serial.println("==================================");

    // ── Camera ──
    if (!initCamera()) {
        Serial.println("Camera init failed — sleeping and retrying next cycle");
        goToSleep();
    }
    Serial.println("Camera OK");

    // ── WiFi ──
    Serial.printf("Connecting to WiFi: %s", SSID);
    WiFi.begin(SSID, PASSWORD);
    unsigned long wt = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - wt > WIFI_TIMEOUT_MS) {
            Serial.println("\nWiFi timeout — sleeping and retrying next cycle");
            goToSleep();
        }
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("WiFi connected. IP: ");
    Serial.println(WiFi.localIP());

    // ── TCP connect ──
    Serial.printf("Connecting to server %s:%d ...\n", SERVER_HOST, SERVER_PORT);
    if (!tcpClient.connect(SERVER_HOST, SERVER_PORT)) {
        Serial.println("TCP connection failed — sleeping and retrying next cycle");
        goToSleep();
    }
    Serial.println("Connected to server!");
    Serial.printf("Streaming for %lu ms then sleeping for %llu s.\n",
                  SEND_DURATION_MS, SLEEP_DURATION_S);
    Serial.println("==================================");
}

// ─────────────────────────────────────────────────────────────────────────────
// loop() — stream frames for SEND_DURATION_MS, then deep sleep
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
    unsigned long streamStart = millis();

    while (millis() - streamStart < SEND_DURATION_MS) {
        if (!tcpClient.connected()) {
            Serial.println("Server disconnected mid-stream");
            break;
        }
        if (!sendFrame()) {
            break;   // sendFrame() already stopped the client
        }
    }

    Serial.printf("Streaming window complete (%lu ms elapsed).\n",
                  millis() - streamStart);
    goToSleep();
    // goToSleep() never returns — loop() is effectively called once per wake
}
