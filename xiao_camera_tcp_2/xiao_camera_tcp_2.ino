// xiao_camera_tcp.ino
// Streams MJPEG frames over raw TCP to a Python server.
//
// ── SETUP INSTRUCTIONS ────────────────────────────────────────────────────────
// 1. SERVER_HOST points to parkingCuFinal.gleeze.com (the public hostname).
//    If you're on the same LAN as the server, replacing this with the server's
//    local IP (e.g. "192.168.1.42") is more reliable — avoids DNS lookups on
//    every reconnect and removes the gleeze.com DDNS as a failure point.
//
// 2. Make sure both the XIAO and the server are reachable on port 5001.
//
// 3. Run parking_server.py on your laptop BEFORE powering the board.
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
#include <WiFi.h>

// ── WiFi credentials ──────────────────────────────────────────────────────────
const char* SSID     = "CMU PSK";      // ← change to your network name
const char* PASSWORD = "AbqhzHkR965";  // ← change to your WiFi password

// ── Server address ────────────────────────────────────────────────────────────
// Using the public hostname. If on the same LAN, replace with the server's
// local IP for faster, more reliable reconnects (no DNS dependency).
const char*    SERVER_HOST = "parkingCuFinal.gleeze.com";
const uint16_t SERVER_PORT = 5001;

// ── Reconnect interval (ms) ───────────────────────────────────────────────────
const unsigned long RECONNECT_INTERVAL = 3000;

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

WiFiClient tcpClient;
unsigned long lastConnectAttempt = 0;
bool streaming = false;

// ── Camera init ───────────────────────────────────────────────────────────────
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
        Serial.println("PSRAM found - VGA mode");
    } else {
        config.frame_size   = FRAMESIZE_QVGA;  // 320×240
        config.jpeg_quality = 12;
        config.fb_count     = 1;
        config.fb_location  = CAMERA_FB_IN_DRAM;
        Serial.println("No PSRAM - QVGA mode");
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

// ── Connect to server ─────────────────────────────────────────────────────────
bool connectToServer() {
    Serial.printf("Connecting to %s:%d ...\n", SERVER_HOST, SERVER_PORT);
    if (tcpClient.connect(SERVER_HOST, SERVER_PORT)) {
        // FIX: Disable Nagle algorithm — reduces TCP buffering latency and
        // helps detect stale connections faster under congested networks.
        tcpClient.setNoDelay(true);
        Serial.println("Connected to server!");
        streaming = true;
        return true;
    }
    Serial.println("Connection failed, will retry...");
    return false;
}

// ── Send one frame: [4-byte big-endian length][jpeg bytes] ───────────────────
bool sendFrame() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Frame capture failed");
        return true;
    }

    uint32_t len = (uint32_t)fb->len;

    uint8_t header[4];
    header[0] = (len >> 24) & 0xFF;
    header[1] = (len >> 16) & 0xFF;
    header[2] = (len >>  8) & 0xFF;
    header[3] = (len      ) & 0xFF;

    bool ok = true;

    if (tcpClient.write(header, 4) != 4) {
        ok = false;
    }

    if (ok) {
        size_t   remaining = fb->len;
        uint8_t* ptr       = fb->buf;
        const size_t CHUNK = 1024;

        while (remaining > 0 && ok) {
            size_t toSend = (remaining > CHUNK) ? CHUNK : remaining;

            // FIX: Use the actual bytes written rather than assuming all-or-nothing.
            // write() can return a partial count on a congested connection.
            // Only treat a zero-byte write (blocked/dead socket) as a hard failure.
            size_t written = tcpClient.write(ptr, toSend);
            if (written == 0) {
                ok = false;
                break;
            }
            ptr       += written;
            remaining -= written;
        }
    }

    esp_camera_fb_return(fb);

    if (!ok) {
        Serial.println("Send failed - server disconnected");
        tcpClient.stop();
        streaming = false;
    }

    return ok;
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    unsigned long t = millis();
    while (!Serial && millis() - t < 3000);

    Serial.println("\nXIAO ESP32S3 - TCP Camera Stream");
    Serial.println("==================================");

    if (!initCamera()) {
        Serial.println("Camera init failed - halting");
        while (1) delay(1000);
    }
    Serial.println("Camera OK");

    Serial.printf("Connecting to WiFi: %s", SSID);
    WiFi.begin(SSID, PASSWORD);

    unsigned long wt = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - wt > 15000) {
            Serial.println("\nWiFi timeout - restarting");
            ESP.restart();
        }
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.print("WiFi connected. IP: ");
    Serial.println(WiFi.localIP());

    // FIX: Disable WiFi modem sleep. The ESP32 Arduino core enables modem
    // sleep by default, which periodically cuts the radio to save power and
    // causes intermittent TCP disconnects under continuous streaming load.
    WiFi.setSleep(false);
    Serial.println("WiFi modem sleep disabled");

    Serial.printf("Target server: %s:%d\n", SERVER_HOST, SERVER_PORT);
    Serial.println("==================================");

    connectToServer();
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
    // Check WiFi first — reconnect at the IP layer before attempting TCP
    if (WiFi.status() != WL_CONNECTED) {
        streaming = false;
        tcpClient.stop();
        Serial.println("WiFi lost - reconnecting...");
        WiFi.reconnect();
        unsigned long wt = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - wt < 10000) {
            delay(500);
            Serial.print(".");
        }
        Serial.println();
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("WiFi reconnect failed, restarting...");
            ESP.restart();
        }
        Serial.println("WiFi reconnected");
        WiFi.setSleep(false);  // re-apply after reconnect
        return;
    }

    if (!streaming || !tcpClient.connected()) {
        streaming = false;
        unsigned long now = millis();
        if (now - lastConnectAttempt >= RECONNECT_INTERVAL) {
            lastConnectAttempt = now;
            connectToServer();
        }
        delay(100);
        return;
    }

    sendFrame();
}
