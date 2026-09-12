/*
 * VividVision caregiver alert beacon  --  Particle Argon
 *
 * The speech board gives someone a voice. This gives that voice reach.
 * A screen only talks to whoever is already looking at it, which is the one
 * thing a person who cannot move or speak can never arrange. When the board
 * says "I need help", this lights up wherever the caregiver actually is.
 *
 * Transport is Bluetooth LE, with USB serial as a fallback. Wi-Fi is not used:
 * Particle Gen 3 boards can only join WPA2-Personal networks, so university
 * and conference Wi-Fi (WPA2-Enterprise, or a captive portal) is unavailable
 * to them. Bluetooth also keeps the whole system working with no network at
 * all, which matters more in a hospital room than it does at a hackathon.
 *
 * Wiring: see WIRING.md. A0 drives the LED, returning through the single GND
 * pin. A second LED on A2 is optional; every alert level is legible from A0
 * alone, so a half-built breadboard still demonstrates the whole idea.
 *
 * Between each LED and its resistor there is a column that only exists so
 * two leads can meet. Any unused analog pin will do for that, and the choice
 * never reaches the firmware, because junction pins are deliberately never
 * configured or written. They stay high-impedance inputs sitting around
 * 1.3 V when their LED is lit, well inside the safe input range.
 */

#include "Particle.h"

SYSTEM_MODE(MANUAL);
SYSTEM_THREAD(ENABLED);

const pin_t LED_URGENT = A0;
const pin_t LED_REQUEST = A2;

enum Level { CLEAR = 0, REQUEST = 1, URGENT = 2 };

static Level level = CLEAR;
static unsigned long levelAt = 0;

/*
 * Alerts expire on their own. A light still burning after the moment has
 * passed teaches the caregiver to ignore it, which is worse than no light.
 */
static const unsigned long REQUEST_HOLD_MS = 6000;
static const unsigned long URGENT_HOLD_MS = 30000;

static const BleUuid SERVICE_UUID("f1e2d300-9a5b-4c7e-8d21-6b0f3a9c1e00");
static const BleUuid ALERT_UUID("f1e2d301-9a5b-4c7e-8d21-6b0f3a9c1e00");

static void onAlertWritten(const uint8_t *data, size_t len,
                           const BlePeerDevice &peer, void *context);

static BleCharacteristic alertCharacteristic(
    "alert", BleCharacteristicProperty::WRITE_WO_RSP | BleCharacteristicProperty::WRITE,
    ALERT_UUID, SERVICE_UUID, onAlertWritten, NULL);

/*
 * A pain alert outranks a request for water. Anything may clear an alert
 * explicitly, but an ordinary word must never quietly cancel an emergency.
 */
static void setLevel(Level next) {
  if (next == REQUEST && level == URGENT) {
    return;
  }
  level = next;
  levelAt = millis();
}

static void onAlertWritten(const uint8_t *data, size_t len,
                           const BlePeerDevice &peer, void *context) {
  if (len < 1) {
    return;
  }
  switch (data[0]) {
    case 0: setLevel(CLEAR); break;
    case 1: setLevel(REQUEST); break;
    case 2: setLevel(URGENT); break;
    default: break;
  }
}

void setup() {
  pinMode(LED_URGENT, OUTPUT);
  pinMode(LED_REQUEST, OUTPUT);
  digitalWrite(LED_URGENT, LOW);
  digitalWrite(LED_REQUEST, LOW);

  RGB.control(true);
  RGB.color(0, 40, 0);

  Serial.begin(9600);

  BLE.on();
  BLE.setDeviceName("VividVision");
  BLE.addCharacteristic(alertCharacteristic);

  /*
   * A 128-bit service UUID and a device name do not both fit in the 31-byte
   * advertising packet, so the name goes in the scan response. Browsers still
   * see both.
   */
  BleAdvertisingData advertisement;
  advertisement.appendServiceUUID(SERVICE_UUID);

  BleAdvertisingData scanResponse;
  scanResponse.appendLocalName("VividVision");

  BLE.advertise(&advertisement, &scanResponse);
}

/* Accepts '0', '1', '2' so the beacon can be tested from any serial monitor. */
static void readSerial() {
  while (Serial.available() > 0) {
    int byteIn = Serial.read();
    if (byteIn == '0') setLevel(CLEAR);
    else if (byteIn == '1') setLevel(REQUEST);
    else if (byteIn == '2') setLevel(URGENT);
  }
}

static void render() {
  unsigned long now = millis();

  if (level == REQUEST && now - levelAt > REQUEST_HOLD_MS) level = CLEAR;
  if (level == URGENT && now - levelAt > URGENT_HOLD_MS) level = CLEAR;

  if (level == URGENT) {
    /*
     * Fast, and with the two LEDs out of phase. A steady or synchronised
     * light reads as status; something switching back and forth reads as
     * alarm and catches peripheral vision from across a room. With only one
     * LED fitted this degrades to a 2.5 Hz blink, which still reads as alarm.
     */
    bool phase = (now % 400) < 200;
    digitalWrite(LED_URGENT, phase ? HIGH : LOW);
    digitalWrite(LED_REQUEST, phase ? LOW : HIGH);
    RGB.color(phase ? 255 : 60, 0, 0);
    return;
  }

  if (level == REQUEST) {
    /*
     * One slow pulse, and both LEDs together. Rate and rhythm carry the
     * difference rather than which bulb is lit, so a single LED on A0 says
     * everything the pair does and fitting the second one later changes
     * nothing in here.
     */
    bool phase = (now % 1800) < 420;
    digitalWrite(LED_URGENT, phase ? HIGH : LOW);
    digitalWrite(LED_REQUEST, phase ? HIGH : LOW);
    RGB.color(0, 0, phase ? 180 : 25);
    return;
  }

  digitalWrite(LED_URGENT, LOW);
  digitalWrite(LED_REQUEST, LOW);

  /* A slow green breath, so "idle" is distinguishable from "unplugged". */
  unsigned long cycle = now % 4000;
  unsigned long up = cycle < 2000 ? cycle : 4000 - cycle;
  RGB.color(0, 6 + (up * 22) / 2000, 0);
}

void loop() {
  readSerial();
  render();
}
