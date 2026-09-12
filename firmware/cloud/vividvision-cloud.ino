/*
 * VividVision caregiver alert beacon  --  Particle Argon, Wi-Fi build
 *
 * The speech board gives someone a voice. This gives that voice reach.
 * A screen only talks to whoever is already looking at it, which is the one
 * thing a person who cannot move or speak can ever arrange for themselves.
 * When the board says "I need help", this lights up wherever the caregiver is.
 *
 * This build reaches the beacon through Particle's cloud rather than over
 * Bluetooth, which changes what the system can claim. Bluetooth stops at
 * roughly ten metres and needs the caregiver in earshot anyway. A cloud round
 * trip has no range at all, so the light can sit at a nurses' station, in a
 * kitchen downstairs, or in another building, and the person using the board
 * does not have to be near anyone for it to work.
 *
 * The trade is a network dependency, which is why firmware/vividvision-beacon.ino
 * still exists. That one speaks Bluetooth and USB serial and needs no network.
 *
 * Wiring is identical either way: see WIRING.md. A0 drives the LED and returns
 * through the single GND pin. A second LED on A2 is optional, because every
 * alert level is legible from A0 alone.
 */

#include "Particle.h"

SYSTEM_MODE(AUTOMATIC);
SYSTEM_THREAD(ENABLED);

const pin_t LED_URGENT = A0;
const pin_t LED_REQUEST = A2;

enum Level { CLEAR = 0, REQUEST = 1, URGENT = 2 };

static Level level = CLEAR;
static unsigned long levelAt = 0;

/*
 * The status LED is worth more as a connection indicator than as decoration,
 * so the firmware only seizes it while an alert is actually running. Idle
 * therefore shows Particle's own breathing cyan, which is live proof to anyone
 * watching that the board is still talking to the cloud. A beacon that has
 * silently dropped off the network looks exactly like a beacon with nothing
 * to report, and that is the one failure this device must never hide.
 */
static bool rgbHeld = false;

/*
 * Alerts expire on their own. A light still burning after the moment has
 * passed teaches the caregiver to ignore it, which is worse than no light.
 */
static const unsigned long REQUEST_HOLD_MS = 6000;
static const unsigned long URGENT_HOLD_MS = 30000;

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

/*
 * Called by Particle's cloud. The argument is the alert level as text, so the
 * same beacon can be driven from the web app, from the Particle console, or
 * from a single curl command, which matters when you have one chance to show
 * a judge that the hardware is real.
 */
int onAlert(String command) {
  command = command.trim();
  if (command == "0") { setLevel(CLEAR); return CLEAR; }
  if (command == "1") { setLevel(REQUEST); return REQUEST; }
  if (command == "2") { setLevel(URGENT); return URGENT; }
  return -1;
}

void setup() {
  pinMode(LED_URGENT, OUTPUT);
  pinMode(LED_REQUEST, OUTPUT);
  digitalWrite(LED_URGENT, LOW);
  digitalWrite(LED_REQUEST, LOW);

  Serial.begin(9600);

  /*
   * Registered before the first connection completes, so the function is
   * callable the instant the device comes online rather than a beat later.
   */
  Particle.function("alert", onAlert);
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

static void holdRgb(bool wanted) {
  if (wanted == rgbHeld) return;
  RGB.control(wanted);
  rgbHeld = wanted;
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
    holdRgb(true);
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
    holdRgb(true);
    bool phase = (now % 1800) < 420;
    digitalWrite(LED_URGENT, phase ? HIGH : LOW);
    digitalWrite(LED_REQUEST, phase ? HIGH : LOW);
    RGB.color(0, 0, phase ? 180 : 25);
    return;
  }

  digitalWrite(LED_URGENT, LOW);
  digitalWrite(LED_REQUEST, LOW);
  holdRgb(false);
}

void loop() {
  readSerial();
  render();
}
