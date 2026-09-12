# VividVision — judge deck

Mapped to the four announced criteria: **Product execution · Technical feasibility ·
Viability beyond the prototype · Business appeal and defensibility.**

Every number is printed by the test suites in this repo (`npm run test:gaze`,
`test:gesture`, `test:suggest`, `test:server`, `test:position`, `test:camera`) or is a
published price. Nothing is estimated.

> **Read the AI-slop rule at the bottom before you present.** If a judge points at any
> part of this and nobody on the team can explain it, you lose points. The crib sheet at
> the end exists for exactly that.

---

## Slide 1 — Title

**VividVision**
A speech board you drive with your head. Any laptop. No hardware.

---

## Slide 2 — The problem

- A person with cerebral palsy, ALS, or locked-in syndrome may have full comprehension and no way to speak.
- The standard answer is an eye-gaze speech device. **Tobii Dynavox units run about $10,000.**
- Funding runs through insurance as durable medical equipment: a speech-language pathologist assessment, prior authorisation, months of waiting.
- **The people who need this most are the ones who can't get funded for it.**

> *Speaker note:* Say the last line slowly, then stop talking.

---

## Slide 3 — Product execution: what we actually built

Not a mockup. A working board, driven end to end by head movement.

- **Compose and speak with your head alone.** Words build a sentence; **Speak** and **Delete** are dwell targets in the sentence bar, so nothing requires a hand.
- **Delete is word-level.** Select Delete, then aim at any word in the sentence to remove it — not just a backspace.
- **Nod and shake** answer Yes and No with no aiming at all.
- **Live captions** of whoever is talking near the laptop, so the user can read what they missed.
- **Conversation-aware suggestions.** The board listens to the question and offers words that fit it.
- **A caregiver beacon.** Urgent words drive a physical light on a Particle Argon over Bluetooth.
- **Rest zones** on both edges so you can look away without saying something by accident.

> *Speaker note:* The delete-by-word detail is worth calling out. It is the difference between a demo and a tool.

---

## Slide 4 — Technical feasibility: why this approach is the sound one

This is the strongest slide. It is a measurement, not a design opinion.

- We built eye tracking first. It topped out around **23–28% error** — unusable on a 12-word board.
- **The reason is physical.** An eyeball rotates through its full gaze range while the pupil travels only about **12 mm** across the face.
- At a normal 55 cm sitting distance, **one pixel of iris-landmark jitter becomes roughly 1.52° of inferred gaze.** The eye is a short lever arm that multiplies camera noise.
- **Head pose has no lever arm.** The head is large, its landmarks are stable, a pixel of noise stays a pixel of noise.
- So head pointing is not a fallback. It is what the geometry says to do on a consumer webcam.

**And it clears the bar it has to clear:**

- Accuracy budget is **half the gap between neighbouring word centres**: **7.2% across, 18.3% down.**
- Measured: **0.5–1.0% across, 0.6–0.7% down.** 4 of 4 test conditions land on the intended word.
- Holds from **640×480 to 1920×1080** — it does not need a good camera.
- **The layout is derived from the budget.** Two rows of six instead of three rows of four, because vertical accuracy is about half of horizontal: that took vertical tolerance from **11.7% to 17.5%** at zero cost in vocabulary.

> *Speaker note:* Offer to run `npm run test:gaze` in front of them. It prints these lines live.

---

## Slide 5 — Viability beyond the prototype

What is already production-shaped, and what honestly is not.

**Already built for the real world:**
- **Six test suites pass**, including the ugly camera cases: permission denied, tab backgrounded and resumed, camera muted, stream ended, cleanup.
- **Degrades gracefully at every layer.** No camera → the board is still clickable. No internet → suggestions fall back to local rules. No beacon → everything else is unchanged.
- **Nothing to operate.** Tracking is entirely client-side; the base configuration has no server to run, scale, or pay for.

**What it would take next, in order:**
1. **Clinical validation** with speech-language pathologists — the accuracy number is geometric, not yet clinical.
2. **Per-user vocabulary and persistence** — real AAC users need their own words, saved.
3. **Offline packaging** as a PWA so a hospital room with bad wifi is a non-event.
4. **Accessibility audit** — screen-reader parity and switch-access as an alternative input.
5. **Multi-language**, since the suggestion rules are currently English-only.

> *Speaker note:* Naming what is missing is what makes the rest of the slide credible. Do not skip this.

---

## Slide 6 — Business appeal and defensibility

**Who uses it and who pays:**
- Schools with 1:1 laptop programmes, hospital rehab and ICU step-down units, and home caregivers. **They already own the hardware.**
- The beacon is the only physical cost, about **$30**, and it is optional.

**Why they would choose it over the alternative:**
- **$10,000 device → a web page.** Marginal cost per additional user is effectively zero.
- **Same-day, not same-quarter.** No assessment, no prior authorisation, no procurement. A family opens a link.

**What makes it hard to replicate:**
- The accuracy work is the moat, not the UI. **The lever-arm finding, the error budget, and the layout derived from it** are the reason this is usable where a naive webcam tracker is not. Anyone can render twelve tiles; landing inside 1% of the one you meant is the hard part.
- **A calibration corpus compounds.** Every session produces labelled head-pose-to-screen data, which is exactly what improves the model.
- **Incumbents are structurally unable to follow.** Tobii's revenue is the $10,000 device and the reimbursement pathway around it. A free browser tool cannibalises their own funnel.

**The honest boundary:**
- This is not a Tobii replacement for everyone. Full locked-in syndrome still needs eye gaze. **For the people who cannot fund a device at all, it is the difference between a voice and no voice.**

---

## Slide 7 — Demo

1. Open the page. Welcome card explains it in three steps.
2. Enable camera, calibrate — about a minute.
3. Compose a sentence with your head. Delete a word. Speak it.
4. Select **Help**, and the beacon lights up across the room.

> *Speaker note:* Put the Argon on a power bank on the far side of the table before you start.

---

## AI-slop insurance: be able to explain any of this

The rule is explicit — the tool may write it, but you have to own it. Be ready for
"point at something and explain it." Short answers you should be able to give:

| If a judge points at… | You say |
| --- | --- |
| **Head tracking** | MediaPipe Face Landmarker gives 3D face landmarks; we take head pose from them, not gaze. |
| **Calibration** | 16 points. We fit a model from head-pose features to screen position, then check it on 5 held-out points. |
| **Why not eye tracking** | The 12 mm lever arm. One pixel of iris noise is 1.52° of gaze at 55 cm. |
| **The One Euro filter** | A low-pass filter whose cutoff rises with speed — steady when you hold still, responsive when you move. |
| **Nearest-word targeting** | The cursor doesn't have to be inside a tile; we pick the closest tile centre. That's why the budget is half the gap between centres. |
| **Dwell** | Hold on a word and a timer fills. There's a cooldown after each pick so one dwell can't fire twice. |
| **Suggestions** | Local rules by default, matching question type to word sets. OpenAI is optional and off without a key. |
| **The beacon** | Web Bluetooth writes one byte, 0/1/2, to a characteristic on a Particle Argon. The firmware turns that into a blink pattern on pin A0. |
| **Why Bluetooth not wifi** | Gen 3 Particle boards only join WPA2-Personal. Campus wifi is enterprise or captive-portal, so neither works. |
| **Tests** | Six suites. The gaze one simulates tracker noise and checks we still land on the intended word. |

**Rule for the expo:** if neither of you can explain a feature in one sentence, cut it
from the demo rather than let a judge find it.
