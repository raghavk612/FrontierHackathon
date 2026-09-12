# ClearSpeak

A head-pointing communication board with optional captions and OpenAI reply suggestions.

## Run locally

Use Node.js 22+ and run `npm ci`, then `npm run dev`. Open the localhost URL printed in the terminal.

Copy `.env.example` to `.env.local` and set `OPENAI_API_KEY` there. Do not add a `VITE_` prefix: this key is used only by the local server. The private file is ignored by Git. `OPENAI_MODEL` defaults to `gpt-4.1-mini` and can be changed to a compatible model available to your account. The server reads this file for each request, so saving a key does not require a restart.

The app uses the [Responses API](https://developers.openai.com/api/docs/guides/text) and [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) to produce five optional replies. Final transcript text and up to six recent conversation turns are sent to OpenAI when the suggestion toggle is enabled. Requests use `store: false`; this is not a promise of zero provider retention. Camera frames are processed locally, while captions use the browser's speech recognition service. Replies are never spoken until the user selects one. Local rule-based replies remain available when the key is missing or the service fails.

## Camera and accessibility

Start in a comfortable position with your face visible, looking toward the middle of the board for the first second. The uncalibrated pointer uses the nose relative to the face and the user's starting pose, rather than the middle of the camera image. Calibration learns the user's movement from that seating position. If you shift, choose **Reset pointing position**, look at the board centre, and wait about three seconds. For a large seating change, calibrate again. Tracking still needs enough visible facial landmarks and light; this does not guarantee accuracy for every posture or movement range.

**Stop camera** releases the camera tracks and detector, including during startup. Clicking words and using captions are independent of the camera. The Talk and Tracking pages group large controls across the right side on desktop. Details may scroll; smaller windows and enlarged text retain scrolling rather than hiding controls.

## Checks

- `npm run build`
- `npm run test:server`
- `npm run test:position`
- `npm run test:gaze`
- `npm run test:gesture`
- `npm run test:suggest`

Server tests use mocked responses and no paid API calls. Position tests cover camera-space translation and pointer recentering. Camera startup/stop, real tracking, captions, and accessibility should also be checked with the intended user and their device.

## Serving a built preview

Run `npm run build` then `npm run preview`. The Vite preview includes the same local suggestion endpoint. A static `dist/` upload alone cannot provide AI suggestions: a production deployment needs an authenticated backend with server-side secrets and appropriate per-user limits. This development server binds to localhost and must not be exposed publicly as a production API.

## Phones and tablets

Portrait phones use a three-column board and a persistent camera dock; calibration is under Tracking. The camera stays mounted through resizing. Rotating keeps the stream alive but resets the pointing model because the screen mapping has changed. Front-camera capture uses modest preferred resolution and retries unsupported constraints. Returning from another app resumes the stream where the browser permits; an ended stream requires tapping Enable camera again. Tracking pauses while the camera is muted or the page is hidden, so it cannot accidentally select words on return.

Camera access on a separate phone needs an **HTTPS** deployment with the backend. The Mac's `localhost` URL is not reachable as that same address from the phone. HTTP over a LAN IP does not satisfy the browser's secure-context requirement. Browser/OS camera suspension cannot be prevented by a webpage. Real iOS Safari and Android Chrome camera testing is still necessary; viewport checks and mocked lifecycle tests do not emulate the camera hardware.

Run `npm run test:camera` for constraint fallback and stream lifecycle tests.

### GitHub Pages deployment

The public site is served from the compiled `gh-pages` branch, not the source on `main`.
Build with `VITE_STATIC_HOST=true npm run build -- --base=/FrontierHackathon/`, then publish the contents of `dist/` (plus an empty `.nojekyll` file) to `gh-pages`. Never copy `.env.local` into that branch. Changes pushed only to `main` do not update the public site.

Pages supports the camera, calibration, speech board, and browser-supported captions over HTTPS. It cannot run the private OpenAI middleware. The static build uses local reply suggestions and disables the OpenAI switch; `npm run dev` retains the private server-backed OpenAI connection. Hosting AI replies publicly requires a server deployment with the API key stored as a private environment variable.

### Hosted OpenAI backend

Deploy the `render.yaml` Blueprint from `main` on Render. It creates a free Docker web service and asks for `OPENAI_API_KEY` privately. The container copies only the server code, never `.env.local`. `ALLOWED_ORIGINS` permits the GitHub Pages origin. `/health` reports liveness and `/api/suggestions/status` reports configuration without exposing credentials. Requests have size, concurrency, and per-process rate limits; CORS is a browser restriction, not user authentication. This is a public demo endpoint, so usage consumes the server owner's OpenAI credits.

After deployment, build Pages with `VITE_STATIC_HOST=true VITE_API_BASE_URL=https://YOUR-SERVICE.onrender.com npm run build -- --base=/FrontierHackathon/` and publish `dist/` to `gh-pages`. The frontend enables OpenAI only when the backend URL is configured. Keep local replies available if the host is waking up or the API fails. Free hosting can take time to wake after inactivity.

For a local standalone API, run `npm run start:api` (port 3001 by default). Set `PORT` to override it; set `ALLOWED_ORIGINS` to a comma-separated list of approved frontend origins.
