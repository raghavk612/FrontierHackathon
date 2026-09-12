// Turns what the other person just said into the words most likely to answer it.
//
// Three decisions worth defending:
//
// Local rules, not a language model. A model call costs a network round trip in the
// middle of a conversation, fails when the wifi does, and makes this a chatbot wearing a
// speech board. Rules are instant, deterministic, testable, and run on the same machine
// as the tracker.
//
// Suggestions never repeat anything in the top row. Yes, no, help and pain are already
// one dwell away at all times, and duplicating them would waste the suggestion slots on
// the one thing the board is already fast at. What a yes/no board cannot give you quickly is everything
// between yes and no - "a little", "worse than yesterday", "not yet" - so that is what
// these slots are for.
//
// Only questions trigger a change. Speech recognition rarely supplies a question mark,
// so this looks for a leading wh-word or auxiliary verb instead of punctuation.

export type Suggestion = { label: string; speech: string }

export type SuggestionSet = {
  words: Suggestion[]
  /** Shown to the user so the board never rearranges for a reason they cannot see. */
  because: string
}

// Spoken forms, so a two-word tile can say a whole sentence.
const SPEECH: Record<string, string> = {
  'A little': 'A little',
  'A lot': 'A lot',
  Worse: 'It is worse',
  Better: 'It is better',
  Same: 'About the same',
  'Not yet': 'Not yet',
  Later: 'Later, please',
  Now: 'Now, please',
  Nothing: 'Nothing right now',
  'Thank you': 'Thank you',
  Please: 'Yes, please',
  Maybe: 'Maybe',
  'Not sure': 'I am not sure',
  Nobody: 'Nobody right now',
  Water: 'I want water',
  Food: 'I want something to eat',
  'Not hungry': 'I am not hungry',
  Bathroom: 'I need the bathroom',
  Blanket: 'I want a blanket',
  'Sit up': 'I want to sit up',
  'Lie down': 'I want to lie down',
  'Turn me': 'Please turn me',
  Ice: 'Ice, please',
  Straw: 'A straw, please',
  Tired: 'I am tired',
  Cold: 'I am cold',
  Hot: 'I am hot',
  Scared: 'I am scared',
  Okay: 'I am okay',
  Mom: 'I want my mom',
  Dad: 'I want my dad',
  Nurse: 'I want the nurse',
  Doctor: 'I want the doctor',
  Friend: 'I want my friend',
  Here: 'It hurts here',
  Soon: 'Soon',
  Stay: 'Please stay',
}

type Rule = {
  name: string
  match: RegExp
  words: string[]
  because: string
  /**
   * True for rules broad enough to fire on almost any question. "Do you want a drink"
   * contains "want", so without this the open-ended rule would take half the slots away
   * from the rule that actually understood the question. Generic rules only apply when
   * nothing specific matched.
   */
  generic?: boolean
}

const RULES: Rule[] = [
  {
    name: 'pain',
    match: /\b(pain|painful|hurt|hurts|hurting|sore|ache|aches|aching)\b/,
    words: ['A little', 'A lot', 'Worse', 'Better', 'Same', 'Here'],
    because: 'about pain',
  },
  {
    name: 'drink',
    match: /\b(drink|drinks|drank|thirsty|thirst|water|juice|tea|coffee|sip|fluids)\b/,
    words: ['Water', 'Ice', 'Straw', 'A little', 'Later', 'Nothing'],
    because: 'about drinking',
  },
  {
    name: 'eat',
    match: /\b(eat|eaten|eating|ate|hungry|hunger|food|meal|breakfast|lunch|dinner|snack)\b/,
    words: ['Food', 'Not hungry', 'A little', 'Water', 'Later', 'Nothing'],
    because: 'about eating',
  },
  {
    name: 'bathroom',
    match: /\b(bathroom|toilet|restroom|bedpan|pee)\b/,
    words: ['Bathroom', 'Now', 'Later', 'Not yet', 'Please', 'Nothing'],
    because: 'about the bathroom',
  },
  {
    name: 'temperature',
    match: /\b(cold|hot|warm|warmer|cooler|chilly|freezing|temperature|thermostat|heat|heating|blanket|blankets)\b/,
    words: ['Cold', 'Hot', 'Blanket', 'Okay', 'A little', 'A lot'],
    because: 'about temperature',
  },
  {
    // Bare "turn" and "move" are too loose - "turn the heat up" is not about the bed - so
    // these only count when they are being done to a person.
    name: 'position',
    match: /\b(sit|sitting|lie|lying|lay|reposition|position|comfortable|pillow|(turn|roll|move|prop) (me|you|him|her|over|up))\b/,
    words: ['Sit up', 'Lie down', 'Turn me', 'Okay', 'Later', 'Please'],
    because: 'about moving you',
  },
  {
    name: 'people',
    match: /\b(who|visitor|visitors|visit|family|mom|mum|mother|dad|father|nurse|doctor|friend|call)\b/,
    words: ['Mom', 'Dad', 'Nurse', 'Doctor', 'Friend', 'Nobody'],
    because: 'about people',
  },
  {
    name: 'sleep',
    match: /\b(sleep|slept|sleeping|tired|rest|resting|nap|awake|night)\b/,
    words: ['Tired', 'Okay', 'Not yet', 'Later', 'Blanket', 'Stay'],
    because: 'about rest',
  },
  {
    name: 'feeling',
    match: /\b(feel|feels|feeling|alright|all right|scared|worried|upset)\b/,
    words: ['Okay', 'Tired', 'Scared', 'Cold', 'A little', 'Better'],
    because: 'about how you feel',
  },
  {
    name: 'timing',
    match: /\b(when|soon|ready)\b/,
    words: ['Now', 'Soon', 'Later', 'Not yet', 'Okay', 'Please'],
    because: 'about timing',
  },
  {
    name: 'wants',
    match: /\b(want|wants|need|needs|anything|something|get you|bring you)\b/,
    words: ['Water', 'Food', 'Bathroom', 'Blanket', 'Nothing', 'Please'],
    because: 'about what you want',
    generic: true,
  },
]

// Leading auxiliaries and wh-words. Recognisers drop question marks constantly, so the
// opening word is the reliable signal that something was asked.
const QUESTION_OPENER =
  /^(are|is|am|do|does|did|can|could|will|would|should|shall|have|has|had|was|were|may|might|want|what|which|who|whom|whose|where|when|why|how|any|anything|ready|tell me)\b/

/** Generic nuance for a question the rules do not recognise. */
const FALLBACK: Suggestion[] = ['Maybe', 'Not sure', 'A little', 'Later', 'Please', 'Thank you'].map((label) => ({
  label,
  speech: SPEECH[label] ?? label,
}))

export function isQuestion(heard: string) {
  const text = heard.trim().toLowerCase()
  if (!text) return false
  if (text.endsWith('?')) return true
  return QUESTION_OPENER.test(text)
}

/**
 * Words to offer in response to `heard`, best first, or null if nothing was asked.
 * `slots` is how many tiles the board has free for suggestions.
 */
export function suggestFor(heard: string, slots: number): SuggestionSet | null {
  if (slots <= 0) return null
  const text = heard.trim().toLowerCase()
  if (!text || !isQuestion(text)) return null

  const hits = RULES.filter((rule) => rule.match.test(text))
  const specific = hits.filter((rule) => !rule.generic)
  const matched = specific.length > 0 ? specific : hits

  // Interleave the matched rules rather than draining the first one. "Are you cold, or
  // do you want a drink?" should offer both, not six ways to say cold.
  const chosen: string[] = []
  for (let depth = 0; depth < 6 && chosen.length < slots; depth += 1) {
    for (const rule of matched) {
      const word = rule.words[depth]
      if (word && !chosen.includes(word) && chosen.length < slots) chosen.push(word)
    }
  }

  for (const suggestion of FALLBACK) {
    if (chosen.length >= slots) break
    if (!chosen.includes(suggestion.label)) chosen.push(suggestion.label)
  }

  const because =
    matched.length === 0
      ? 'a question'
      : matched.length === 1
        ? matched[0].because
        : `${matched[0].because} and ${matched[1].because.replace(/^about /, '')}`

  return { words: chosen.map((label) => ({ label, speech: SPEECH[label] ?? label })), because }
}
