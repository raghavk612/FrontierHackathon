// Two things have to hold. Real questions have to surface a word that actually answers
// them, and ordinary conversation has to leave the board alone - rearranging tiles under
// someone who is aiming at one is worse than never rearranging at all.

import { isQuestion, suggestFor } from './suggest'

declare const process: { exit(code: number): never }

const SLOTS = 6

let failures = 0
function check(ok: boolean, line: string) {
  if (!ok) failures += 1
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${line}`)
}

console.log('answers a question with words that fit it')
const answers: [string, string][] = [
  ['Are you in pain?', 'A little'],
  ['Does it hurt more than yesterday', 'Worse'],
  ['Do you want something to drink', 'Water'],
  ['Are you thirsty', 'Water'],
  ['Have you eaten anything today', 'Food'],
  ['Do you need the bathroom', 'Bathroom'],
  ['Are you cold', 'Blanket'],
  ['Do you want me to turn the heat up', 'Hot'],
  ['Should I sit you up a bit', 'Sit up'],
  ['Are you comfortable', 'Turn me'],
  ['Who do you want me to call', 'Mom'],
  ['Did you sleep alright', 'Tired'],
  ['How are you feeling', 'Okay'],
  ['Do you want anything', 'Water'],
  ['When do you want your lunch', 'Now'],
]
for (const [heard, expected] of answers) {
  const result = suggestFor(heard, SLOTS)
  const labels = result?.words.map((word) => word.label) ?? []
  check(labels.includes(expected), `"${heard}" -> offers ${expected}  [${labels.join(', ')}]`)
}

console.log('\nleaves the board alone when nothing was asked')
const quiet = [
  "I'll get the nurse now.",
  'The weather is terrible today.',
  'Your mother called earlier.',
  'Okay.',
  'I am going to open the blinds.',
  '',
  '   ',
]
for (const heard of quiet) {
  check(suggestFor(heard, SLOTS) === null, `"${heard}" -> no change`)
}

console.log('\nstructure')
const painResult = suggestFor('Are you in pain?', SLOTS)
check(painResult?.words.length === SLOTS, `fills all ${SLOTS} slots (got ${painResult?.words.length})`)
check(
  new Set(painResult?.words.map((word) => word.label)).size === SLOTS,
  'no duplicate tiles',
)
// The top row is a permanent fixture; spending a suggestion slot on anything already
// sitting there would waste the only tiles that can adapt.
const reserved = ['Yes', 'No', 'Help', 'Pain', 'Wait', 'Not that']
const everySuggestion = [...answers, ['Anything else?', ''] as [string, string]]
  .flatMap(([heard]) => suggestFor(heard, SLOTS)?.words.map((word) => word.label) ?? [])
check(!everySuggestion.some((label) => reserved.includes(label)), 'never suggests yes / no / help / pain')
check(
  (painResult?.words ?? []).every((word) => word.speech.length > 0),
  'every suggestion has something to say',
)
check(painResult?.because === 'about pain', `explains itself: "${painResult?.because}"`)

// A compound question should draw from both topics rather than six ways to say one.
const both = suggestFor('Are you cold, or do you want a drink?', SLOTS)
const bothLabels = both?.words.map((word) => word.label) ?? []
check(
  bothLabels.includes('Cold') && bothLabels.includes('Water'),
  `compound question offers both  [${bothLabels.join(', ')}]`,
)

check(isQuestion('are you okay') && !isQuestion('the nurse is coming'), 'question detection without punctuation')

// A narrower board must not overflow.
check((suggestFor('Are you in pain?', 3)?.words.length ?? 0) === 3, 'respects a smaller slot count')
check(suggestFor('Are you in pain?', 0) === null, 'no slots means no suggestions')

console.log(failures === 0 ? '\nall suggestion cases pass' : `\n${failures} suggestion cases FAILED`)
process.exit(failures === 0 ? 0 : 1)
