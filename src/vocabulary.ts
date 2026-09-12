export type Tile = {
  label: string
  // What gets spoken, when that differs from the tile text.
  speech?: string
  // Navigates to another board instead of speaking.
  goTo?: string
  // Clears any suggested words and puts the usual board back.
  restore?: boolean
  accent?: string
}

export type Board = {
  id: string
  title: string
  tiles: Tile[]
}

// The top row never changes, on any board, at any point in a conversation. Everything
// else on screen can be replaced - folders swap the bottom row, and a question from the
// other person replaces it with answers - but these six stay where they are so that
// reaching them is muscle memory rather than a search.
//
// The last two are the ones people leave off a demo board and regret in real use. A dwell
// selection takes over a second, so partners routinely talk over someone mid-sentence or
// walk away assuming they are done; "Wait" is how you hold the floor. And an accidental
// dwell says something out loud that cannot be taken back, so the undo for a speech board
// is not a delete key, it is being able to say that was not what you meant.
export const TOP_ROW: Tile[] = [
  { label: 'Yes', speech: 'Yes', accent: 'yes' },
  { label: 'No', speech: 'No', accent: 'no' },
  { label: 'Help', speech: 'I need help', accent: 'urgent' },
  { label: 'Pain', speech: 'I am in pain', accent: 'urgent' },
  { label: 'Wait', speech: 'Please wait, I am still choosing', accent: 'action' },
  { label: 'Not that', speech: 'That is not what I meant', accent: 'action' },
]

// Every tile below speaks a whole sentence the moment it is chosen.
//
// The alternative - tapping words into a sentence bar and then tapping speak - is what
// most board software does, and it costs three selections to ask for water. At eight
// hundred milliseconds of dwell plus cooldown per selection, that is the difference
// between answering someone and watching them give up and leave the room. Composing
// arbitrary sentences would be worth that price if there were a keyboard behind it;
// picking from a fixed set of words, it buys nothing that a well-written tile does not
// already say better.
export const BOARDS: Record<string, Board> = {
  home: {
    id: 'home',
    title: 'Home',
    tiles: [
      { label: 'Stop', speech: 'Please stop', accent: 'urgent' },
      { label: 'I want', goTo: 'wants', accent: 'folder' },
      { label: 'I feel', goTo: 'feelings', accent: 'folder' },
      { label: 'My body', goTo: 'body', accent: 'folder' },
      { label: 'People', goTo: 'people', accent: 'folder' },
      { label: 'Ask', goTo: 'ask', accent: 'folder' },
    ],
  },
  wants: {
    id: 'wants',
    title: 'I want',
    tiles: [
      { label: 'Water', speech: 'Can I have some water?' },
      { label: 'Food', speech: 'I would like something to eat' },
      { label: 'Bathroom', speech: 'I need the bathroom' },
      { label: 'Blanket', speech: 'Can I have a blanket?' },
      { label: 'My phone', speech: 'Can I have my phone?' },
      { label: 'Back', goTo: 'home', accent: 'folder' },
    ],
  },
  feelings: {
    id: 'feelings',
    title: 'I feel',
    tiles: [
      { label: 'Okay', speech: 'I feel okay' },
      { label: 'Tired', speech: 'I feel tired' },
      { label: 'Scared', speech: 'I feel scared' },
      { label: 'Frustrated', speech: 'I feel frustrated' },
      { label: 'Sad', speech: 'I feel sad' },
      { label: 'Back', goTo: 'home', accent: 'folder' },
    ],
  },
  body: {
    id: 'body',
    title: 'My body',
    tiles: [
      { label: 'Sit me up', speech: 'Please sit me up' },
      { label: 'Turn me', speech: 'Please turn me over' },
      { label: 'Too cold', speech: 'I am too cold' },
      { label: 'Too hot', speech: 'I am too hot' },
      { label: 'Itchy', speech: 'I am itchy' },
      { label: 'Back', goTo: 'home', accent: 'folder' },
    ],
  },
  people: {
    id: 'people',
    title: 'People',
    tiles: [
      { label: 'My family', speech: 'I want to see my family' },
      { label: 'Nurse', speech: 'Please get the nurse' },
      { label: 'Doctor', speech: 'I want to see the doctor' },
      { label: 'Thank you', speech: 'Thank you' },
      { label: 'I love you', speech: 'I love you' },
      { label: 'Back', goTo: 'home', accent: 'folder' },
    ],
  },
  // Questions are the half of communication that request-only boards leave out. Someone
  // who can only ask for things is a patient; someone who can ask what is happening to
  // them is a person.
  ask: {
    id: 'ask',
    title: 'Ask',
    tiles: [
      { label: 'What time?', speech: 'What time is it?' },
      { label: 'How long?', speech: 'How much longer will this take?' },
      { label: 'What happened?', speech: 'What happened to me?' },
      { label: 'Go home?', speech: 'When can I go home?' },
      { label: 'Stay?', speech: 'Will you stay with me?' },
      { label: 'Back', goTo: 'home', accent: 'folder' },
    ],
  },
}
