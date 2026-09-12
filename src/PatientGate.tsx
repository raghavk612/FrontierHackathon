import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import type { Session } from '@supabase/supabase-js'
import App from './App'
import {
  cloudConfigured,
  currentSession,
  loadProfile,
  makeRoomCode,
  saveProfile,
  supabase,
} from './supabase'

export default function PatientGate() {
  const [session, setSession] = useState<Session | null>(null)
  const [checking, setChecking] = useState(cloudConfigured)
  const [offline, setOffline] = useState(false)
  const [mode, setMode] = useState<'signin' | 'signup'>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [roomCode] = useState(makeRoomCode)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (!supabase) return
    void currentSession().then((value) => {
      setSession(value)
      setChecking(false)
    })
    const { data } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next)
      setChecking(false)
    })
    return () => data.subscription.unsubscribe()
  }, [])

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!supabase) return
    setBusy(true)
    setError('')
    try {
      if (mode === 'signin') {
        const { data, error: authError } = await supabase.auth.signInWithPassword({ email, password })
        if (authError) throw authError
        const profile = await loadProfile(data.user.id)
        if (profile?.role === 'caregiver') {
          await supabase.auth.signOut()
          throw new Error('This is a caregiver account. Sign in on the caregiver dashboard.')
        }
        if (!profile) throw new Error('This account has no patient profile. Create a patient account first.')
      } else {
        const { data, error: authError } = await supabase.auth.signUp({ email, password })
        if (authError) throw authError
        if (!data.session) {
          throw new Error('Check your email, then sign in. For instant demo access, disable email confirmation in Supabase.')
        }
        await saveProfile({
          id: data.user!.id,
          display_name: name.trim() || 'VividVision user',
          role: 'patient',
          room_code: roomCode,
        })
      }
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Could not sign in')
    } finally {
      setBusy(false)
    }
  }

  if (!cloudConfigured || offline) return <App />
  if (checking) return <div className="auth-loading">Connecting securely…</div>
  if (session) return <App />

  return (
    <main className="auth-page">
      <section className="auth-card">
        <img src={`${import.meta.env.BASE_URL}icons/vividvision.png`} alt="VividVision eye logo" />
        <p className="auth-kicker">PATIENT BOARD</p>
        <h1>{mode === 'signin' ? 'Welcome back.' : 'Create your VividVision account.'}</h1>
        <p className="auth-copy">
          {mode === 'signin'
            ? 'Sign in to send spoken messages securely to your caregiver.'
            : 'Your private room code connects this board with the caregiver dashboard.'}
        </p>
        <form onSubmit={submit}>
          {mode === 'signup' && (
            <>
              <label>Name<input required value={name} onChange={(event) => setName(event.target.value)} autoComplete="name" /></label>
              <div className="room-code"><span>Your caregiver code</span><strong>{roomCode}</strong><small>Share this code only with your caregiver.</small></div>
            </>
          )}
          <label>Email<input required type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" /></label>
          <label>Password<input required minLength={6} type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={mode === 'signin' ? 'current-password' : 'new-password'} /></label>
          {error && <p className="auth-error">{error}</p>}
          <button type="submit" disabled={busy}>{busy ? 'Please wait…' : mode === 'signin' ? 'Sign in to VividVision' : 'Create patient account'}</button>
        </form>
        <button className="auth-switch" onClick={() => { setMode(mode === 'signin' ? 'signup' : 'signin'); setError('') }}>
          {mode === 'signin' ? 'New patient? Create an account' : 'Already have an account? Sign in'}
        </button>
        <button className="auth-offline" onClick={() => setOffline(true)}>Continue without caregiver sync</button>
      </section>
    </main>
  )
}
