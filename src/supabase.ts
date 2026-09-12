import { createClient } from '@supabase/supabase-js'
import type { Session } from '@supabase/supabase-js'

export type UserRole = 'patient' | 'caregiver'
export type Profile = {
  id: string
  display_name: string
  role: UserRole
  room_code: string
}

const url = import.meta.env.VITE_SUPABASE_URL?.trim()
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY?.trim()

export const cloudConfigured = Boolean(url && anonKey && !url.includes('YOUR_'))
export const supabase = cloudConfigured ? createClient(url, anonKey, {
  auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
}) : null

export function makeRoomCode() {
  const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
  const bytes = crypto.getRandomValues(new Uint8Array(8))
  return Array.from(bytes, (byte) => alphabet[byte % alphabet.length]).join('')
}

export async function currentSession(): Promise<Session | null> {
  if (!supabase) return null
  const { data } = await supabase.auth.getSession()
  return data.session
}

export async function loadProfile(userId: string): Promise<Profile | null> {
  if (!supabase) return null
  const { data, error } = await supabase.from('profiles').select('*').eq('id', userId).maybeSingle()
  if (error) throw error
  return data as Profile | null
}

export async function saveProfile(profile: Profile) {
  if (!supabase) throw new Error('Supabase is not configured')
  const { error } = await supabase.from('profiles').upsert(profile)
  if (error) throw error
}

export async function publishCloudMessage(text: string) {
  if (!supabase) return
  const session = await currentSession()
  if (!session) return
  const profile = await loadProfile(session.user.id)
  if (!profile || profile.role !== 'patient') return
  const level = /\b(help|pain|hurt|emergency|stop|can't breathe|cannot breathe)\b/i.test(text)
    ? 'urgent'
    : 'message'
  const { error } = await supabase.from('messages').insert({
    patient_id: session.user.id,
    room_code: profile.room_code,
    text,
    level,
  })
  if (error) throw error
}
