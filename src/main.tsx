import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import PatientGate from './PatientGate'
import './style.css'
import './polish.css'
import './welcome.css'
import './board.css'
import './accessibility.css'
import './auth.css'

createRoot(document.getElementById('app')!).render(
  <StrictMode>
    <PatientGate />
  </StrictMode>,
)
