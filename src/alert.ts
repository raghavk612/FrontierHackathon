// Caregiver alert beacon -> Particle Argon.
//
// A speech board only reaches people who are already looking at it, and arranging for
// someone to be looking at you is the exact thing its user cannot do. This sends the
// urgent words to a physical light that can sit wherever the caregiver actually is.
//
// Bluetooth rather than Particle's cloud, for two reasons. Gen 3 boards can only join
// WPA2-Personal networks, so university and conference Wi-Fi is unavailable to them; and
// a hospital room is a worse network environment than a hackathon, so a system that needs
// no network is the more honest design.
//
// USB serial is kept as a second transport because a hall full of people is a hostile
// 2.4 GHz environment. The cable tells a weaker story and survives a worse room.

export const CLEAR = 0
export const REQUEST = 1
export const URGENT = 2
export type AlertLevel = typeof CLEAR | typeof REQUEST | typeof URGENT

const SERVICE_UUID = 'f1e2d300-9a5b-4c7e-8d21-6b0f3a9c1e00'
const ALERT_UUID = 'f1e2d301-9a5b-4c7e-8d21-6b0f3a9c1e00'
const BAUD_RATE = 9600

// Neither Web Bluetooth nor Web Serial is in lib.dom, so both get the minimum surface
// this file actually touches.
type GattCharacteristic = {
  writeValue(value: BufferSource): Promise<void>
  writeValueWithoutResponse?(value: BufferSource): Promise<void>
}
type GattService = { getCharacteristic(uuid: string): Promise<GattCharacteristic> }
type GattServer = {
  connected: boolean
  connect(): Promise<GattServer>
  disconnect(): void
  getPrimaryService(uuid: string): Promise<GattService>
}
type RemoteDevice = {
  name?: string
  gatt?: GattServer
  addEventListener(type: string, handler: () => void): void
}
type BluetoothApi = {
  requestDevice(options: {
    filters: { services: string[] }[]
    optionalServices?: string[]
  }): Promise<RemoteDevice>
}

type SerialWriter = { write(data: Uint8Array): Promise<void>; releaseLock(): void }
type SerialPort = {
  open(options: { baudRate: number }): Promise<void>
  close(): Promise<void>
  writable: { getWriter(): SerialWriter } | null
}
type SerialApi = { requestPort(): Promise<SerialPort> }

function bluetoothApi(): BluetoothApi | null {
  if (typeof navigator === 'undefined') return null
  return (navigator as unknown as { bluetooth?: BluetoothApi }).bluetooth ?? null
}

function serialApi(): SerialApi | null {
  if (typeof navigator === 'undefined') return null
  return (navigator as unknown as { serial?: SerialApi }).serial ?? null
}

export function bluetoothAvailable() {
  return bluetoothApi() !== null
}

export function serialAvailable() {
  return serialApi() !== null
}

export type BeaconStatus = 'off' | 'connecting' | 'bluetooth' | 'serial' | 'error'
export type Beacon = {
  connectBluetooth(): void
  connectSerial(): void
  disconnect(): void
  send(level: AlertLevel): void
  test(): void
}

export function createBeacon(onStatus: (status: BeaconStatus, detail: string) => void): Beacon {
  let characteristic: GattCharacteristic | null = null
  let server: GattServer | null = null
  let port: SerialPort | null = null
  let writer: SerialWriter | null = null

  // Selections can arrive faster than a BLE write completes. Holding only the newest
  // pending level means a burst of words cannot build a backlog that keeps lighting the
  // board up after the person has stopped talking.
  let inFlight = false
  let queued: AlertLevel | null = null

  function reset() {
    characteristic = null
    server = null
    port = null
    writer = null
    inFlight = false
    queued = null
  }

  async function deliver(level: AlertLevel) {
    if (characteristic) {
      const payload = new Uint8Array([level])
      if (characteristic.writeValueWithoutResponse) {
        await characteristic.writeValueWithoutResponse(payload)
      } else {
        await characteristic.writeValue(payload)
      }
      return
    }
    if (writer) {
      // The firmware reads characters so the beacon can also be driven from any serial
      // monitor, which is the fastest way to prove the hardware works on its own.
      await writer.write(new Uint8Array([0x30 + level]))
    }
  }

  async function drain() {
    if (inFlight) return
    inFlight = true
    try {
      while (queued !== null) {
        const level = queued
        queued = null
        await deliver(level)
      }
    } catch (problem) {
      // A beacon that throws into the tracking loop would take the whole board down with
      // it. Losing the light is survivable; losing the voice is not.
      reset()
      onStatus('error', problem instanceof Error ? problem.message : 'Beacon disconnected')
      return
    }
    inFlight = false
  }

  function send(level: AlertLevel) {
    if (!characteristic && !writer) return
    queued = level
    void drain()
  }

  function connectBluetooth() {
    const api = bluetoothApi()
    if (!api) {
      onStatus('error', 'This browser has no Web Bluetooth. Chrome or Edge will work.')
      return
    }
    onStatus('connecting', 'Pick "VividVision" in the pairing window.')
    void (async () => {
      try {
        const device = await api.requestDevice({ filters: [{ services: [SERVICE_UUID] }] })
        const gatt = device.gatt
        if (!gatt) throw new Error('That device has no GATT server.')
        device.addEventListener('gattserverdisconnected', () => {
          reset()
          onStatus('off', 'Beacon disconnected.')
        })
        server = await gatt.connect()
        const service = await server.getPrimaryService(SERVICE_UUID)
        characteristic = await service.getCharacteristic(ALERT_UUID)
        onStatus('bluetooth', `Connected to ${device.name ?? 'the beacon'} over Bluetooth.`)
        send(CLEAR)
      } catch (problem) {
        reset()
        const message = problem instanceof Error ? problem.message : String(problem)
        // Closing the chooser is a decision, not a fault.
        if (/cancel|User cancelled/i.test(message)) {
          onStatus('off', 'Pairing cancelled.')
        } else {
          onStatus('error', message)
        }
      }
    })()
  }

  function connectSerial() {
    const api = serialApi()
    if (!api) {
      onStatus('error', 'This browser has no Web Serial. Chrome or Edge will work.')
      return
    }
    onStatus('connecting', 'Pick the Argon in the port window.')
    void (async () => {
      try {
        port = await api.requestPort()
        await port.open({ baudRate: BAUD_RATE })
        if (!port.writable) throw new Error('That port cannot be written to.')
        writer = port.writable.getWriter()
        onStatus('serial', 'Connected to the beacon over USB.')
        send(CLEAR)
      } catch (problem) {
        reset()
        const message = problem instanceof Error ? problem.message : String(problem)
        if (/cancel|No port selected/i.test(message)) {
          onStatus('off', 'Port selection cancelled.')
        } else {
          onStatus('error', message)
        }
      }
    })()
  }

  function disconnect() {
    try {
      writer?.releaseLock()
      void port?.close()
      server?.disconnect()
    } catch {
      // Already gone. Nothing to do but forget it.
    }
    reset()
    onStatus('off', 'Beacon off.')
  }

  // Lights the urgent pattern on demand, so the hardware can be shown to work before
  // anyone has calibrated a camera.
  function test() {
    send(URGENT)
  }

  return { connectBluetooth, connectSerial, disconnect, send, test }
}
