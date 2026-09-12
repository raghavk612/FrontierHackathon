interface WebGazer {
  params: { faceMeshSolutionPath: string }
  setRegression(name: string): WebGazer
  setTracker(name: string): WebGazer
  applyKalmanFilter(enabled: boolean): WebGazer
  saveDataAcrossSessions(enabled: boolean): WebGazer
  showPredictionPoints(enabled: boolean): WebGazer
  showFaceOverlay(enabled: boolean): WebGazer
  showFaceFeedbackBox(enabled: boolean): WebGazer
  showVideoPreview(enabled: boolean): WebGazer
  setGazeListener(listener: (data: { x: number; y: number } | null, elapsedTime: number) => void): WebGazer
  clearGazeListener(): WebGazer
  recordScreenPosition(x: number, y: number, eventType?: string): WebGazer
  removeMouseEventListeners(): WebGazer
  clearData(): Promise<void>
  begin(): Promise<WebGazer>
  end(): WebGazer
  stopVideo(): WebGazer
}

interface Window {
  webgazer?: WebGazer
}
declare module 'webgazer' {
  type GazeListener = (data: { x: number; y: number } | null, elapsedTime: number) => void

  interface WebGazer {
    params: { faceMeshSolutionPath: string }
    setRegression(name: string): WebGazer
    setTracker(name: string): WebGazer
    applyKalmanFilter(enabled: boolean): WebGazer
    saveDataAcrossSessions(enabled: boolean): WebGazer
    showPredictionPoints(enabled: boolean): WebGazer
    showFaceOverlay(enabled: boolean): WebGazer
    showFaceFeedbackBox(enabled: boolean): WebGazer
    showVideoPreview(enabled: boolean): WebGazer
    setGazeListener(listener: GazeListener): WebGazer
    clearGazeListener(): WebGazer
    recordScreenPosition(x: number, y: number, eventType?: string): WebGazer
    removeMouseEventListeners(): WebGazer
    clearData(): Promise<void>
    begin(): Promise<WebGazer>
    end(): WebGazer
    stopVideo(): WebGazer
  }

  const webgazer: WebGazer
  export default webgazer
}
