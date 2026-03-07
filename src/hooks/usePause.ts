import { useState, useCallback, useEffect } from 'react'

let paused = false
const listeners = new Set<() => void>()

export function isPaused(): boolean {
  return paused
}

export function subscribePause(fn: (paused: boolean) => void): () => void {
  const wrapper = () => fn(paused)
  listeners.add(wrapper)
  return () => { listeners.delete(wrapper) }
}

function notify() {
  listeners.forEach((fn) => fn())
}

export function usePause() {
  const [, forceUpdate] = useState(0)

  useEffect(() => {
    const fn = () => forceUpdate((n) => n + 1)
    listeners.add(fn)
    return () => { listeners.delete(fn) }
  }, [])

  const toggle = useCallback(() => {
    paused = !paused
    notify()
  }, [])

  return { paused, toggle }
}
