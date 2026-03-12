import { useEffect, useState } from 'react'

export interface TimingEntry {
  opName: string
  ms: number
}

// モジュールレベルのタイミングストア
const timingMap = new Map<string, TimingEntry>()
const listeners = new Set<() => void>()
let rafScheduled = false

/** 指定ノードのタイミングを削除 */
export function clearTimingForNode(nodeId: string): void {
  if (timingMap.delete(nodeId)) {
    listeners.forEach((fn) => fn())
  }
}

/** チェーンに属するノードのうち、今回処理されなかったもののタイミングを削除 */
export function clearTimingsForChain(chainNodeIds: string[], processedNodeIds: Set<string>): void {
  for (const id of chainNodeIds) {
    if (!processedNodeIds.has(id)) timingMap.delete(id)
  }
}

export function setTiming(nodeId: string, opName: string, ms: number): void {
  timingMap.set(nodeId, { opName, ms })
  // 同一フレーム内の複数呼び出しを1回の通知にまとめる
  if (!rafScheduled) {
    rafScheduled = true
    requestAnimationFrame(() => {
      rafScheduled = false
      listeners.forEach((fn) => fn())
    })
  }
}

export function useTiming(): Map<string, TimingEntry> {
  const [, forceUpdate] = useState(0)
  useEffect(() => {
    const fn = () => forceUpdate((n) => n + 1)
    listeners.add(fn)
    return () => { listeners.delete(fn) }
  }, [])
  return new Map(timingMap)
}
