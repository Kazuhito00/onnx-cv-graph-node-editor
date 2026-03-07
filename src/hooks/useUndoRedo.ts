/**
 * Undo/Redo フック
 *
 * nodes と edges のスナップショットを履歴として保持し、
 * Ctrl+Z で undo、Ctrl+Y で redo を実行する。
 */

import { useCallback, useRef } from 'react'
import type { Node, Edge } from '@xyflow/react'

interface Snapshot {
  nodes: Node[]
  edges: Edge[]
}

const MAX_HISTORY = 50

export function useUndoRedo(
  getNodes: () => Node[],
  getEdges: () => Edge[],
  setNodes: (nodes: Node[]) => void,
  setEdges: (edges: Edge[]) => void,
) {
  const undoStack = useRef<Snapshot[]>([])
  const redoStack = useRef<Snapshot[]>([])

  /** 現在の状態をスナップショットとして保存（変更前に呼ぶ） */
  const saveSnapshot = useCallback(() => {
    const snapshot: Snapshot = {
      nodes: structuredClone(getNodes()),
      edges: structuredClone(getEdges()),
    }
    undoStack.current.push(snapshot)
    if (undoStack.current.length > MAX_HISTORY) {
      undoStack.current.shift()
    }
    // 新しい操作をしたら redo 履歴はクリア
    redoStack.current = []
  }, [getNodes, getEdges])

  /** Undo: 1つ前の状態に戻す */
  const undo = useCallback(() => {
    const prev = undoStack.current.pop()
    if (!prev) return
    // 現在の状態を redo スタックに保存
    redoStack.current.push({
      nodes: structuredClone(getNodes()),
      edges: structuredClone(getEdges()),
    })
    setNodes(prev.nodes)
    setEdges(prev.edges)
  }, [getNodes, getEdges, setNodes, setEdges])

  /** Redo: undo した操作をやり直す */
  const redo = useCallback(() => {
    const next = redoStack.current.pop()
    if (!next) return
    // 現在の状態を undo スタックに保存
    undoStack.current.push({
      nodes: structuredClone(getNodes()),
      edges: structuredClone(getEdges()),
    })
    setNodes(next.nodes)
    setEdges(next.edges)
  }, [getNodes, getEdges, setNodes, setEdges])

  return { saveSnapshot, undo, redo }
}
