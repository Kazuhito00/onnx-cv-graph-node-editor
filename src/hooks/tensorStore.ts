// テンソルデータを React state の外で管理する
// ReactFlow の node.data に巨大な Float32Array を入れると
// setNodes 時の React diff で数秒ブロックされるため

import type { TensorData } from '../types'

const store = new Map<string, TensorData>()

export function setTensor(nodeId: string, tensor: TensorData): void {
  store.set(nodeId, tensor)
}

export function getTensor(nodeId: string): TensorData | undefined {
  return store.get(nodeId)
}

export function deleteTensor(nodeId: string): void {
  store.delete(nodeId)
}
