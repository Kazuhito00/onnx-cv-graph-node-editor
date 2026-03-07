import { useCallback } from 'react'
import { useReactFlow, type Node, type Edge } from '@xyflow/react'
import type { SubgraphNodeData } from '../types'
import { markDirty } from './useDirtyFlag'

let subgraphCounter = 1

/**
 * 選択ノードが線形チェーンかどうか検証し、順序付きリストを返す
 * 不正な場合は null を返す
 */
function validateLinearChain(
  selectedIds: string[],
  nodes: Node[],
  edges: Edge[],
): { orderedNodeIds: string[]; internalEdgeIds: string[] } | 'io' | null {
  const selectedSet = new Set(selectedIds)
  for (const id of selectedIds) {
    const node = nodes.find((n) => n.id === id)
    if (!node) return null
    if (node.type === 'imageInput' || node.type === 'imageOutput') return 'io'
    if (node.type !== 'processing') return null
  }

  // 選択ノード間のエッジを抽出
  const internalEdges = edges.filter(
    (e) => selectedSet.has(e.source) && selectedSet.has(e.target),
  )

  // 各ノードの内部接続数を確認（先頭/末尾以外は入出力各1つ）
  const inDegree = new Map<string, number>()
  const outDegree = new Map<string, number>()
  for (const id of selectedIds) {
    inDegree.set(id, 0)
    outDegree.set(id, 0)
  }
  for (const e of internalEdges) {
    outDegree.set(e.source, (outDegree.get(e.source) ?? 0) + 1)
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1)
  }

  // 先頭ノード（内部入力なし）を見つける
  const heads = selectedIds.filter((id) => inDegree.get(id) === 0)
  if (heads.length !== 1) return null

  // チェーンを辿る
  const orderedNodeIds: string[] = []
  const internalEdgeIds: string[] = []
  let current = heads[0]
  while (current) {
    orderedNodeIds.push(current)
    const outEdge = internalEdges.find((e) => e.source === current)
    if (!outEdge) break
    internalEdgeIds.push(outEdge.id)
    current = outEdge.target
  }

  // 全ノードが含まれているか確認
  if (orderedNodeIds.length !== selectedIds.length) return null

  return { orderedNodeIds, internalEdgeIds }
}

export function useSubgraph() {
  const { getNodes, getEdges, setNodes, setEdges } = useReactFlow()

  /**
   * 選択ノードをグループ化
   */
  const groupNodes = useCallback(
    (selectedNodeIds: string[]) => {
      const nodes = getNodes()
      const edges = getEdges()

      const result = validateLinearChain(selectedNodeIds, nodes, edges)
      if (result === 'io') {
        alert('InputノードとOutputノードはグループ化できません')
        return
      }
      if (!result) {
        alert('線形チェーン（1対1接続）のProcessingNodeのみグループ化できます')
        return
      }

      const { orderedNodeIds, internalEdgeIds } = result
      const firstChildId = orderedNodeIds[0]
      const lastChildId = orderedNodeIds[orderedNodeIds.length - 1]

      // サブグラフノードの位置（子ノード群のバウンディングボックス中央）
      const childNodes = orderedNodeIds.map((id) => nodes.find((n) => n.id === id)!)
      const minX = Math.min(...childNodes.map((n) => n.position.x))
      const minY = Math.min(...childNodes.map((n) => n.position.y))
      const maxX = Math.max(...childNodes.map((n) => n.position.x + 220))
      const maxY = Math.max(...childNodes.map((n) => n.position.y + 100))

      const subgraphId = `sg${subgraphCounter++}`

      // 外部接続エッジを見つける
      const incomingEdge = edges.find(
        (e) => e.target === firstChildId && !orderedNodeIds.includes(e.source),
      )
      const outgoingEdge = edges.find(
        (e) => e.source === lastChildId && !orderedNodeIds.includes(e.target),
      )

      const subgraphData: SubgraphNodeData = {
        name: `Group ${subgraphCounter - 1}`,
        onnxExportName: '',
        collapsed: true,
        childNodeIds: orderedNodeIds,
        childEdgeIds: internalEdgeIds,
        externalSourceEdge: incomingEdge
          ? { edgeId: incomingEdge.id, originalTarget: incomingEdge.target }
          : undefined,
        externalTargetEdge: outgoingEdge
          ? { edgeId: outgoingEdge.id, originalSource: outgoingEdge.source }
          : undefined,
      }

      const subgraphNode: Node = {
        id: subgraphId,
        type: 'subgraph',
        position: { x: (minX + maxX) / 2 - 110, y: (minY + maxY) / 2 - 50 },
        data: subgraphData,
      }

      // ノード更新: 子ノードを非表示にし、サブグラフノードを追加
      setNodes((ns) => [
        ...ns
          .map((n) => {
            if (orderedNodeIds.includes(n.id)) {
              return { ...n, hidden: true }
            }
            return n
          }),
        subgraphNode,
      ])

      // エッジ更新: 内部エッジを非表示、外部エッジをリルート
      setEdges((es) =>
        es.map((e) => {
          if (internalEdgeIds.includes(e.id)) {
            return { ...e, hidden: true }
          }
          if (incomingEdge && e.id === incomingEdge.id) {
            return { ...e, target: subgraphId }
          }
          if (outgoingEdge && e.id === outgoingEdge.id) {
            return { ...e, source: subgraphId }
          }
          return e
        }),
      )

      markDirty()
    },
    [getNodes, getEdges, setNodes, setEdges],
  )

  /**
   * サブグラフを解除して子ノードを復元
   */
  const ungroupNode = useCallback(
    (subgraphNodeId: string) => {
      const nodes = getNodes()
      const subNode = nodes.find((n) => n.id === subgraphNodeId)
      if (!subNode || subNode.type !== 'subgraph') return

      const data = subNode.data as unknown as SubgraphNodeData

      // ノード更新: 子ノードを再表示（階段状に配置）、サブグラフノードを削除
      const sgPos = subNode.position
      setNodes((ns) =>
        ns
          .filter((n) => n.id !== subgraphNodeId)
          .map((n) => {
            const childIdx = data.childNodeIds.indexOf(n.id)
            if (childIdx >= 0) {
              return {
                ...n,
                hidden: false,
                position: {
                  x: sgPos.x + childIdx * 30,
                  y: sgPos.y + childIdx * 55,
                },
              }
            }
            return n
          }),
      )

      // エッジ更新: 内部エッジを再表示、外部エッジを元に戻す、孤立エッジを削除
      setEdges((es) =>
        es
          .map((e) => {
            if (data.childEdgeIds.includes(e.id)) {
              return { ...e, hidden: false }
            }
            if (data.externalSourceEdge && e.id === data.externalSourceEdge.edgeId) {
              return { ...e, target: data.externalSourceEdge.originalTarget }
            }
            if (data.externalTargetEdge && e.id === data.externalTargetEdge.edgeId) {
              return { ...e, source: data.externalTargetEdge.originalSource }
            }
            return e
          })
          .filter((e) => e.source !== subgraphNodeId && e.target !== subgraphNodeId),
      )

      markDirty()
    },
    [getNodes, setNodes, setEdges],
  )

  /**
   * 折りたたみ/展開トグル
   */
  const toggleCollapse = useCallback(
    (subgraphNodeId: string) => {
      const nodes = getNodes()
      const subNode = nodes.find((n) => n.id === subgraphNodeId)
      if (!subNode || subNode.type !== 'subgraph') return

      const data = subNode.data as unknown as SubgraphNodeData
      const willCollapse = !data.collapsed

      setNodes((ns) =>
        ns.map((n) => {
          if (n.id === subgraphNodeId) {
            return { ...n, hidden: !willCollapse, data: { ...n.data, collapsed: willCollapse } }
          }
          if (data.childNodeIds.includes(n.id)) {
            return { ...n, hidden: willCollapse }
          }
          return n
        }),
      )

      setEdges((es) =>
        es.map((e) => {
          // 内部エッジの表示/非表示
          if (data.childEdgeIds.includes(e.id)) {
            return { ...e, hidden: willCollapse }
          }
          // 外部エッジのリルート
          if (data.externalSourceEdge && e.id === data.externalSourceEdge.edgeId) {
            return {
              ...e,
              target: willCollapse ? subgraphNodeId : data.externalSourceEdge.originalTarget,
            }
          }
          if (data.externalTargetEdge && e.id === data.externalTargetEdge.edgeId) {
            return {
              ...e,
              source: willCollapse ? subgraphNodeId : data.externalTargetEdge.originalSource,
            }
          }
          return e
        }),
      )

      markDirty()
    },
    [getNodes, setNodes, setEdges],
  )

  return { groupNodes, ungroupNode, toggleCollapse }
}
