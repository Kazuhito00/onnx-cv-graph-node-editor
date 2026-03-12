/**
 * 推論 Worker とのメッセージングをラップするフック
 *
 * - preload: 全モデルをWorkerでプリロード
 * - requestRun: グラフ構造 + テンソルを送って推論を要求
 * - 結果を受け取って previewUrl / timing を更新
 */

import { useCallback, useEffect, useRef } from 'react'
import { useReactFlow, type Node, type Edge } from '@xyflow/react'
import { setTiming, clearTimingsForChain } from './useTiming'
import type { TensorData, ModelsMeta, ProcessingNodeData, SubgraphNodeData } from '../types'

/** sourceNodeId からエッジを辿り、チェーン上の全ノードIDを返す */
function getChainNodeIds(sourceNodeId: string, nodes: Node[], edges: Edge[]): string[] {
  const ids: string[] = []
  let currentId = sourceNodeId
  while (true) {
    const outEdge = edges.find((e) => e.source === currentId && !e.hidden)
    if (!outEdge) break
    const nextNode = nodes.find((n) => n.id === outEdge.target)
    if (!nextNode) break
    ids.push(nextNode.id)
    if (nextNode.type === 'subgraph') {
      const sd = nextNode.data as unknown as SubgraphNodeData
      if (sd.collapsed && sd.childNodeIds) {
        ids.push(...sd.childNodeIds)
      }
    }
    currentId = nextNode.id
  }
  return ids
}

// Worker のシリアライズ用の軽量ノード型
interface GraphNode {
  id: string
  type: string
  hidden?: boolean
  data: {
    opName?: string
    categoryId?: string
    params?: Record<string, number>
    paramMeta?: Record<string, { min: number; max: number; default: number }>
    uploadModelKey?: string
    collapsed?: boolean
    childNodeIds?: string[]
    childEdgeIds?: string[]
  }
}

interface GraphEdge {
  id: string
  source: string
  target: string
  hidden?: boolean
}

interface NodePreview {
  nodeId: string
  blob: Blob | null
}

interface TimingEntry {
  nodeId: string
  opName: string
  ms: number
}

// Worker → メインのメッセージ型
interface PreloadProgressMsg {
  type: 'preload-progress'
  loaded: number
  total: number
}

interface PreloadDoneMsg {
  type: 'preload-done'
}

interface RunResultMsg {
  type: 'run-result'
  sourceNodeId: string
  previews: NodePreview[]
  timings: TimingEntry[]
  processedNodeIds: string[]
}

type WorkerOutMsg = PreloadProgressMsg | PreloadDoneMsg | RunResultMsg

let workerInstance: Worker | null = null

function getWorker(): Worker {
  if (!workerInstance) {
    workerInstance = new Worker(
      new URL('../workers/inferenceWorker.ts', import.meta.url),
      { type: 'module' },
    )
  }
  return workerInstance
}

/**
 * Worker にアップロードされた ONNX モデルを登録する
 * 入力が "input" 1つのみのモデルのみ対応。非対応の場合は null を返す
 */
export function registerUploadedModel(
  modelKey: string,
  modelData: ArrayBuffer,
): Promise<null | true> {
  const worker = getWorker()
  return new Promise((resolve) => {
    const handler = (e: MessageEvent) => {
      if (e.data.type === 'model-registered' && e.data.modelKey === modelKey) {
        worker.removeEventListener('message', handler)
        resolve(e.data.ok ? true : null)
      }
    }
    worker.addEventListener('message', handler)
    worker.postMessage({ type: 'register-model', modelKey, modelData }, [modelData])
  })
}

/**
 * 相対 base パスを絶対URLに解決する
 */
function resolveBaseUrl(base: string): string {
  const url = new URL(base, window.location.href)
  return url.href.endsWith('/') ? url.href : url.href + '/'
}

/**
 * Worker でモデルをプリロードする
 */
export function preloadWithWorker(
  base: string,
  meta: ModelsMeta,
  onProgress: (loaded: number, total: number) => void,
): Promise<void> {
  const worker = getWorker()
  const baseUrl = resolveBaseUrl(base)

  return new Promise<void>((resolve) => {
    const handler = (e: MessageEvent<WorkerOutMsg>) => {
      if (e.data.type === 'preload-progress') {
        onProgress(e.data.loaded, e.data.total)
      } else if (e.data.type === 'preload-done') {
        worker.removeEventListener('message', handler)
        resolve()
      }
    }
    worker.addEventListener('message', handler)
    worker.postMessage({ type: 'preload', base, baseUrl, meta })
  })
}

/**
 * ノードとエッジを Worker 送信用に軽量化する
 */
function serializeNodes(nodes: Node[]): GraphNode[] {
  return nodes.map((n) => {
    const d = n.data as Record<string, unknown>
    const gn: GraphNode = {
      id: n.id,
      type: n.type ?? '',
      hidden: n.hidden,
      data: {},
    }
    if (n.type === 'processing') {
      const pd = d as unknown as ProcessingNodeData
      gn.data = {
        opName: pd.opName,
        categoryId: pd.categoryId,
        params: pd.params,
        paramMeta: pd.paramMeta,
        uploadModelKey: (pd as unknown as Record<string, unknown>).uploadModelKey as string | undefined,
      }
    } else if (n.type === 'subgraph') {
      const sd = d as unknown as SubgraphNodeData
      gn.data = {
        collapsed: sd.collapsed,
        childNodeIds: sd.childNodeIds,
        childEdgeIds: sd.childEdgeIds,
      }
    }
    return gn
  })
}

function serializeEdges(edges: Edge[]): GraphEdge[] {
  return edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    hidden: e.hidden,
  }))
}

// ---------- モジュールレベルのシングルトン結果ハンドラ ----------

// グローバル busy フラグ（Worker に同時に1リクエストのみ）
let busy = false
// プレビューURL管理（古いURLをrevokeするため）
const prevUrls = new Map<string, string>()

// ReactFlow の getNodes/getEdges/setNodes を最新の参照で保持
let latestGetNodes: (() => Node[]) | null = null
let latestGetEdges: (() => Edge[]) | null = null
let latestSetNodes: ((updater: (ns: Node[]) => Node[]) => void) | null = null
let listenerInstalled = false

function handleRunResult(msg: RunResultMsg) {
  busy = false

  if (!latestGetNodes || !latestGetEdges || !latestSetNodes) return

  // タイミング更新: このチェーンに属するノードのうち未処理のもののみクリア
  const currentNodes = latestGetNodes()
  const currentEdges = latestGetEdges()
  const existingNodeIds = new Set(currentNodes.map((n) => n.id))
  const processedSet = new Set(msg.processedNodeIds)
  const chainNodeIds = getChainNodeIds(msg.sourceNodeId, currentNodes, currentEdges)
  clearTimingsForChain(chainNodeIds, processedSet)
  for (const t of msg.timings) {
    if (existingNodeIds.has(t.nodeId)) {
      setTiming(t.nodeId, t.opName, t.ms)
    }
  }

  // プレビュー更新
  const urlUpdates = new Map<string, string | undefined>()
  for (const p of msg.previews) {
    const oldUrl = prevUrls.get(p.nodeId)
    if (oldUrl) URL.revokeObjectURL(oldUrl)

    if (p.blob) {
      const url = URL.createObjectURL(p.blob)
      urlUpdates.set(p.nodeId, url)
      prevUrls.set(p.nodeId, url)
    } else {
      urlUpdates.set(p.nodeId, undefined)
      prevUrls.delete(p.nodeId)
    }
  }

  if (urlUpdates.size > 0) {
    latestSetNodes((ns) =>
      ns.map((n) => {
        if (!urlUpdates.has(n.id)) return n
        const previewUrl = urlUpdates.get(n.id)
        return { ...n, data: { ...n.data, previewUrl } }
      }),
    )
  }
}

function ensureWorkerListener() {
  if (listenerInstalled) return
  listenerInstalled = true
  const worker = getWorker()
  worker.addEventListener('message', (e: MessageEvent<WorkerOutMsg>) => {
    if (e.data.type === 'run-result') {
      handleRunResult(e.data as RunResultMsg)
    }
  })
}

/**
 * 推論リクエストを Worker に送り、結果で React state を更新するフック
 */
export function useInferenceWorker(base: string) {
  const { getNodes, getEdges, setNodes } = useReactFlow()

  // ReactFlow の最新参照をモジュールレベルに保持
  const getNodesRef = useRef(getNodes)
  const getEdgesRef = useRef(getEdges)
  const setNodesRef = useRef(setNodes)
  useEffect(() => {
    getNodesRef.current = getNodes
    getEdgesRef.current = getEdges
    setNodesRef.current = setNodes
    latestGetNodes = getNodes
    latestGetEdges = getEdges
    latestSetNodes = setNodes
  })

  // Worker リスナーを1回だけ登録
  useEffect(() => {
    ensureWorkerListener()
  }, [])

  const propagateFrom = useCallback(
    (nodeId: string, tensor: TensorData): boolean => {
      if (busy) return false

      const nodes = getNodes()
      const edges = getEdges()
      const worker = getWorker()

      const graphNodes = serializeNodes(nodes)
      const graphEdges = serializeEdges(edges)

      const tensorCopy = new Float32Array(tensor.data)

      busy = true
      worker.postMessage(
        {
          type: 'run',
          base,
          sourceNodeId: nodeId,
          tensorData: tensorCopy,
          tensorDims: tensor.dims,
          nodes: graphNodes,
          edges: graphEdges,
        },
        [tensorCopy.buffer],
      )
      return true
    },
    [base, getNodes, getEdges],
  )

  return { propagateFrom }
}
