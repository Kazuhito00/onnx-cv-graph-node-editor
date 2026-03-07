/**
 * 推論 Web Worker
 *
 * メインスレッドからグラフ構造 + 入力テンソルを受け取り、
 * チェーンを順次推論してプレビュー Blob + タイミングを返す。
 */

// WASM-only ビルドを使用（jsep/WebGPU の動的インポートを回避）
import * as ort from 'onnxruntime-web/wasm'

// Worker ではスレッド生成を無効化（Worker 内から Worker は生成できないため）
ort.env.wasm.numThreads = 1
// wasmPaths はメインスレッドから受け取った baseUrl で初期化する
let baseUrl = ''

// ---------- 型定義 ----------

interface TensorData {
  data: Float32Array
  dims: number[]
}

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

interface CategoryMeta {
  id: string
  models: string[]
}

interface ModelsMeta {
  categories: CategoryMeta[]
}

// ---------- メッセージ型 ----------

interface PreloadMsg {
  type: 'preload'
  base: string
  baseUrl: string
  meta: ModelsMeta
}

interface RunMsg {
  type: 'run'
  base: string
  sourceNodeId: string
  tensorData: Float32Array
  tensorDims: number[]
  nodes: GraphNode[]
  edges: GraphEdge[]
}

interface RegisterModelMsg {
  type: 'register-model'
  modelKey: string
  modelData: ArrayBuffer
}

type InMsg = PreloadMsg | RunMsg | RegisterModelMsg

interface TimingEntry {
  nodeId: string
  opName: string
  ms: number
}

interface NodePreview {
  nodeId: string
  blob: Blob | null
}

interface PreloadProgressOut {
  type: 'preload-progress'
  loaded: number
  total: number
}

interface PreloadDoneOut {
  type: 'preload-done'
}

interface RunResultOut {
  type: 'run-result'
  sourceNodeId: string
  previews: NodePreview[]
  timings: TimingEntry[]
  processedNodeIds: string[]
}

type OutMsg = PreloadProgressOut | PreloadDoneOut | RunResultOut

// ---------- セッションキャッシュ ----------

const sessionCache = new Map<string, Promise<ort.InferenceSession>>()

function getSession(url: string): Promise<ort.InferenceSession> {
  if (!sessionCache.has(url)) {
    sessionCache.set(url, ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
    }))
  }
  return sessionCache.get(url)!
}

// ---------- プレビュー生成（OffscreenCanvas） ----------

function float32HWCToBlob(
  data: Float32Array,
  width: number,
  height: number,
): Promise<Blob | null> {
  const out = new Uint8ClampedArray(height * width * 4)
  for (let i = 0; i < height * width; i++) {
    out[i * 4 + 0] = data[i * 3 + 0]
    out[i * 4 + 1] = data[i * 3 + 1]
    out[i * 4 + 2] = data[i * 3 + 2]
    out[i * 4 + 3] = 255
  }
  const imageData = new ImageData(out, width, height)
  const canvas = new OffscreenCanvas(width, height)
  canvas.getContext('2d')!.putImageData(imageData, 0, 0)
  return canvas.convertToBlob({ type: 'image/jpeg', quality: 0.85 })
}

function tensorToBlob(tensor: TensorData): Promise<Blob | null> {
  const { data, dims } = tensor
  const isNchw4d = dims.length === 4 && dims[0] === 1 && dims[1] === 3

  if (!isNchw4d) {
    return fallbackToBlob(data, dims)
  }

  const [, , h, w] = dims
  let maxVal = 0
  for (let i = 0; i < data.length; i++) {
    const v = Math.abs(data[i])
    if (v > maxVal) maxVal = v
  }
  const scale = maxVal > 1.001 ? (maxVal <= 255.5 ? 255 : maxVal) : 1

  const hwc = new Float32Array(h * w * 3)
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        hwc[(y * w + x) * 3 + c] =
          Math.max(0, Math.min(255, (data[c * h * w + y * w + x] / scale) * 255))
      }
    }
  }
  return float32HWCToBlob(hwc, w, h)
}

function fallbackToBlob(data: Float32Array, dims: number[]): Promise<Blob | null> {
  let h: number, w: number, hwcData: Float32Array

  if (dims.length === 3) {
    const [d0, d1, d2] = dims
    if (d2 === 3 || d2 === 1) {
      h = d0; w = d1
      hwcData = d2 === 3 ? data : expandGrayToRGB(data, d0 * d1)
    } else if (d0 === 3 || d0 === 1) {
      h = d1; w = d2
      hwcData = d0 === 3 ? chwToHwcJs(data, d0, d1, d2) : expandGrayToRGB(data, d1 * d2)
    } else {
      h = d0; w = d1
      hwcData = data.slice(0, h * w * 3)
    }
  } else if (dims.length === 4) {
    const [, d1, d2, d3] = dims
    if (d3 === 3 || d3 === 1) {
      h = d1; w = d2
      hwcData = d3 === 3 ? data.slice(0, h * w * 3) : expandGrayToRGB(data.slice(0, h * w), h * w)
    } else {
      h = d2; w = d3
      hwcData = d1 === 3 ? chwToHwcJs(data, d1, h, w) : expandGrayToRGB(data.slice(0, h * w), h * w)
    }
  } else {
    const side = Math.floor(Math.sqrt(data.length / 3))
    h = w = side
    hwcData = data.slice(0, h * w * 3)
  }

  const normalized = normalizeToUint8Range(hwcData)
  return float32HWCToBlob(normalized, w, h)
}

function chwToHwcJs(data: Float32Array, c: number, h: number, w: number): Float32Array {
  const out = new Float32Array(h * w * c)
  for (let ci = 0; ci < c; ci++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        out[(y * w + x) * c + ci] = data[ci * h * w + y * w + x]
      }
    }
  }
  return out
}

function expandGrayToRGB(gray: Float32Array, pixels: number): Float32Array {
  const out = new Float32Array(pixels * 3)
  for (let i = 0; i < pixels; i++) {
    out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = gray[i]
  }
  return out
}

function normalizeToUint8Range(data: Float32Array): Float32Array {
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < data.length; i++) {
    if (data[i] < min) min = data[i]
    if (data[i] > max) max = data[i]
  }
  const range = max - min
  if (range === 0) return new Float32Array(data.length).fill(0)
  const out = new Float32Array(data.length)
  for (let i = 0; i < data.length; i++) {
    out[i] = ((data[i] - min) / range) * 255
  }
  return out
}

// ---------- 推論ロジック ----------

async function runProcessingNode(
  node: GraphNode,
  currentTensor: TensorData,
  base: string,
  previews: NodePreview[],
  timings: TimingEntry[],
  processedNodeIds: Set<string>,
): Promise<TensorData | null> {
  const { opName, categoryId, params, paramMeta, uploadModelKey } = node.data
  if (!opName || !categoryId) return null

  try {
    let sess: ort.InferenceSession
    if (categoryId === '__upload__' && uploadModelKey) {
      const cached = sessionCache.get(uploadModelKey)
      if (!cached) return null
      sess = await cached
    } else {
      const modelUrl = `${baseUrl}models/${categoryId}/${opName}.onnx`
      sess = await getSession(modelUrl)
    }
    const inputs: Record<string, ort.Tensor> = {}

    for (const inputName of sess.inputNames) {
      if (inputName === 'input') {
        inputs[inputName] = new ort.Tensor('float32', currentTensor.data, currentTensor.dims)
      } else {
        const val = params?.[inputName] ?? paramMeta?.[inputName]?.default ?? 0
        inputs[inputName] = new ort.Tensor('float32', [val], [1])
      }
    }

    const t0 = performance.now()
    const result = await sess.run(inputs)
    timings.push({ nodeId: node.id, opName, ms: performance.now() - t0 })
    processedNodeIds.add(node.id)

    const outTensor = result[sess.outputNames[0]]
    // WASMヒープへの参照を保持しないようコピー
    const nextTensor: TensorData = {
      data: new Float32Array(outTensor.data as Float32Array),
      dims: [...outTensor.dims] as number[],
    }

    const skipPreview = categoryId === '10_ml_preprocess' && opName !== 'letterbox'
    const blob = skipPreview ? null : await tensorToBlob(nextTensor)
    previews.push({ nodeId: node.id, blob })

    return nextTensor
  } catch (e) {
    console.error(`推論エラー [${opName}]:`, e)
    return null
  }
}

async function runChain(msg: RunMsg): Promise<RunResultOut> {
  const { base, sourceNodeId, nodes, edges } = msg
  const inputTensor: TensorData = {
    data: msg.tensorData,
    dims: msg.tensorDims,
  }

  let currentId = sourceNodeId
  let currentTensor = inputTensor

  const previews: NodePreview[] = []
  const timings: TimingEntry[] = []
  const processedNodeIds = new Set<string>()

  while (true) {
    const outEdge = edges.find((e) => e.source === currentId && !e.hidden)
    if (!outEdge) break

    const nextNode = nodes.find((n) => n.id === outEdge.target)
    if (!nextNode) break

    // imageOutput
    if (nextNode.type === 'imageOutput') {
      const h = currentTensor.dims[0]
      const w = currentTensor.dims[1]
      const blob = await float32HWCToBlob(currentTensor.data, w, h)
      previews.push({ nodeId: nextNode.id, blob })
      break
    }

    // SubgraphNode（折りたたみ時）
    if (nextNode.type === 'subgraph') {
      const sgData = nextNode.data
      if (sgData.collapsed && sgData.childNodeIds && sgData.childNodeIds.length > 0) {
        let chainTensor = currentTensor
        for (const childId of sgData.childNodeIds) {
          const childNode = nodes.find((n) => n.id === childId)
          if (!childNode || childNode.type !== 'processing') break
          const result = await runProcessingNode(childNode, chainTensor, base, previews, timings, processedNodeIds)
          if (!result) break
          chainTensor = result
        }
        const allSkip = sgData.childNodeIds.every((cid) => {
          const cn = nodes.find((n) => n.id === cid)
          if (!cn || cn.type !== 'processing') return false
          return cn.data.categoryId === '10_ml_preprocess' && cn.data.opName !== 'letterbox'
        })
        const blob = allSkip ? null : await tensorToBlob(chainTensor)
        previews.push({ nodeId: nextNode.id, blob })
        currentId = nextNode.id
        currentTensor = chainTensor
        continue
      }
    }

    // processing
    if (nextNode.type !== 'processing') break

    const result = await runProcessingNode(nextNode, currentTensor, base, previews, timings, processedNodeIds)
    if (!result) break

    currentId = nextNode.id
    currentTensor = result
  }

  return {
    type: 'run-result',
    sourceNodeId,
    previews,
    timings,
    processedNodeIds: [...processedNodeIds],
  }
}

// ---------- プリロード ----------

async function preloadAll(msg: PreloadMsg) {
  const { meta } = msg
  // メインスレッドから受け取った絶対URLでWASMパスとbaseUrlを初期化
  baseUrl = msg.baseUrl
  // オブジェクト形式で wasm バイナリのパスのみ指定する
  // 文字列形式だと .mjs の動的 import が発生し、静的サーバーで MIME type エラーになる
  ort.env.wasm.wasmPaths = { wasm: baseUrl + 'ort-wasm-simd-threaded.wasm' }

  const urls: string[] = []
  for (const cat of meta.categories) {
    for (const model of cat.models) {
      urls.push(`${baseUrl}models/${cat.id}/${model}.onnx`)
    }
  }

  let loaded = 0
  const total = urls.length
  self.postMessage({ type: 'preload-progress', loaded: 0, total } as OutMsg)

  const CONCURRENCY = 4
  for (let i = 0; i < total; i += CONCURRENCY) {
    const batch = urls.slice(i, i + CONCURRENCY)
    await Promise.all(batch.map((url) => getSession(url).catch(() => {})))
    loaded = Math.min(i + CONCURRENCY, total)
    self.postMessage({ type: 'preload-progress', loaded, total } as OutMsg)
  }

  self.postMessage({ type: 'preload-done' } as OutMsg)
}

// ---------- メッセージハンドラ ----------

// 推論の直列化キュー（async onmessage は await 中に次のメッセージを処理してしまうため）
let runQueue: Promise<void> = Promise.resolve()

self.onmessage = (e: MessageEvent<InMsg>) => {
  const msg = e.data
  if (msg.type === 'preload') {
    runQueue = runQueue.then(() => preloadAll(msg))
  } else if (msg.type === 'run') {
    runQueue = runQueue.then(async () => {
      const result = await runChain(msg)
      self.postMessage(result)
    })
  } else if (msg.type === 'register-model') {
    runQueue = runQueue.then(async () => {
      try {
        const sessPromise = ort.InferenceSession.create(
          new Uint8Array(msg.modelData),
          { executionProviders: ['wasm'] },
        )
        const sess = await sessPromise
        // 入力が "input" 1つのみ対応
        if (sess.inputNames.length !== 1 || sess.inputNames[0] !== 'input') {
          self.postMessage({ type: 'model-registered', modelKey: msg.modelKey, ok: false })
          return
        }
        sessionCache.set(msg.modelKey, Promise.resolve(sess))
        self.postMessage({ type: 'model-registered', modelKey: msg.modelKey, ok: true })
      } catch (e) {
        console.error('Failed to register model:', e)
        self.postMessage({ type: 'model-registered', modelKey: msg.modelKey, ok: false })
      }
    })
  }
}
