// ---------- モデルメタデータ ----------

export interface ParamMeta {
  min: number
  max: number
  default: number
}

export interface CategoryMeta {
  id: string
  label_ja: string
  label_en: string
  models: string[]
  hidden?: boolean
}

export interface ModelsMeta {
  categories: CategoryMeta[]
  params: Record<string, Record<string, [number, number, number]>>
}

// ---------- テンソルデータ ----------

export interface TensorData {
  data: Float32Array
  dims: number[]  // NCHW: [1, 3, H, W] が基本だが ml_preprocess ノード通過後は異なる
}

// ---------- ノード data 型 ----------

// ReactFlow の Node<data> は data: Record<string, unknown> を要求するため
// インデックスシグネチャを追加している
export interface BaseNodeData extends Record<string, unknown> {
  comment?: string
}

export interface InputImageNodeData extends BaseNodeData {
  previewUrl?: string
}

export interface ProcessingNodeData extends BaseNodeData {
  opName: string
  categoryId: string
  params: Record<string, number>
  paramMeta: Record<string, ParamMeta>
  uploadModelKey?: string
  previewUrl?: string
}

export interface OutputImageNodeData extends BaseNodeData {
  previewUrl?: string
}

export interface SubgraphNodeData extends BaseNodeData {
  name: string
  onnxExportName: string
  collapsed: boolean
  childNodeIds: string[]
  childEdgeIds: string[]
  // 折りたたみ時のエッジリルート用: 元の外部接続情報
  externalSourceEdge?: { edgeId: string; originalTarget: string }
  externalTargetEdge?: { edgeId: string; originalSource: string }
  previewUrl?: string
}

// ---------- コンテキストメニュー ----------

export interface ContextMenuState {
  nodeId: string
  nodeType: string
  x: number
  y: number
}

// ---------- サイドバードラッグデータ ----------

export const DRAG_TYPE_MODEL = 'application/onnx-model'
export const DRAG_TYPE_INPUT = 'application/onnx-input'
export const DRAG_TYPE_OUTPUT = 'application/onnx-output'
export const DRAG_TYPE_UPLOAD_ONNX = 'application/onnx-upload'

export interface DragModelPayload {
  opName: string
  categoryId: string
}
