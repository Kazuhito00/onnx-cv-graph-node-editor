/**
 * ONNX エクスポートモーダル
 *
 * 全体グラフのチェーンと各 SubgraphNode を一覧表示し、
 * チェックボックスで選択したものを ONNX ファイルとしてダウンロードする。
 */

import { useState, useCallback } from 'react'
import { useReactFlow, type Node, type Edge } from '@xyflow/react'
import { mergeOnnxChain, type ChainModel, type MergeOptions } from '../utils/onnxMerge'
import type { ProcessingNodeData, SubgraphNodeData } from '../types'

interface Props {
  base: string
  onClose: () => void
}

interface ExportEntry {
  id: string
  label: string
  fileName: string
  models: ChainModel[]
}

/** base パスを絶対URLに解決する */
function resolveBaseUrl(base: string): string {
  const url = new URL(base, window.location.href)
  return url.href.endsWith('/') ? url.href : url.href + '/'
}

/** ノードから ChainModel を生成 */
function nodeToChainModel(node: Node, baseUrl: string): ChainModel | null {
  if (node.type !== 'processing') return null
  const d = node.data as ProcessingNodeData
  if (!d.opName || !d.categoryId) return null
  return {
    url: `${baseUrl}models/${d.categoryId}/${d.opName}.onnx`,
    params: { ...d.params },
  }
}

/** エッジを辿ってチェーン上の processing ノードを順序通り収集 */
function collectChainModels(
  startNodeId: string,
  nodes: Node[],
  edges: Edge[],
  baseUrl: string,
): ChainModel[] {
  const models: ChainModel[] = []
  let currentId = startNodeId

  while (true) {
    const outEdge = edges.find((e) => e.source === currentId && !e.hidden)
    if (!outEdge) break

    const nextNode = nodes.find((n) => n.id === outEdge.target)
    if (!nextNode) break

    if (nextNode.type === 'processing') {
      const m = nodeToChainModel(nextNode, baseUrl)
      if (m) models.push(m)
    } else if (nextNode.type === 'subgraph') {
      const sd = nextNode.data as SubgraphNodeData
      if (sd.childNodeIds) {
        for (const childId of sd.childNodeIds) {
          const childNode = nodes.find((n) => n.id === childId)
          if (childNode) {
            const m = nodeToChainModel(childNode, baseUrl)
            if (m) models.push(m)
          }
        }
      }
    } else if (nextNode.type === 'imageOutput') {
      break
    }

    currentId = nextNode.id
  }

  return models
}

/** エクスポート対象の一覧を構築 */
function buildExportEntries(nodes: Node[], edges: Edge[], baseUrl: string): ExportEntry[] {
  const entries: ExportEntry[] = []

  // 全体グラフ: 各 InputNode からのチェーン
  const inputNodes = nodes.filter((n) => n.type === 'imageInput')
  inputNodes.forEach((inputNode, i) => {
    const models = collectChainModels(inputNode.id, nodes, edges, baseUrl)
    if (models.length > 0) {
      const label = inputNodes.length === 1
        ? '全体グラフ / Full Graph'
        : `全体グラフ ${i + 1} / Full Graph ${i + 1}`
      entries.push({
        id: `chain_${inputNode.id}`,
        label,
        fileName: inputNodes.length === 1 ? 'full_graph.onnx' : `full_graph_${i + 1}.onnx`,
        models,
      })
    }
  })

  // 各 SubgraphNode
  const subgraphNodes = nodes.filter((n) => n.type === 'subgraph')
  for (const sg of subgraphNodes) {
    const sd = sg.data as SubgraphNodeData
    const models: ChainModel[] = []
    if (sd.childNodeIds) {
      for (const childId of sd.childNodeIds) {
        const childNode = nodes.find((n) => n.id === childId)
        if (childNode) {
          const m = nodeToChainModel(childNode, baseUrl)
          if (m) models.push(m)
        }
      }
    }
    if (models.length > 0) {
      const name = sd.onnxExportName || sd.name || 'subgraph'
      const fileName = name.endsWith('.onnx') ? name : `${name}.onnx`
      entries.push({
        id: `sg_${sg.id}`,
        label: sd.name,
        fileName,
        models,
      })
    }
  }

  return entries
}

export default function OnnxExportModal({ base, onClose }: Props) {
  const { getNodes, getEdges } = useReactFlow()
  const baseUrl = resolveBaseUrl(base)
  const entries = buildExportEntries(getNodes(), getEdges(), baseUrl)

  const [checked, setChecked] = useState<Set<string>>(() => new Set(entries.filter((e) => e.id.startsWith('chain_')).map((e) => e.id)))
  const [bakeParams, setBakeParams] = useState(true)
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const toggle = useCallback((id: string) => {
    setChecked((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const handleExport = useCallback(async () => {
    setExporting(true)
    setError(null)
    try {
      const selected = entries.filter((e) => checked.has(e.id))
      const opts: MergeOptions = { bakeParams }
      for (const entry of selected) {
        const data = await mergeOnnxChain(entry.models, entry.fileName.replace('.onnx', ''), opts)
        const blob = new Blob([data], { type: 'application/octet-stream' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = entry.fileName
        a.click()
        URL.revokeObjectURL(url)
      }
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setExporting(false)
    }
  }, [entries, checked, bakeParams, onClose])

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal__header">ONNX Export</div>
        <div className="modal__body">
          {entries.length === 0 ? (
            <div className="modal__empty">エクスポート可能なチェーンがありません<br />No exportable chains found</div>
          ) : (
            <div className="modal__list">
              {entries.map((entry) => (
                <label key={entry.id} className="modal__item">
                  <input
                    type="checkbox"
                    checked={checked.has(entry.id)}
                    onChange={() => toggle(entry.id)}
                  />
                  <div className="modal__item-info">
                    <div className="modal__item-label">{entry.label}</div>
                    <div className="modal__item-file">{entry.fileName} ({entry.models.length} models)</div>
                  </div>
                </label>
              ))}
            </div>
          )}
          <label className="modal__option">
            <input
              type="checkbox"
              checked={bakeParams}
              onChange={() => setBakeParams((v) => !v)}
            />
            <span>パラメータを固定値として埋め込む / Bake parameters</span>
          </label>
          {error && <div className="modal__error">{error}</div>}
        </div>
        <div className="modal__footer">
          <button className="modal__btn modal__btn--cancel" onClick={onClose}>Cancel</button>
          <button
            className="modal__btn"
            onClick={handleExport}
            disabled={exporting || checked.size === 0}
          >
            {exporting ? 'Exporting...' : 'Export'}
          </button>
        </div>
      </div>
    </div>
  )
}
