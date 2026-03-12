import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type Connection,
  type Node,
  type Edge,
  type NodeChange,
  type EdgeChange,
  ReactFlowProvider,
  Controls,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import Sidebar from './components/Sidebar'
import ContextMenu from './components/ContextMenu'
import TimingOverlay from './components/TimingOverlay'
import OnnxExportModal from './components/OnnxExportModal'
import InputImageNode from './nodes/InputImageNode'
import ProcessingNode from './nodes/ProcessingNode'
import OutputImageNode from './nodes/OutputImageNode'
import SubgraphNode from './nodes/SubgraphNode'

import { useModelsMeta, toParamMeta } from './hooks/useModelsMeta'
import { useSubgraph } from './hooks/useSubgraph'
import { preloadWithWorker, registerUploadedModel } from './hooks/useInferenceWorker'
import { markDirty } from './hooks/useDirtyFlag'
import { useUndoRedo } from './hooks/useUndoRedo'
import { clearTimingForNode } from './hooks/useTiming'
import {
  DRAG_TYPE_MODEL,
  DRAG_TYPE_INPUT,
  DRAG_TYPE_OUTPUT,
  DRAG_TYPE_UPLOAD_ONNX,
  type ContextMenuState,
  type ProcessingNodeData,
  type SubgraphNodeData,
} from './types'

const BASE = './'
const NODE_TYPES = {
  imageInput: InputImageNode,
  processing: ProcessingNode,
  imageOutput: OutputImageNode,
  subgraph: SubgraphNode,
}

// InputImageNode ドロップ時に自動配置する前処理3ノード（右方向）
const PREPROCESS_NODES = [
  { opName: 'batch_unsqueeze_nhwc', categoryId: '10_ml_preprocess' },
  { opName: 'hwc_to_chw',           categoryId: '10_ml_preprocess' },
  { opName: 'scale_from_255',        categoryId: '10_ml_preprocess' },
]

// OutputImageNode ドロップ時に自動配置する後処理3ノード（左方向）
const POSTPROCESS_NODES = [
  { opName: 'scale_to_255',        categoryId: '10_ml_preprocess' },
  { opName: 'chw_to_hwc',          categoryId: '10_ml_preprocess' },
  { opName: 'batch_squeeze_nhwc',  categoryId: '10_ml_preprocess' },
]

let nodeIdCounter = 1
const newId = () => `n${nodeIdCounter++}`

function AppInner() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null)
  const [showOnnxExport, setShowOnnxExport] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const flowWrapper = useRef<HTMLDivElement>(null)
  const importRef = useRef<HTMLInputElement>(null)
  const { screenToFlowPosition, getNodes, getEdges, fitView } = useReactFlow()
  const { meta } = useModelsMeta(BASE)
  const { groupNodes, ungroupNode, toggleCollapse } = useSubgraph()
  const { saveSnapshot, undo, redo } = useUndoRedo(getNodes, getEdges, setNodes, setEdges)

  // ノード変更: 削除時にスナップショット保存
  const handleNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const removes = changes.filter((c) => c.type === 'remove')
      if (removes.length > 0) {
        saveSnapshot()
        removes.forEach((c) => clearTimingForNode(c.id))
      }
      onNodesChange(changes)
    },
    [onNodesChange, saveSnapshot],
  )

  // ドラッグ開始時にスナップショット保存（移動前の位置を記録）
  const onNodeDragStart = useCallback(() => {
    saveSnapshot()
  }, [saveSnapshot])

  // エッジ変更: 削除時にスナップショット保存
  const handleEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const hasRemove = changes.some((c) => c.type === 'remove')
      if (hasRemove) saveSnapshot()
      onEdgesChange(changes)
    },
    [onEdgesChange, saveSnapshot],
  )

  // Export: ノードとエッジをJSONファイルとしてダウンロード
  const handleExport = useCallback(() => {
    // previewUrl と不要なランタイムデータを除外
    const cleanNodes = nodes.map((n) => {
      const rest = { ...n.data } as Record<string, unknown>
      delete rest.previewUrl
      return { ...n, data: rest }
    })
    const data = JSON.stringify({ nodes: cleanNodes, edges }, null, 2)
    const blob = new Blob([data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const now = new Date()
    const ts = now.getFullYear().toString()
      + String(now.getMonth() + 1).padStart(2, '0')
      + String(now.getDate()).padStart(2, '0')
      + '_'
      + String(now.getHours()).padStart(2, '0')
      + String(now.getMinutes()).padStart(2, '0')
      + String(now.getSeconds()).padStart(2, '0')
    a.download = `graph_${ts}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [nodes, edges])

  // Import: JSONファイルからノードとエッジを読み込み
  const handleImport = useCallback((file: File) => {
    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const json = JSON.parse(ev.target?.result as string)
        if (json.nodes && json.edges) {
          saveSnapshot()
          setNodes(json.nodes)
          setEdges(json.edges)
          // nodeIdCounter を復元してID衝突を防ぐ
          const maxId = json.nodes.reduce((max: number, n: Node) => {
            const num = parseInt(n.id.replace(/\D/g, ''), 10)
            return isNaN(num) ? max : Math.max(max, num)
          }, 0)
          nodeIdCounter = maxId + 1
        }
      } catch (e) {
        console.error('Import failed:', e)
      }
    }
    reader.readAsText(file)
  }, [saveSnapshot, setNodes, setEdges])

  // トースト自動消去 + どこクリックでも消去
  useEffect(() => {
    if (!toast) return
    const timer = setTimeout(() => setToast(null), 5000)
    const dismiss = () => setToast(null)
    window.addEventListener('pointerdown', dismiss)
    return () => {
      clearTimeout(timer)
      window.removeEventListener('pointerdown', dismiss)
    }
  }, [toast])

  // オートレイアウト: チェーンを左→右にアニメーション付きで自動配置
  const handleAutoLayout = useCallback(() => {
    saveSnapshot()
    const currentNodes = getNodes()
    const currentEdges = getEdges()
    const visibleNodes = currentNodes.filter((n) => !n.hidden)
    const visibleEdges = currentEdges.filter((e) => !e.hidden)

    const hasIncoming = new Set(visibleEdges.map((e) => e.target))
    const startNodes = visibleNodes.filter((n) => !hasIncoming.has(n.id))

    const NODE_W = 260
    const NODE_H = 200
    const targetPositions = new Map<string, { x: number; y: number }>()
    let chainIndex = 0

    for (const start of startNodes) {
      let currentId: string | null = start.id
      let col = 0
      const baseY = chainIndex * (NODE_H + 40)

      while (currentId) {
        if (!targetPositions.has(currentId)) {
          targetPositions.set(currentId, { x: col * NODE_W, y: baseY })
        }
        col++
        const outEdge = visibleEdges.find((e) => e.source === currentId)
        currentId = outEdge ? outEdge.target : null
      }
      chainIndex++
    }

    for (const node of visibleNodes) {
      if (!targetPositions.has(node.id)) {
        targetPositions.set(node.id, { x: 0, y: chainIndex * (NODE_H + 40) })
        chainIndex++
      }
    }

    // アニメーション
    const duration = 300
    const startTime = performance.now()
    const startPositions = new Map<string, { x: number; y: number }>()
    for (const node of currentNodes) {
      if (targetPositions.has(node.id)) {
        startPositions.set(node.id, { ...node.position })
      }
    }

    const animate = (now: number) => {
      const t = Math.min((now - startTime) / duration, 1)
      const ease = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2

      setNodes((ns) =>
        ns.map((n) => {
          const from = startPositions.get(n.id)
          const to = targetPositions.get(n.id)
          if (!from || !to) return n
          return {
            ...n,
            position: {
              x: from.x + (to.x - from.x) * ease,
              y: from.y + (to.y - from.y) * ease,
            },
          }
        }),
      )

      if (t < 1) {
        requestAnimationFrame(animate)
      } else {
        setTimeout(() => fitView({ duration: 300, padding: 0.1 }), 50)
      }
    }

    requestAnimationFrame(animate)
  }, [saveSnapshot, getNodes, getEdges, setNodes, fitView])

  // Ctrl+S: Export, Ctrl+L: Import, Ctrl+Z: Undo, Ctrl+Y: Redo, Ctrl+A: Auto Layout
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === 's') {
        e.preventDefault()
        handleExport()
      } else if (e.ctrlKey && e.key === 'l') {
        e.preventDefault()
        importRef.current?.click()
      } else if (e.ctrlKey && e.key === 'z') {
        e.preventDefault()
        undo()
      } else if (e.ctrlKey && e.key === 'y') {
        e.preventDefault()
        redo()
      } else if (e.ctrlKey && e.key === 'a') {
        e.preventDefault()
        handleAutoLayout()
      } else if (e.ctrlKey && e.key === 'e') {
        e.preventDefault()
        setShowOnnxExport(true)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [handleExport, undo, redo, handleAutoLayout])

  // 1対1エッジ制約: 既に接続があるハンドルへの接続を拒否
  // hidden エッジと、存在しないノードを参照する孤立エッジは除外
  const isValidConnection = useCallback(
    (connection: Edge | Connection) => {
      const nodeIds = new Set(nodes.map((n) => n.id))
      const activeEdges = edges.filter(
        (e) => !e.hidden && nodeIds.has(e.source) && nodeIds.has(e.target),
      )
      const hasSourceEdge = activeEdges.some((e) => e.source === connection.source)
      const hasTargetEdge = activeEdges.some((e) => e.target === connection.target)
      return !hasSourceEdge && !hasTargetEdge
    },
    [nodes, edges],
  )

  const onConnect = useCallback(
    (connection: Connection) => {
      saveSnapshot()
      setEdges((eds) => addEdge(connection, eds))
      markDirty()
    },
    [saveSnapshot, setEdges],
  )

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      if (!flowWrapper.current) return

      const { x, y } = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      saveSnapshot()

      // InputImageNode ドロップ: 前処理3ノードをSubgraphNodeとして自動配置
      if (e.dataTransfer.getData(DRAG_TYPE_INPUT)) {
        const inputId = newId()
        const inputNode: Node = {
          id: inputId,
          type: 'imageInput',
          position: { x, y },
          data: {},
        }
        // 子ノード（非表示）
        const prepNodes: Node[] = PREPROCESS_NODES.map((p) => ({
          id: newId(),
          type: 'processing',
          position: { x: 0, y: 0 },
          hidden: true,
          data: {
            opName: p.opName,
            categoryId: p.categoryId,
            params: {},
            paramMeta: {},
          } as ProcessingNodeData,
        }))
        const childNodeIds = prepNodes.map((n) => n.id)
        // 内部エッジ（非表示）
        const internalEdges: Edge[] = prepNodes.slice(0, -1).map((n, i) => ({
          id: `e${n.id}-${prepNodes[i + 1].id}`,
          source: n.id,
          target: prepNodes[i + 1].id,
          hidden: true,
        }))
        const childEdgeIds = internalEdges.map((e) => e.id)
        // SubgraphNode
        const sgId = newId()
        const sgNode: Node = {
          id: sgId,
          type: 'subgraph',
          position: { x: x + 250, y },
          data: {
            name: 'preprocess',
            onnxExportName: '',
            collapsed: true,
            childNodeIds,
            childEdgeIds,
            externalSourceEdge: undefined,
            externalTargetEdge: undefined,
          } as SubgraphNodeData,
        }
        // Input → SubgraphNode エッジ
        const connectEdge: Edge = {
          id: `e${inputId}-${sgId}`,
          source: inputId,
          target: sgId,
        }
        // externalSourceEdge を設定（Input→SubgraphNodeのエッジがfirstChildに対応）
        ;(sgNode.data as SubgraphNodeData).externalSourceEdge = {
          edgeId: connectEdge.id,
          originalTarget: childNodeIds[0],
        }
        setNodes((ns) => [...ns, inputNode, ...prepNodes, sgNode])
        setEdges((es) => [...es, ...internalEdges, connectEdge])
        return
      }

      // OutputImageNode ドロップ: 後処理3ノードをSubgraphNodeとして自動配置
      if (e.dataTransfer.getData(DRAG_TYPE_OUTPUT)) {
        const outputId = newId()
        const outputNode: Node = {
          id: outputId,
          type: 'imageOutput',
          position: { x, y },
          data: {},
        }
        // 子ノード（非表示）
        const postNodes: Node[] = POSTPROCESS_NODES.map((p) => ({
          id: newId(),
          type: 'processing',
          position: { x: 0, y: 0 },
          hidden: true,
          data: {
            opName: p.opName,
            categoryId: p.categoryId,
            params: {},
            paramMeta: {},
          } as ProcessingNodeData,
        }))
        const childNodeIds = postNodes.map((n) => n.id)
        // 内部エッジ（非表示）
        const internalEdges: Edge[] = postNodes.slice(0, -1).map((n, i) => ({
          id: `e${n.id}-${postNodes[i + 1].id}`,
          source: n.id,
          target: postNodes[i + 1].id,
          hidden: true,
        }))
        const childEdgeIds = internalEdges.map((e) => e.id)
        // SubgraphNode
        const sgId = newId()
        // OutputNode は preview-area--large (16:9) があるため背が高い
        // SubgraphNode (preview なし, 子3つ) との高さ差を補正して下揃え
        const sgNode: Node = {
          id: sgId,
          type: 'subgraph',
          position: { x: x - 250, y: y + 50 },
          data: {
            name: 'postprocess',
            onnxExportName: '',
            collapsed: true,
            childNodeIds,
            childEdgeIds,
            externalSourceEdge: undefined,
            externalTargetEdge: undefined,
          } as SubgraphNodeData,
        }
        // SubgraphNode → Output エッジ
        const connectEdge: Edge = {
          id: `e${sgId}-${outputId}`,
          source: sgId,
          target: outputId,
        }
        // externalTargetEdge を設定（SubgraphNode→OutputのエッジがlastChildに対応）
        ;(sgNode.data as SubgraphNodeData).externalTargetEdge = {
          edgeId: connectEdge.id,
          originalSource: childNodeIds[childNodeIds.length - 1],
        }
        setNodes((ns) => [...ns, ...postNodes, sgNode, outputNode])
        setEdges((es) => [...es, ...internalEdges, connectEdge])
        return
      }

      // Upload ONNX ドロップ
      if (e.dataTransfer.getData(DRAG_TYPE_UPLOAD_ONNX)) {
        const input = document.createElement('input')
        input.type = 'file'
        input.accept = '.onnx'
        input.onchange = async () => {
          const file = input.files?.[0]
          if (!file) return
          const arrayBuffer = await file.arrayBuffer()
          const modelKey = `upload_${Date.now()}_${file.name}`
          const opName = file.name.replace(/\.onnx$/i, '')
          const paramInputs = await registerUploadedModel(modelKey, arrayBuffer)
          if (paramInputs === null) {
            setToast('このモデルは対応していません。入力が "input" 1つのみのモデルに対応しています。\nThis model is not supported. Only models with a single "input" are supported.')
            return
          }
          const id = newId()
          setNodes((ns) => [
            ...ns,
            {
              id,
              type: 'processing',
              position: { x, y },
              data: { opName, categoryId: '__upload__', params: {}, paramMeta: {}, uploadModelKey: modelKey } as ProcessingNodeData,
            },
          ])
        }
        input.click()
        return
      }

      // ProcessingNode ドロップ
      const modelRaw = e.dataTransfer.getData(DRAG_TYPE_MODEL)
      if (modelRaw) {
        const { opName, categoryId } = JSON.parse(modelRaw)
        const paramRaw = meta?.params[opName] ?? {}
        const paramMeta = toParamMeta(paramRaw)
        const params = Object.fromEntries(
          Object.entries(paramMeta).map(([k, v]) => [k, v.default]),
        )
        const id = newId()
        setNodes((ns) => [
          ...ns,
          {
            id,
            type: 'processing',
            position: { x, y },
            data: { opName, categoryId, params, paramMeta } as ProcessingNodeData,
          },
        ])
      }
    },
    [meta, saveSnapshot, setNodes, setEdges, screenToFlowPosition],
  )

  const onNodeContextMenu = useCallback(
    (e: React.MouseEvent, node: Node) => {
      e.preventDefault()
      setContextMenu({
        nodeId: node.id,
        nodeType: node.type ?? '',
        x: e.clientX,
        y: e.clientY,
      })
    },
    [],
  )

  return (
    <div className="app">
      <Sidebar meta={meta} onExport={handleExport} onImport={() => importRef.current?.click()} onOnnxExport={() => setShowOnnxExport(true)} onAutoLayout={handleAutoLayout} />
      <input
        ref={importRef}
        type="file"
        accept=".json"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) handleImport(file)
          e.target.value = ''
        }}
      />
      <div className="flow-wrapper" ref={flowWrapper}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          onNodesChange={handleNodesChange}
          onEdgesChange={handleEdgesChange}
          onConnect={onConnect}
          onNodeDragStart={onNodeDragStart}
          isValidConnection={isValidConnection}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onNodeContextMenu={onNodeContextMenu}
          onPaneClick={() => { setContextMenu(null); setToast(null) }}
          selectionOnDrag
          defaultEdgeOptions={{ type: 'straight' }}
          deleteKeyCode={['Backspace', 'Delete']}
          colorMode="light"
          minZoom={0.1}
          maxZoom={8}
          defaultViewport={{ x: 0, y: 0, zoom: 1 }}
        >
          <Background variant={BackgroundVariant.Cross} gap={40} size={9} color="#e0e0e0" />
          <Controls showZoom={false} />
        </ReactFlow>
        <TimingOverlay />
        {showOnnxExport && (
          <OnnxExportModal base={BASE} onClose={() => setShowOnnxExport(false)} />
        )}
        {toast && (
          <div className="toast" onClick={() => setToast(null)}>
            <div className="toast__content">
              <div className="toast__title">Not Supported</div>
              <div className="toast__message">{toast}</div>
            </div>
          </div>
        )}
        {contextMenu && (
          <ContextMenu
            menu={contextMenu}
            onClose={() => setContextMenu(null)}
            selectedNodeIds={getNodes().filter((n) => n.selected).map((n) => n.id)}
            onGroup={groupNodes}
            onUngroup={ungroupNode}
            onToggleCollapse={toggleCollapse}
            onBeforeAction={saveSnapshot}
          />
        )}
      </div>
    </div>
  )
}

function LoadingScreen({ loaded, total }: { loaded: number; total: number }) {
  const pct = total > 0 ? Math.round((loaded / total) * 100) : 0
  return (
    <div className="loading-screen">
      <div className="loading-screen__content">
        <div className="loading-screen__title">Loading Models...</div>
        <div className="loading-screen__bar-bg">
          <div className="loading-screen__bar-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="loading-screen__text">{loaded} / {total} ({pct}%)</div>
      </div>
    </div>
  )
}

export default function App() {
  const { meta } = useModelsMeta(BASE)
  const [ready, setReady] = useState(false)
  const [progress, setProgress] = useState({ loaded: 0, total: 0 })

  useEffect(() => {
    if (!meta) return
    preloadWithWorker(BASE, meta, (loaded, total) => {
      setProgress({ loaded, total })
    }).then(() => setReady(true))
  }, [meta])

  if (!ready) {
    return <LoadingScreen loaded={progress.loaded} total={progress.total} />
  }

  return (
    <ReactFlowProvider>
      <AppInner />
    </ReactFlowProvider>
  )
}
