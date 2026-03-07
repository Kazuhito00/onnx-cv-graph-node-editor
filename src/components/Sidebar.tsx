import { useState } from 'react'
import type { ModelsMeta, DragModelPayload } from '../types'
import { DRAG_TYPE_MODEL, DRAG_TYPE_INPUT, DRAG_TYPE_OUTPUT, DRAG_TYPE_UPLOAD_ONNX } from '../types'

interface Props {
  meta: ModelsMeta | null
  onExport?: () => void
  onImport?: () => void
  onOnnxExport?: () => void
  onAutoLayout?: () => void
}

export default function Sidebar({ meta, onExport, onImport, onOnnxExport, onAutoLayout }: Props) {
  const [openCategories, setOpenCategories] = useState<Set<string>>(new Set(['__io__']))
  const [search, setSearch] = useState('')
  const searchLower = search.toLowerCase()

  const toggleCategory = (id: string, e: React.MouseEvent) => {
    const el = e.currentTarget as HTMLElement
    setOpenCategories((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
        requestAnimationFrame(() => {
          el.scrollIntoView({ behavior: 'smooth', block: 'start' })
        })
      }
      return next
    })
  }

  const onDragStartModel = (e: React.DragEvent, payload: DragModelPayload) => {
    e.dataTransfer.setData(DRAG_TYPE_MODEL, JSON.stringify(payload))
    e.dataTransfer.effectAllowed = 'move'
  }

  const onDragStartInput = (e: React.DragEvent) => {
    e.dataTransfer.setData(DRAG_TYPE_INPUT, '1')
    e.dataTransfer.effectAllowed = 'move'
  }

  const onDragStartOutput = (e: React.DragEvent) => {
    e.dataTransfer.setData(DRAG_TYPE_OUTPUT, '1')
    e.dataTransfer.effectAllowed = 'move'
  }

  const onDragStartUploadOnnx = (e: React.DragEvent) => {
    e.dataTransfer.setData(DRAG_TYPE_UPLOAD_ONNX, '1')
    e.dataTransfer.effectAllowed = 'move'
  }

  const isSearching = searchLower.length > 0

  // IO ノードの検索フィルタ
  const ioItems = [
    { label: 'Input', onDragStart: onDragStartInput },
    { label: 'Output', onDragStart: onDragStartOutput },
  ].filter((item) => !isSearching || item.label.toLowerCase().includes(searchLower))

  // Other カテゴリの検索フィルタ
  const otherItems = [
    { label: 'Upload ONNX', onDragStart: onDragStartUploadOnnx },
  ].filter((item) => !isSearching || item.label.toLowerCase().includes(searchLower))

  // カテゴリ別モデルの検索フィルタ
  const filteredCategories = (meta?.categories.filter((cat) => !cat.hidden) ?? [])
    .map((cat) => ({
      ...cat,
      filteredModels: isSearching
        ? cat.models.filter((op) => op.toLowerCase().includes(searchLower))
        : cat.models,
    }))
    .filter((cat) => cat.filteredModels.length > 0)

  return (
    <aside className="sidebar">
      <input
        className="sidebar__search"
        type="text"
        placeholder="検索 / Search..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
      <div className="sidebar__divider" />

      {/* 入力・出力ノード */}
      {ioItems.length > 0 && (
        <div className="sidebar__category">
          <button
            className="sidebar__category-header"
            onClick={(e) => toggleCategory('__io__', e)}
          >
            <span>{openCategories.has('__io__') || isSearching ? '▼' : '▶'}</span>
            <span className="sidebar__category-label">
              <span>入力・出力ノード</span>
              <span className="sidebar__category-label-en">Input / Output</span>
            </span>
          </button>
          {(openCategories.has('__io__') || isSearching) && (
            <div className="sidebar__model-list">
              {ioItems.map((item) => (
                <div
                  key={item.label}
                  className="sidebar__model-card"
                  draggable
                  onDragStart={item.onDragStart}
                >
                  {item.label}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* カテゴリ別モデル */}
      {filteredCategories.map((cat) => (
        <div key={cat.id} className="sidebar__category">
          <button
            className="sidebar__category-header"
            onClick={(e) => toggleCategory(cat.id, e)}
          >
            <span>{openCategories.has(cat.id) || isSearching ? '▼' : '▶'}</span>
            <span className="sidebar__category-label">
              <span>{cat.label_ja}</span>
              <span className="sidebar__category-label-en">{cat.label_en}</span>
            </span>
          </button>
          {(openCategories.has(cat.id) || isSearching) && (
            <div className="sidebar__model-list">
              {cat.filteredModels.map((opName) => (
                <div
                  key={opName}
                  className="sidebar__model-card"
                  draggable
                  onDragStart={(e) =>
                    onDragStartModel(e, { opName, categoryId: cat.id })
                  }
                >
                  {opName}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}

      {/* その他 */}
      {otherItems.length > 0 && (
        <div className="sidebar__category">
          <button
            className="sidebar__category-header"
            onClick={(e) => toggleCategory('__other__', e)}
          >
            <span>{openCategories.has('__other__') || isSearching ? '▼' : '▶'}</span>
            <span className="sidebar__category-label">
              <span>その他</span>
              <span className="sidebar__category-label-en">Other</span>
            </span>
          </button>
          {(openCategories.has('__other__') || isSearching) && (
            <div className="sidebar__model-list">
              {otherItems.map((item) => (
                <div
                  key={item.label}
                  className="sidebar__model-card"
                  draggable
                  onDragStart={item.onDragStart}
                >
                  {item.label}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="sidebar__actions">
        <button className="sidebar__action-btn sidebar__action-btn--layout" onClick={onAutoLayout} title="Ctrl+A">Auto Layout</button>
        <button className="sidebar__action-btn sidebar__action-btn--onnx" onClick={onOnnxExport} title="Ctrl+E">ONNX Export</button>
        <div className="sidebar__actions-row">
          <button className="sidebar__action-btn sidebar__action-btn--export" onClick={onExport} title="Ctrl+S">Save</button>
          <button className="sidebar__action-btn sidebar__action-btn--import" onClick={onImport} title="Ctrl+L">Load</button>
        </div>
      </div>
    </aside>
  )
}
