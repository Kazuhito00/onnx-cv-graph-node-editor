import { useCallback } from 'react'
import { Handle, Position, useReactFlow } from '@xyflow/react'
import { markDirty } from '../hooks/useDirtyFlag'
import type { ProcessingNodeData } from '../types'

interface Props {
  id: string
  data: ProcessingNodeData
}

export default function ProcessingNode({ id, data }: Props) {
  const { setNodes } = useReactFlow()
  // スライダー変更: data.params を即時更新するだけ
  // 推論は InputImageNode の 50ms インターバルが駆動する propagateFrom が担う
  const handleParamChange = useCallback(
    (name: string, value: number) => {
      setNodes((ns) =>
        ns.map((n) =>
          n.id === id
            ? { ...n, data: { ...n.data, params: { ...(n.data as ProcessingNodeData).params, [name]: value } } }
            : n,
        ),
      )
      markDirty()
    },
    [id, setNodes],
  )

  const paramEntries = Object.entries(data.paramMeta)
  const showPreview = !(data.categoryId === '10_ml_preprocess' && data.opName !== 'letterbox')

  return (
    <div className="node node--processing">
      {data.comment && (
        <div className="node__comment">{data.comment}</div>
      )}
      <div className="node__header">{data.opName}</div>
      <div className="node__body">
        {showPreview && (
          <div className="node__preview-area">
            {data.previewUrl && (
              <img className="node__preview-area-img" src={data.previewUrl} alt="preview" />
            )}
          </div>
        )}
        {paramEntries.map(([name, meta]) => (
          <div key={name} className="node__param">
            <label className="node__param-label">
              {name}
              <span className="node__param-value">
                {(data.params[name] ?? meta.default).toFixed(2)}
              </span>
            </label>
            <div
              className="nodrag"
              onMouseDown={(e) => e.stopPropagation()}
              onPointerDown={(e) => e.stopPropagation()}
            >
              <input
                type="range"
                min={meta.min}
                max={meta.max}
                step={(meta.max - meta.min) / 100}
                value={data.params[name] ?? meta.default}
                onChange={(e) => handleParamChange(name, parseFloat(e.target.value))}
                className="node__slider"
              />
            </div>
          </div>
        ))}
      </div>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
