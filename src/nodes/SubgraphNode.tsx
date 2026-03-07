import { Handle, Position, useReactFlow } from '@xyflow/react'
import type { SubgraphNodeData, ProcessingNodeData } from '../types'

interface Props {
  id: string
  data: SubgraphNodeData
}

export default function SubgraphNode({ data }: Props) {
  const { getNodes } = useReactFlow()
  const allNodes = getNodes()

  // 子ノードの opName リストを取得
  const childNames = data.childNodeIds
    .map((id) => {
      const n = allNodes.find((node) => node.id === id)
      return (n?.data as ProcessingNodeData)?.opName ?? id
    })

  // 全子ノードがプレビュー非表示対象かどうか
  const allSkipPreview = data.childNodeIds.every((id) => {
    const n = allNodes.find((node) => node.id === id)
    if (!n || n.type !== 'processing') return false
    const d = n.data as ProcessingNodeData
    return d.categoryId === '10_ml_preprocess' && d.opName !== 'letterbox'
  })

  return (
    <div className="node node--subgraph">
      {data.comment && (
        <div className="node__comment">{data.comment}</div>
      )}
      <div className="node__header">
        {data.name}
      </div>
      <div className="node__body">
        {!allSkipPreview && (
          <div className="node__preview-area">
            {data.previewUrl && (
              <img className="node__preview-area-img" src={data.previewUrl} alt="preview" />
            )}
          </div>
        )}
        <div className="node__child-list">
          {childNames.map((name, i) => (
            <div key={i} className="node__child-item">{name}</div>
          ))}
        </div>
      </div>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
