import { Handle, Position } from '@xyflow/react'
import type { OutputImageNodeData } from '../types'

interface Props {
  id: string
  data: OutputImageNodeData
}

export default function OutputImageNode({ data }: Props) {
  return (
    <div className="node node--output">
      {data.comment && (
        <div className="node__comment">{data.comment}</div>
      )}
      <div className="node__header">Output</div>
      <div className="node__body">
        <div className="node__preview-area node__preview-area--large">
          {data.previewUrl && (
            <img className="node__preview-area-img" src={data.previewUrl} alt="output" />
          )}
        </div>
      </div>
      <Handle type="target" position={Position.Left} />
    </div>
  )
}
