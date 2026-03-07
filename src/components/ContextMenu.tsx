import { useEffect, useRef, useState } from 'react'
import { useReactFlow } from '@xyflow/react'
import type { ContextMenuState } from '../types'

interface Props {
  menu: ContextMenuState
  onClose: () => void
  selectedNodeIds?: string[]
  onGroup?: (nodeIds: string[]) => void
  onUngroup?: (nodeId: string) => void
  onToggleCollapse?: (nodeId: string) => void
  onBeforeAction?: () => void
}

export default function ContextMenu({
  menu,
  onClose,
  selectedNodeIds = [],
  onGroup,
  onUngroup,
  onToggleCollapse,
  onBeforeAction,
}: Props) {
  const { getNode, setNodes } = useReactFlow()
  const ref = useRef<HTMLDivElement>(null)
  const [editingComment, setEditingComment] = useState(false)
  const [commentText, setCommentText] = useState('')
  const [editingName, setEditingName] = useState(false)
  const [nameText, setNameText] = useState('')

  const node = getNode(menu.nodeId)
  const currentComment = (node?.data as { comment?: string })?.comment ?? ''
  const isSubgraph = menu.nodeType === 'subgraph'
  const collapsed = isSubgraph ? (node?.data as { collapsed?: boolean })?.collapsed : false
  const currentName = isSubgraph ? (node?.data as { name?: string })?.name ?? '' : ''

  // 複数選択されている processing ノード（右クリック対象含む）
  const groupCandidates = selectedNodeIds.length > 1
    ? selectedNodeIds
    : []

  useEffect(() => {
    setCommentText(currentComment)
  }, [currentComment])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose()
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [onClose])

  const saveComment = () => {
    onBeforeAction?.()
    setNodes((nodes) =>
      nodes.map((n) =>
        n.id === menu.nodeId
          ? { ...n, data: { ...n.data, comment: commentText || undefined } }
          : n,
      ),
    )
    setEditingComment(false)
    onClose()
  }

  const saveName = () => {
    onBeforeAction?.()
    setNodes((nodes) =>
      nodes.map((n) =>
        n.id === menu.nodeId
          ? { ...n, data: { ...n.data, name: nameText || 'Group' } }
          : n,
      ),
    )
    setEditingName(false)
    onClose()
  }

  const deleteComment = () => {
    onBeforeAction?.()
    setNodes((nodes) =>
      nodes.map((n) =>
        n.id === menu.nodeId
          ? { ...n, data: { ...n.data, comment: undefined } }
          : n,
      ),
    )
    onClose()
  }

  return (
    <div
      ref={ref}
      className="context-menu"
      style={{ left: menu.x, top: menu.y }}
    >
      {editingComment ? (
        <div className="context-menu__comment-editor">
          <textarea
            className="context-menu__textarea"
            value={commentText}
            onChange={(e) => setCommentText(e.target.value)}
            autoFocus
            rows={3}
            placeholder="コメントを入力..."
          />
          <div className="context-menu__comment-actions">
            <button className="context-menu__btn" onClick={saveComment}>保存</button>
            <button className="context-menu__btn context-menu__btn--cancel" onClick={() => setEditingComment(false)}>キャンセル</button>
          </div>
        </div>
      ) : (
        <>
          <button
            className="context-menu__item"
            onClick={() => setEditingComment(true)}
          >
            {currentComment ? 'コメントを編集' : 'コメントを追加'}
          </button>
          {currentComment && (
            <button
              className="context-menu__item context-menu__item--danger"
              onClick={deleteComment}
            >
              コメントを削除
            </button>
          )}

          {/* SubgraphNode: 名前変更 */}
          {isSubgraph && !editingName && (
            <button
              className="context-menu__item"
              onClick={() => { setNameText(currentName); setEditingName(true) }}
            >
              名前を変更
            </button>
          )}
          {isSubgraph && editingName && (
            <div className="context-menu__comment-editor">
              <input
                className="context-menu__textarea"
                value={nameText}
                onChange={(e) => setNameText(e.target.value)}
                autoFocus
                placeholder="グループ名..."
                onKeyDown={(e) => { if (e.key === 'Enter') saveName() }}
              />
              <div className="context-menu__comment-actions">
                <button className="context-menu__btn" onClick={saveName}>保存</button>
                <button className="context-menu__btn context-menu__btn--cancel" onClick={() => setEditingName(false)}>キャンセル</button>
              </div>
            </div>
          )}

          {/* SubgraphNode: グループ解除 */}
          {isSubgraph && onUngroup && (
            <button
              className="context-menu__item context-menu__item--danger"
              onClick={() => { onBeforeAction?.(); onUngroup(menu.nodeId); onClose() }}
            >
              グループ解除
            </button>
          )}

          {/* 複数選択時: グループ化 */}
          {!isSubgraph && groupCandidates.length > 1 && onGroup && (
            <button
              className="context-menu__item"
              onClick={() => { onBeforeAction?.(); onGroup(groupCandidates); onClose() }}
            >
              グループ化 ({groupCandidates.length}ノード)
            </button>
          )}
        </>
      )}
    </div>
  )
}
