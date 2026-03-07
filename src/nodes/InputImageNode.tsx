import { useCallback, useEffect, useRef, useState } from 'react'
import { Handle, Position, useReactFlow } from '@xyflow/react'
import { imageDataToFloat32 } from '../utils/imageConvert'
import { useInferenceWorker } from '../hooks/useInferenceWorker'
import { setTensor } from '../hooks/tensorStore'
import { isPaused, subscribePause } from '../hooks/usePause'
import type { InputImageNodeData, TensorData } from '../types'

const BASE = './'

type InputMode = 'image' | 'camera' | 'video'

interface Props {
  id: string
  data: InputImageNodeData
}

export default function InputImageNode({ id, data }: Props) {
  const imageInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const { setNodes } = useReactFlow()
  const { propagateFrom } = useInferenceWorker(BASE)

  const [mode, setMode] = useState<InputMode>('image')
  const modeRef = useRef<InputMode>(mode)
  useEffect(() => { modeRef.current = mode }, [mode])

  // フレームループ用 ref
  const tensorRef = useRef<TensorData | null>(null)
  const propagateFromRef = useRef(propagateFrom)
  useEffect(() => { propagateFromRef.current = propagateFrom }, [propagateFrom])

  // 一時停止で video を pause/play
  useEffect(() => {
    return subscribePause((p) => {
      const video = videoRef.current
      if (!video || modeRef.current === 'image') return
      if (p) {
        video.pause()
      } else {
        video.play()
      }
    })
  }, [])

  // video 要素からフレームを取得して tensorRef を更新
  const captureFrame = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.paused || video.ended || video.readyState < 2) return

    const w = video.videoWidth
    const h = video.videoHeight
    if (w === 0 || h === 0) return

    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(video, 0, 0, w, h)
    const imageData = ctx.getImageData(0, 0, w, h)
    const { data: rawData, width, height } = imageDataToFloat32(imageData)

    const tensor: TensorData = { data: rawData, dims: [height, width, 3] }
    tensorRef.current = tensor
    setTensor(id, tensor)
  }, [id])

  // 50ms インターバルループ
  useEffect(() => {
    const timerId = setInterval(() => {
      if (modeRef.current !== 'image') {
        captureFrame()
      }
      if (tensorRef.current && !isPaused()) {
        propagateFromRef.current(id, tensorRef.current)
      }
    }, 50)
    return () => clearInterval(timerId)
  }, [id, captureFrame])

  // カメラ停止用
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
      videoRef.current.removeAttribute('src')
    }
  }, [])

  // アンマウント時のクリーンアップ
  useEffect(() => {
    return () => { stopCamera() }
  }, [stopCamera])

  // 画像ファイル選択
  const handleImageFile = useCallback(
    async (file: File) => {
      stopCamera()
      setMode('image')

      const previewUrl = URL.createObjectURL(file)
      const img = new Image()
      img.src = previewUrl
      await new Promise<void>((res) => { img.onload = () => res() })

      const canvas = document.createElement('canvas')
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0)
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
      const { data: rawData, width, height } = imageDataToFloat32(imageData)

      const tensor: TensorData = { data: rawData, dims: [height, width, 3] }
      tensorRef.current = tensor
      setTensor(id, tensor)

      setNodes((nodes) =>
        nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, previewUrl } } : n,
        ),
      )
    },
    [id, setNodes, stopCamera],
  )

  // カメラ開始
  const startCamera = useCallback(async () => {
    stopCamera()
    tensorRef.current = null
    setMode('camera')

    setNodes((nodes) =>
      nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, previewUrl: undefined } } : n,
      ),
    )

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }
    } catch (e) {
      console.error('カメラ取得エラー:', e)
      setMode('image')
    }
  }, [id, setNodes, stopCamera])

  // 動画ファイル選択
  const handleVideoFile = useCallback(
    (file: File) => {
      stopCamera()
      tensorRef.current = null
      setMode('video')

      const url = URL.createObjectURL(file)

      setNodes((nodes) =>
        nodes.map((n) =>
          n.id === id ? { ...n, data: { ...n.data, previewUrl: undefined } } : n,
        ),
      )

      const video = videoRef.current
      if (video) {
        video.srcObject = null
        video.src = url
        video.loop = true
        video.play()
      }
    },
    [id, setNodes, stopCamera],
  )

  const showVideo = mode === 'camera' || mode === 'video'

  return (
    <div className="node node--input">
      {data.comment && (
        <div className="node__comment">{data.comment}</div>
      )}
      <div className="node__header">Input</div>
      <div className="node__body">
        <div className="node__preview-area">
          {/* video は常にレンダリングし、モードに応じて表示/非表示 */}
          <video
            ref={videoRef}
            className="node__preview-area-img"
            style={{ display: showVideo ? 'block' : 'none' }}
            muted
            playsInline
          />
          {!showVideo && data.previewUrl && (
            <img className="node__preview-area-img" src={data.previewUrl} alt="preview" />
          )}
        </div>
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        <div
          className="nodrag"
          onMouseDown={(e) => e.stopPropagation()}
          onPointerDown={(e) => e.stopPropagation()}
        >
          <div className="node__input-buttons">
            <button
              className="node__upload-btn"
              onClick={() => imageInputRef.current?.click()}
            >
              画像 / Image
            </button>
            <button
              className={`node__upload-btn${mode === 'camera' ? ' node__upload-btn--active' : ''}`}
              onClick={() => mode === 'camera' ? (stopCamera(), setMode('image')) : startCamera()}
            >
              {mode === 'camera' ? '停止 / Stop' : 'カメラ / Camera'}
            </button>
            <button
              className="node__upload-btn"
              onClick={() => videoInputRef.current?.click()}
            >
              動画 / Video
            </button>
          </div>
        </div>
        <input
          ref={imageInputRef}
          type="file"
          accept="image/png,image/jpeg"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) handleImageFile(file)
            e.target.value = ''
          }}
        />
        <input
          ref={videoInputRef}
          type="file"
          accept="video/*"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) handleVideoFile(file)
            e.target.value = ''
          }}
        />
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
