// JS側で不可避な最小限の変換のみ実装
// ピクセル算術はすべて ONNX モデルに委譲する

/**
 * HTMLImageElement → Float32Array [H, W, 3] (RGB, 値域 0–255)
 * alpha チャンネルを除去するだけ。スケーリング等は ONNX モデルに委譲。
 */
export function imageElementToFloat32(img: HTMLImageElement): {
  data: Float32Array
  width: number
  height: number
} {
  const canvas = document.createElement('canvas')
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(img, 0, 0)
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  return imageDataToFloat32(imageData)
}

/**
 * ImageData (HWC RGBA uint8) → Float32Array [H, W, 3] (RGB, 値域 0–255)
 * alpha 除去と型変換のみ。
 */
export function imageDataToFloat32(imageData: ImageData): {
  data: Float32Array
  width: number
  height: number
} {
  const { width, height, data } = imageData
  const out = new Float32Array(height * width * 3)
  for (let i = 0; i < height * width; i++) {
    out[i * 3 + 0] = data[i * 4 + 0]  // R
    out[i * 3 + 1] = data[i * 4 + 1]  // G
    out[i * 3 + 2] = data[i * 4 + 2]  // B
  }
  return { data: out, width, height }
}

/**
 * Float32Array [H, W, 3] (RGB, 任意値域) → ImageData (HWC RGBA uint8)
 * alpha 付与と型変換のみ。値のクランプは呼び出し側で行うこと。
 */
export function float32HWCToImageData(
  data: Float32Array,
  width: number,
  height: number,
): ImageData {
  const out = new Uint8ClampedArray(height * width * 4)
  for (let i = 0; i < height * width; i++) {
    out[i * 4 + 0] = data[i * 3 + 0]  // R
    out[i * 4 + 1] = data[i * 3 + 1]  // G
    out[i * 4 + 2] = data[i * 3 + 2]  // B
    out[i * 4 + 3] = 255               // A
  }
  return new ImageData(out, width, height)
}

/**
 * Float32Array (任意) → min/max 正規化して 0–255 にスケール
 * 値域外テンソルのプレビュー表示フォールバック用
 */
export function normalizeToUint8Range(data: Float32Array): Float32Array {
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

/**
 * Float32Array [H, W, 3] (RGB, 0–255) を canvas に描画して ObjectURL を返す
 * toBlob + createObjectURL を使い、PNG圧縮・base64変換のコストを回避する
 */
export async function float32HWCToObjectUrl(
  data: Float32Array,
  width: number,
  height: number,
): Promise<string> {
  const imageData = float32HWCToImageData(data, width, height)
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  canvas.getContext('2d')!.putImageData(imageData, 0, 0)
  return new Promise<string>((resolve) => {
    canvas.toBlob(
      (blob) => resolve(URL.createObjectURL(blob!)),
      'image/jpeg',
      0.85,
    )
  })
}
