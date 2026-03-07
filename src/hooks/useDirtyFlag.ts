// モジュールレベルの dirty フラグ
// 画像アップロード、パラメータ変更、エッジ変更時に markDirty() を呼ぶ
// InputImageNode のインターバルループが dirty を消費して推論を実行する

let dirty = false

export function markDirty(): void {
  dirty = true
}

export function consumeDirty(): boolean {
  if (!dirty) return false
  dirty = false
  return true
}
