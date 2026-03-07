[[Japanese](README.md)/[English](README_EN.md)]

# onnx-cv-graph-node-editor

[onnx-cv-graph](https://github.com/Kazuhito00/onnx-cv-graph) のONNXモデルを用いた画像処理ノードエディタです。<br>
ブラウザ上でノードを接続し、ONNX Runtime Webでリアルタイムに画像処理を実行できます。

<img width="1628" height="968" alt="image" src="https://github.com/user-attachments/assets/4e0872bb-6743-44b7-a69d-9b9b30758185" />

# Web Demo

以下のページからデモを確認できます。
* https://kazuhito00.github.io/onnx-cv-graph-node-editor/

# Features

以下の特徴があります。
- ノードをドラッグ＆ドロップで配置し画像処理パイプラインを構築
- ONNX Runtime Web (WASM) によるブラウザ内リアルタイム推論
- 各ノードの処理結果をプレビュー表示
- 複数ノードをサブグラフとしてグループ化
- 構築したチェーンやサブグラフを単一のONNXファイルとしてエクスポートする機能を試験的にサポート
- onnx-cv-graph-node-editor でエクスポートしたONNXファイルの読み込み（パラメータ埋め込みONNXのみ）
- グラフの保存（JSON）・読み込みに対応
- Undo/Redo、オートレイアウト対応

# Keyboard Shortcuts

| ショートカット | 機能 |
|---|---|
| Ctrl+S | グラフをJSONファイルとして保存 |
| Ctrl+L | JSONファイルからグラフを読み込み |
| Ctrl+E | ONNXエクスポート |
| Ctrl+A | オートレイアウト |
| Ctrl+Z | 元に戻す (Undo) |
| Ctrl+Y | やり直し (Redo) |
| Delete / Backspace | 選択ノード・エッジを削除 |

# Requirements

```
Node.js 20 or later
```

# Installation

```bash
# リポジトリクローン
git clone https://github.com/Kazuhito00/onnx-cv-graph-node-editor
cd onnx-cv-graph-node-editor

# パッケージインストール
npm install
```

# Usage

### 開発サーバー起動
```bash
npm run dev
```

### プロダクションビルド
```bash
npm run build
```

### ビルド結果のプレビュー
```bash
npm run preview
```

### Lint
```bash
npm run lint
```

### テスト
```bash
npm run test
```

# Project Structure

```text
README.md                      # README（日本語）
index.html                     # エントリHTML（coi-serviceworker読み込み）
vite.config.ts                 # Vite設定（COOP/COEPヘッダー、onnxruntime-web除外）
public/
  models/                      # ONNXモデルファイル格納先
    models_meta.json           # カテゴリ・モデル一覧・パラメータ定義
  ort-wasm-simd-threaded.wasm  # ONNX Runtime WASMバイナリ
src/
  App.tsx                      # メインアプリ（ReactFlow、ドラッグ＆ドロップ、キーバインド）
  types.ts                     # 型定義（ノードdata型、メタデータ型、ドラッグ定数）
  nodes/                       # ReactFlowカスタムノード
    InputImageNode.tsx         # 画像入力ノード
    ProcessingNode.tsx         # ONNX処理ノード
    OutputImageNode.tsx        # 画像出力ノード
    SubgraphNode.tsx           # サブグラフ（グループ化）ノード
  hooks/                       # カスタムフック
    useInferenceWorker.ts      # Worker通信・推論リクエスト管理
    tensorStore.ts             # テンソルデータ管理（React state外）
    useModelsMeta.ts           # models_meta.jsonの取得
    useSubgraph.ts             # ノードのグループ化・展開
    useUndoRedo.ts             # Undo/Redo
  workers/
    inferenceWorker.ts         # Web Worker（ONNX推論の実体）
  components/                  # UIコンポーネント
    Sidebar.tsx                # サイドバー（モデル一覧・操作ボタン）
    ContextMenu.tsx            # 右クリックメニュー
    OnnxExportModal.tsx        # ONNXエクスポートダイアログ
    TimingOverlay.tsx          # 推論時間表示
  utils/
    onnxMerge.ts               # 複数ONNXモデルの直列結合
    imageConvert.ts            # 画像変換ユーティリティ
```

# Author
高橋かずひと(https://x.com/KzhtTkhs)

# License
onnx-cv-graph-node-editor is under [Apache-2.0 license](LICENSE).
