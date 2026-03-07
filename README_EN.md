[[Japanese](README.md)/[English](README_EN.md)]

# onnx-cv-graph-node-editor

A visual node editor for image processing using ONNX models from [onnx-cv-graph](https://github.com/Kazuhito00/onnx-cv-graph).<br>
Connect nodes in the browser and run real-time image processing inference with ONNX Runtime Web.

<img width="1628" height="968" alt="image" src="https://github.com/user-attachments/assets/4e0872bb-6743-44b7-a69d-9b9b30758185" />

# Web Demo

Try the demo here:
* https://kazuhito00.github.io/onnx-cv-graph-node-editor/

# Features

- Drag & drop nodes to build image processing pipelines
- Real-time in-browser inference with ONNX Runtime Web (WASM)
- Preview processing results on each node
- Group multiple nodes into subgraphs
- Experimental support for exporting constructed chains and subgraphs as a single ONNX file.
- Import ONNX files exported by onnx-cv-graph-node-editor (baked parameters only)
- Save / load graphs as JSON
- Undo/Redo and auto-layout support

# Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| Ctrl+S | Save graph as JSON |
| Ctrl+L | Load graph from JSON |
| Ctrl+E | ONNX export |
| Ctrl+A | Auto layout |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Delete / Backspace | Delete selected nodes/edges |

# Requirements

```
Node.js 20 or later
```

# Installation

```bash
# Clone the repository
git clone https://github.com/Kazuhito00/onnx-cv-graph-node-editor
cd onnx-cv-graph-node-editor

# Install packages
npm install
```

# Usage

### Development server
```bash
npm run dev
```

### Production build
```bash
npm run build
```

### Preview build output
```bash
npm run preview
```

### Lint
```bash
npm run lint
```

### Test
```bash
npm run test
```

# Project Structure

```text
README.md                      # README (Japanese)
README_EN.md                   # README (English)
index.html                     # Entry HTML (loads coi-serviceworker)
vite.config.ts                 # Vite config (COOP/COEP headers, onnxruntime-web exclusion)
public/
  models/                      # ONNX model files
    models_meta.json           # Category, model list, and parameter definitions
  ort-wasm-simd-threaded.wasm  # ONNX Runtime WASM binary
src/
  App.tsx                      # Main app (ReactFlow, drag & drop, keybindings)
  types.ts                     # Type definitions (node data types, metadata types, drag constants)
  nodes/                       # ReactFlow custom nodes
    InputImageNode.tsx         # Image input node
    ProcessingNode.tsx         # ONNX processing node
    OutputImageNode.tsx        # Image output node
    SubgraphNode.tsx           # Subgraph (grouping) node
  hooks/                       # Custom hooks
    useInferenceWorker.ts      # Worker communication and inference request management
    tensorStore.ts             # Tensor data management (outside React state)
    useModelsMeta.ts           # Fetching models_meta.json
    useSubgraph.ts             # Node grouping and expansion
    useUndoRedo.ts             # Undo/Redo
  workers/
    inferenceWorker.ts         # Web Worker (ONNX inference implementation)
  components/                  # UI components
    Sidebar.tsx                # Sidebar (model list and action buttons)
    ContextMenu.tsx            # Right-click context menu
    OnnxExportModal.tsx        # ONNX export dialog
    TimingOverlay.tsx          # Inference timing overlay
  utils/
    onnxMerge.ts               # Serial merging of multiple ONNX models
    imageConvert.ts            # Image conversion utilities
```

# Author
Kazuhito Takahashi(https://x.com/KzhtTkhs)

# License
onnx-cv-graph-node-editor is under [Apache-2.0 license](LICENSE).
