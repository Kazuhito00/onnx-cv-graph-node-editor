/**
 * ONNX モデルチェーン結合
 *
 * 複数の ONNX モデルを直列に結合し、単一の ONNX ファイルとして出力する。
 * - 各モデルの内部テンソル名をプレフィックスでリネームして衝突回避
 * - スカラーパラメータ（スライダー値）はイニシャライザーとしてベイク or 外部入力として残す
 * - 中間の input/output を接続
 */

import { onnx } from 'onnx-proto'

export interface ChainModel {
  url: string
  params: Record<string, number>
}

export interface MergeOptions {
  /** true: パラメータを固定値として埋め込む / false: 外部入力として残す */
  bakeParams?: boolean
}

/** デコード済みモデル + パラメータ */
export interface DecodedModel {
  model: onnx.ModelProto
  params: Record<string, number>
}

/**
 * デコード済みモデル配列からマージ済み ONNX バイナリを生成する（テスト可能な純粋関数）
 */
export function mergeDecodedModels(
  decoded: DecodedModel[],
  graphName: string,
  options?: MergeOptions,
): Uint8Array {
  const bakeParams = options?.bakeParams ?? true
  if (decoded.length === 0) throw new Error('No models to merge')

  const mergedNodes: onnx.INodeProto[] = []
  const mergedInitializers: onnx.ITensorProto[] = []
  const mergedInputs: onnx.IValueInfoProto[] = []
  let mergedOutput: onnx.IValueInfoProto | null = null

  let prevOutputName = 'input'

  for (let i = 0; i < decoded.length; i++) {
    const { model, params } = decoded[i]
    const graph = model.graph!
    const prefix = `m${i}_`
    const isLast = i === decoded.length - 1

    // このモデルの output テンソル名
    const newOutputName = isLast ? 'output' : `${prefix}out`

    // リネームマップ構築
    const rename = new Map<string, string>()
    rename.set('input', prevOutputName)
    rename.set('output', newOutputName)

    // スカラーパラメータ入力の処理
    const paramInputs: onnx.IValueInfoProto[] = []
    for (const inp of graph.input ?? []) {
      const name = inp.name!
      if (name === 'input') continue
      const newName = prefix + name
      rename.set(name, newName)
      if (bakeParams) {
        mergedInitializers.push(
          onnx.TensorProto.create({
            name: newName,
            dims: [1],
            dataType: onnx.TensorProto.DataType.FLOAT,
            floatData: [params[name] ?? 0],
          }),
        )
      } else {
        paramInputs.push(onnx.ValueInfoProto.create({ ...inp, name: newName }))
      }
    }

    // 既存イニシャライザー名をリネーム
    for (const init of graph.initializer ?? []) {
      if (!rename.has(init.name!)) {
        rename.set(init.name!, prefix + init.name!)
      }
    }

    // ノード内部テンソル名をリネーム
    for (const node of graph.node ?? []) {
      for (const t of [...(node.input ?? []), ...(node.output ?? [])]) {
        if (t && !rename.has(t)) {
          rename.set(t, prefix + t)
        }
      }
    }

    const r = (name: string) => rename.get(name) ?? name

    // ノード追加
    for (const node of graph.node ?? []) {
      mergedNodes.push(
        onnx.NodeProto.create({
          opType: node.opType,
          domain: node.domain,
          name: prefix + (node.name || `node${mergedNodes.length}`),
          input: node.input?.map(r),
          output: node.output?.map(r),
          attribute: node.attribute,
        }),
      )
    }

    // イニシャライザー追加
    const graphInputNames = new Set((graph.input ?? []).map((inp) => inp.name!))
    for (const init of graph.initializer ?? []) {
      if (graphInputNames.has(init.name!) && init.name !== 'input') continue
      mergedInitializers.push(
        onnx.TensorProto.create({
          ...init,
          name: r(init.name!),
        }),
      )
    }

    // チェーン全体の入力/出力
    if (i === 0) {
      const inp = graph.input?.find((x) => x.name === 'input')
      if (inp) mergedInputs.push(onnx.ValueInfoProto.create({ ...inp, name: 'input' }))
    }
    mergedInputs.push(...paramInputs)
    if (isLast) {
      const out = graph.output?.find((x) => x.name === 'output')
      if (out) mergedOutput = onnx.ValueInfoProto.create({ ...out, name: 'output' })
    }

    prevOutputName = newOutputName
  }

  const firstModel = decoded[0].model
  const merged = onnx.ModelProto.create({
    irVersion: firstModel.irVersion,
    opsetImport: firstModel.opsetImport,
    producerName: 'onnx-cv-graph-node-editor',
    graph: onnx.GraphProto.create({
      name: graphName,
      node: mergedNodes,
      initializer: mergedInitializers,
      input: mergedInputs,
      output: mergedOutput ? [mergedOutput] : [],
    }),
  })

  return onnx.ModelProto.encode(merged).finish()
}

/**
 * URL からモデルをフェッチして結合する（ブラウザ用エントリポイント）
 */
export async function mergeOnnxChain(
  models: ChainModel[],
  graphName: string,
  options?: MergeOptions,
): Promise<Uint8Array> {
  if (models.length === 0) throw new Error('No models to merge')

  const decoded: DecodedModel[] = await Promise.all(
    models.map(async (m) => {
      const resp = await fetch(m.url)
      const buf = await resp.arrayBuffer()
      return {
        model: onnx.ModelProto.decode(new Uint8Array(buf)),
        params: m.params,
      }
    }),
  )

  return mergeDecodedModels(decoded, graphName, options)
}
