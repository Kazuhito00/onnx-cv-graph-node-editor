import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'
import { onnx } from 'onnx-proto'
import { mergeDecodedModels, type DecodedModel } from './onnxMerge'

const MODELS_DIR = join(__dirname, '../../public/models')

function loadModel(categoryId: string, opName: string): onnx.ModelProto {
  const buf = readFileSync(join(MODELS_DIR, categoryId, `${opName}.onnx`))
  return onnx.ModelProto.decode(buf)
}

function decodeMerged(data: Uint8Array): onnx.ModelProto {
  return onnx.ModelProto.decode(data)
}

// ---- ヘルパー ----

function getNodeInputs(model: onnx.ModelProto): string[][] {
  return model.graph!.node!.map((n) => [...(n.input ?? [])])
}

function getNodeOutputs(model: onnx.ModelProto): string[][] {
  return model.graph!.node!.map((n) => [...(n.output ?? [])])
}

function getInputNames(model: onnx.ModelProto): string[] {
  return model.graph!.input!.map((i) => i.name!)
}

function getOutputNames(model: onnx.ModelProto): string[] {
  return model.graph!.output!.map((o) => o.name!)
}

function getInitializerNames(model: onnx.ModelProto): string[] {
  return model.graph!.initializer!.map((i) => i.name!)
}

/** グラフ内の全テンソル名を収集 */
function collectAllTensorNames(model: onnx.ModelProto): Set<string> {
  const names = new Set<string>()
  for (const node of model.graph!.node!) {
    for (const t of [...(node.input ?? []), ...(node.output ?? [])]) {
      if (t) names.add(t)
    }
  }
  return names
}

/** ノードの attribute を名前→値のマップで取得 */
function getNodeAttributes(node: onnx.INodeProto): Map<string, unknown> {
  const attrs = new Map<string, unknown>()
  for (const a of node.attribute ?? []) {
    if (a.i != null) attrs.set(a.name!, a.i)
    else if (a.f != null) attrs.set(a.name!, a.f)
    else if (a.s != null) attrs.set(a.name!, a.s)
    else if (a.ints && a.ints.length > 0) attrs.set(a.name!, a.ints)
    else if (a.t != null) attrs.set(a.name!, a.t)
  }
  return attrs
}

/**
 * グラフの接続性を検証:
 * 各ノードの入力が、graph.input / initializer / 先行ノードの出力 のいずれかに存在するか
 */
function validateConnectivity(model: onnx.ModelProto): string[] {
  const errors: string[] = []
  const available = new Set<string>()

  // graph inputs + initializers
  for (const inp of model.graph!.input ?? []) available.add(inp.name!)
  for (const init of model.graph!.initializer ?? []) available.add(init.name!)

  for (const node of model.graph!.node!) {
    for (const inp of node.input ?? []) {
      if (inp && !available.has(inp)) {
        errors.push(`Node "${node.name}" input "${inp}" is not produced by any prior node or input`)
      }
    }
    for (const out of node.output ?? []) {
      if (out) available.add(out)
    }
  }

  // graph output must be produced
  for (const out of model.graph!.output ?? []) {
    if (!available.has(out.name!)) {
      errors.push(`Graph output "${out.name}" is not produced`)
    }
  }

  return errors
}

// ================================================================
// テストケース
// ================================================================

describe('mergeDecodedModels', () => {
  // ---- 1. 基本: 単一モデル ----

  describe('単一モデル（パラメータなし）', () => {
    it('grayscale: ノード数・入力・出力が保持される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'grayscale'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_single'))

      expect(merged.graph!.name).toBe('test_single')
      expect(getInputNames(merged)).toEqual(['input'])
      expect(getOutputNames(merged)).toEqual(['output'])
      expect(merged.graph!.node!.length).toBe(3)
      expect(getNodeInputs(merged)[0]).toContain('input')
      const lastOutputs = getNodeOutputs(merged)[merged.graph!.node!.length - 1]
      expect(lastOutputs).toContain('output')
    })
  })

  describe('単一モデル（パラメータあり、bakeParams=true）', () => {
    it('brightness: パラメータがイニシャライザーとしてベイクされる', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: 0.3 } },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_bake'))

      expect(getInputNames(merged)).toEqual(['input'])
      const initNames = getInitializerNames(merged)
      expect(initNames).toContain('m0_brightness')
      const bakedInit = merged.graph!.initializer!.find((i) => i.name === 'm0_brightness')!
      expect(bakedInit.floatData![0]).toBeCloseTo(0.3)
    })
  })

  describe('単一モデル（パラメータあり、bakeParams=false）', () => {
    it('brightness: パラメータが外部入力として残る', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: 0.3 } },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_ext', { bakeParams: false }))

      const inputNames = getInputNames(merged)
      expect(inputNames).toContain('input')
      expect(inputNames).toContain('m0_brightness')
      const initNames = getInitializerNames(merged)
      expect(initNames).not.toContain('m0_brightness')
    })
  })

  // ---- 2. 2モデル結合 ----

  describe('2モデル結合（パラメータなし同士）', () => {
    it('grayscale → invert: ノード数が合計、中間テンソルが接続される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'grayscale'), params: {} },
        { model: loadModel('01_elementwise', 'invert'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_chain2'))

      expect(getInputNames(merged)).toEqual(['input'])
      expect(getOutputNames(merged)).toEqual(['output'])

      const srcGray = loadModel('01_elementwise', 'grayscale')
      const srcInv = loadModel('01_elementwise', 'invert')
      const expectedNodes = srcGray.graph!.node!.length + srcInv.graph!.node!.length
      expect(merged.graph!.node!.length).toBe(expectedNodes)

      expect(getNodeInputs(merged)[0]).toContain('input')
      const lastOutputs = getNodeOutputs(merged)[merged.graph!.node!.length - 1]
      expect(lastOutputs).toContain('output')

      // 中間接続: m0 の最後の出力 = m1 の最初の入力
      const m0LastOut = getNodeOutputs(merged)[srcGray.graph!.node!.length - 1]
      const m1FirstIn = getNodeInputs(merged)[srcGray.graph!.node!.length]
      expect(m0LastOut).toContain('m0_out')
      expect(m1FirstIn).toContain('m0_out')
    })
  })

  describe('2モデル結合（パラメータあり、bakeParams=true）', () => {
    it('grayscale → brightness: パラメータがベイクされ、入力は input のみ', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'grayscale'), params: {} },
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: 0.2 } },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_bake2'))

      expect(getInputNames(merged)).toEqual(['input'])
      expect(getInitializerNames(merged)).toContain('m1_brightness')
      const bakedInit = merged.graph!.initializer!.find((i) => i.name === 'm1_brightness')!
      expect(bakedInit.floatData![0]).toBeCloseTo(0.2)
    })
  })

  describe('2モデル結合（パラメータあり、bakeParams=false）', () => {
    it('grayscale → brightness: パラメータが外部入力として残る', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'grayscale'), params: {} },
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: 0.2 } },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_ext2', { bakeParams: false }))

      const inputNames = getInputNames(merged)
      expect(inputNames).toContain('input')
      expect(inputNames).toContain('m1_brightness')
    })
  })

  // ---- 3. 3モデル以上の結合 ----

  describe('3モデル結合', () => {
    it('grayscale → brightness → contrast: 全ノードが直列接続される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'grayscale'), params: {} },
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: 0.1 } },
        { model: loadModel('01_elementwise', 'contrast'), params: { contrast: 1.5, center: 0.5 } },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_chain3'))

      expect(getInputNames(merged)).toEqual(['input'])
      expect(getOutputNames(merged)).toEqual(['output'])

      const srcGray = loadModel('01_elementwise', 'grayscale')
      const srcBright = loadModel('01_elementwise', 'brightness')
      const srcContrast = loadModel('01_elementwise', 'contrast')
      const expectedNodes =
        srcGray.graph!.node!.length +
        srcBright.graph!.node!.length +
        srcContrast.graph!.node!.length
      expect(merged.graph!.node!.length).toBe(expectedNodes)

      expect(getNodeInputs(merged)[0]).toContain('input')
      const lastOutputs = getNodeOutputs(merged)[merged.graph!.node!.length - 1]
      expect(lastOutputs).toContain('output')
    })
  })

  // ---- 4. テンソル名衝突回避 ----

  describe('同一モデルの連続結合', () => {
    it('invert → invert: 内部テンソル名が衝突しない', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'invert'), params: {} },
        { model: loadModel('01_elementwise', 'invert'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_dup'))

      const allTensors = collectAllTensorNames(merged)
      const m0Tensors = [...allTensors].filter((t) => t.startsWith('m0_'))
      const m1Tensors = [...allTensors].filter((t) => t.startsWith('m1_'))
      expect(m0Tensors.length).toBeGreaterThan(0)
      expect(m1Tensors.length).toBeGreaterThan(0)
      const shared = m0Tensors.filter((t) => m1Tensors.includes(t))
      expect(shared).toEqual([])
    })

    it('sharpen × 3: 属性付きノードが3回結合されてもテンソル名が衝突しない', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('03_conv_filter', 'sharpen'), params: {} },
        { model: loadModel('03_conv_filter', 'sharpen'), params: {} },
        { model: loadModel('03_conv_filter', 'sharpen'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_sharpen3'))

      const allTensors = collectAllTensorNames(merged)
      // 各プレフィックスの内部テンソルがそれぞれ分離されている
      for (let i = 0; i < 3; i++) {
        const prefix = `m${i}_`
        const tensors = [...allTensors].filter((t) => t.startsWith(prefix))
        expect(tensors.length).toBeGreaterThan(0)
      }
      // 接続性も正しい
      expect(validateConnectivity(merged)).toEqual([])
    })
  })

  // ---- 5. 前処理チェーン ----

  describe('前処理チェーン', () => {
    it('batch_unsqueeze_nhwc → hwc_to_chw → scale_from_255: ML前処理が正しく結合される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('10_ml_preprocess', 'batch_unsqueeze_nhwc'), params: {} },
        { model: loadModel('10_ml_preprocess', 'hwc_to_chw'), params: {} },
        { model: loadModel('10_ml_preprocess', 'scale_from_255'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'preprocess'))

      expect(getInputNames(merged)).toEqual(['input'])
      expect(getOutputNames(merged)).toEqual(['output'])
      expect(merged.graph!.node!.length).toBeGreaterThan(0)
      expect(validateConnectivity(merged)).toEqual([])
    })
  })

  // ---- 6. 複数パラメータモデル ----

  describe('複数パラメータモデル', () => {
    it('contrast (contrast + center): 両パラメータがベイクされる', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'contrast'), params: { contrast: 2.0, center: 0.4 } },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_multi_param'))

      const initNames = getInitializerNames(merged)
      expect(initNames).toContain('m0_contrast')
      expect(initNames).toContain('m0_center')

      const contrastInit = merged.graph!.initializer!.find((i) => i.name === 'm0_contrast')!
      const centerInit = merged.graph!.initializer!.find((i) => i.name === 'm0_center')!
      expect(contrastInit.floatData![0]).toBeCloseTo(2.0)
      expect(centerInit.floatData![0]).toBeCloseTo(0.4)
    })

    it('contrast (bakeParams=false): 両パラメータが外部入力として残る', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'contrast'), params: { contrast: 2.0, center: 0.4 } },
      ]
      const merged = decodeMerged(
        mergeDecodedModels(decoded, 'test_multi_ext', { bakeParams: false }),
      )

      const inputNames = getInputNames(merged)
      expect(inputNames).toContain('input')
      expect(inputNames).toContain('m0_contrast')
      expect(inputNames).toContain('m0_center')
    })
  })

  // ---- 7. エラーケース ----

  describe('エラーケース', () => {
    it('空配列でエラー', () => {
      expect(() => mergeDecodedModels([], 'empty')).toThrow('No models to merge')
    })
  })

  // ---- 8. ラウンドトリップ ----

  describe('ラウンドトリップ', () => {
    it('エンコード → デコードで構造が保持される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'grayscale'), params: {} },
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: 0.5 } },
      ]
      const bytes = mergeDecodedModels(decoded, 'roundtrip')
      const re = onnx.ModelProto.decode(bytes)

      expect(re.graph!.name).toBe('roundtrip')
      expect(re.producerName).toBe('onnx-cv-graph-node-editor')
      expect(getInputNames(re)).toEqual(['input'])
      expect(getOutputNames(re)).toEqual(['output'])
      const bytes2 = onnx.ModelProto.encode(re).finish()
      expect(bytes2).toEqual(bytes)
    })
  })

  // ================================================================
  // 複雑なケース
  // ================================================================

  // ---- 9. 多パラメータモデル (levels: 5 params) ----

  describe('多パラメータモデル (levels)', () => {
    it('bake: 5パラメータ全てがイニシャライザーとしてベイクされる', () => {
      const params = { in_black: 0.1, in_white: 0.9, gamma: 1.2, out_black: 0.0, out_white: 1.0 }
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'levels'), params },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_levels'))

      expect(getInputNames(merged)).toEqual(['input'])
      const initNames = getInitializerNames(merged)
      for (const [key, val] of Object.entries(params)) {
        const initName = `m0_${key}`
        expect(initNames).toContain(initName)
        const init = merged.graph!.initializer!.find((i) => i.name === initName)!
        expect(init.floatData![0]).toBeCloseTo(val)
      }
      // levels は 11 ノード
      expect(merged.graph!.node!.length).toBe(11)
      expect(validateConnectivity(merged)).toEqual([])
    })

    it('external: 5パラメータ全てが外部入力として残る', () => {
      const params = { in_black: 0.1, in_white: 0.9, gamma: 1.2, out_black: 0.0, out_white: 1.0 }
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'levels'), params },
      ]
      const merged = decodeMerged(
        mergeDecodedModels(decoded, 'test_levels_ext', { bakeParams: false }),
      )

      const inputNames = getInputNames(merged)
      expect(inputNames[0]).toBe('input')
      for (const key of Object.keys(params)) {
        expect(inputNames).toContain(`m0_${key}`)
      }
      expect(inputNames.length).toBe(6) // input + 5 params
    })
  })

  // ---- 10. 属性付きモデル (Conv, MaxPool 等) ----

  describe('属性付きモデル', () => {
    it('sharpen (Conv + Clip + Pad): attribute が保持される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('03_conv_filter', 'sharpen'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_attrs'))

      const convNode = merged.graph!.node!.find((n) => n.opType === 'Conv')!
      expect(convNode).toBeDefined()
      const attrs = getNodeAttributes(convNode)
      // Conv の group 属性が保持されている
      expect(attrs.has('group')).toBe(true)
      // Pad の mode 属性も保持されている
      const padNode = merged.graph!.node!.find((n) => n.opType === 'Pad')!
      expect(padNode).toBeDefined()
      expect(getNodeAttributes(padNode).has('mode')).toBe(true)
      expect(validateConnectivity(merged)).toEqual([])
    })

    it('opening_3x3 (MaxPool × 2): MaxPool 属性が保持される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('04_morphology', 'opening_3x3'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_morpho'))

      const maxPoolNodes = merged.graph!.node!.filter((n) => n.opType === 'MaxPool')
      expect(maxPoolNodes.length).toBe(2)
      for (const node of maxPoolNodes) {
        const attrs = getNodeAttributes(node)
        expect(attrs.has('kernel_shape')).toBe(true)
      }
    })
  })

  // ---- 11. 複雑なモデル (crop: 28ノード, 4パラメータ) ----

  describe('複雑なモデル (crop)', () => {
    it('bake: 28ノードと4パラメータが正しく結合される', () => {
      const params = { crop_top: 0.1, crop_left: 0.2, crop_h: 0.5, crop_w: 0.6 }
      const decoded: DecodedModel[] = [
        { model: loadModel('05_geometric', 'crop'), params },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_crop'))

      expect(merged.graph!.node!.length).toBe(28)
      expect(getInputNames(merged)).toEqual(['input'])
      for (const key of Object.keys(params)) {
        expect(getInitializerNames(merged)).toContain(`m0_${key}`)
      }
      expect(validateConnectivity(merged)).toEqual([])
    })
  })

  // ---- 12. 超複雑モデル (affine: 40ノード, 6パラメータ, 14イニシャライザー) ----

  describe('超複雑モデル (affine)', () => {
    it('40ノード + 6パラメータ + 14イニシャライザーが正しく結合される', () => {
      const params = { a: 1.0, b: 0.0, tx: 0.0, c: 0.0, d: 1.0, ty: 0.0 }
      const decoded: DecodedModel[] = [
        { model: loadModel('05_geometric', 'affine'), params },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_affine'))

      expect(merged.graph!.node!.length).toBe(40)
      expect(getInputNames(merged)).toEqual(['input'])
      // 6パラメータがベイクされている
      for (const key of Object.keys(params)) {
        expect(getInitializerNames(merged)).toContain(`m0_${key}`)
      }
      // 14個の元イニシャライザーも含まれる
      const srcAffine = loadModel('05_geometric', 'affine')
      const srcInitCount = srcAffine.graph!.initializer!.length
      // ベイクされたパラメータ(6) + 元イニシャライザー(14) = 20
      expect(merged.graph!.initializer!.length).toBe(6 + srcInitCount)
      expect(validateConnectivity(merged)).toEqual([])
    })

    it('bakeParams=false: 6パラメータが外部入力として残る', () => {
      const params = { a: 1.0, b: 0.0, tx: 0.0, c: 0.0, d: 1.0, ty: 0.0 }
      const decoded: DecodedModel[] = [
        { model: loadModel('05_geometric', 'affine'), params },
      ]
      const merged = decodeMerged(
        mergeDecodedModels(decoded, 'test_affine_ext', { bakeParams: false }),
      )

      const inputNames = getInputNames(merged)
      expect(inputNames.length).toBe(7) // input + 6 params
      expect(inputNames[0]).toBe('input')
    })
  })

  // ---- 13. 多段パラメータ付きチェーン ----

  describe('多段パラメータ付きチェーン', () => {
    it('levels → brightness → contrast: 各モデルのパラメータが独立してベイクされる', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'levels'), params: { in_black: 0.0, in_white: 1.0, gamma: 1.0, out_black: 0.0, out_white: 1.0 } },
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: 0.15 } },
        { model: loadModel('01_elementwise', 'contrast'), params: { contrast: 1.8, center: 0.5 } },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_multi_chain'))

      expect(getInputNames(merged)).toEqual(['input'])
      // m0_ の levels パラメータ
      expect(getInitializerNames(merged)).toContain('m0_in_black')
      expect(getInitializerNames(merged)).toContain('m0_out_white')
      // m1_ の brightness パラメータ
      expect(getInitializerNames(merged)).toContain('m1_brightness')
      // m2_ の contrast パラメータ
      expect(getInitializerNames(merged)).toContain('m2_contrast')
      expect(getInitializerNames(merged)).toContain('m2_center')

      // ベイク値が正しい
      expect(
        merged.graph!.initializer!.find((i) => i.name === 'm1_brightness')!.floatData![0],
      ).toBeCloseTo(0.15)
      expect(
        merged.graph!.initializer!.find((i) => i.name === 'm2_contrast')!.floatData![0],
      ).toBeCloseTo(1.8)

      expect(validateConnectivity(merged)).toEqual([])
    })

    it('bakeParams=false: 全パラメータが外部入力として残り、input の次に並ぶ', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'levels'), params: { in_black: 0.0, in_white: 1.0, gamma: 1.0, out_black: 0.0, out_white: 1.0 } },
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: 0.15 } },
        { model: loadModel('01_elementwise', 'contrast'), params: { contrast: 1.8, center: 0.5 } },
      ]
      const merged = decodeMerged(
        mergeDecodedModels(decoded, 'test_multi_ext', { bakeParams: false }),
      )

      const inputNames = getInputNames(merged)
      // input + 5 (levels) + 1 (brightness) + 2 (contrast) = 9
      expect(inputNames.length).toBe(9)
      expect(inputNames[0]).toBe('input')
      // パラメータ入力はモデル順に並ぶ
      expect(inputNames.slice(1, 6)).toEqual([
        'm0_in_black', 'm0_in_white', 'm0_gamma', 'm0_out_black', 'm0_out_white',
      ])
      expect(inputNames[6]).toBe('m1_brightness')
      expect(inputNames.slice(7)).toEqual(['m2_contrast', 'm2_center'])
    })
  })

  // ---- 14. カテゴリ横断チェーン ----

  describe('カテゴリ横断チェーン', () => {
    it('sepia → sharpen → dilate_3x3: 異なるカテゴリのモデルが正しく直列結合される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('02_color_space', 'sepia'), params: {} },
        { model: loadModel('03_conv_filter', 'sharpen'), params: {} },
        { model: loadModel('04_morphology', 'dilate_3x3'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_cross_cat'))

      expect(getInputNames(merged)).toEqual(['input'])
      expect(getOutputNames(merged)).toEqual(['output'])

      // 各モデルのオペレーションタイプが全て含まれる
      const opTypes = merged.graph!.node!.map((n) => n.opType!)
      // sepia は MatMul や Clip を持つ
      // sharpen は Conv を持つ
      expect(opTypes).toContain('Conv')
      // dilate は MaxPool を持つ
      expect(opTypes).toContain('MaxPool')

      expect(validateConnectivity(merged)).toEqual([])
    })
  })

  // ---- 15. 実際のパイプライン (前処理→処理→後処理) ----

  describe('フルパイプライン (前処理→処理→後処理)', () => {
    it('unsqueeze → hwc_to_chw → scale_from_255 → grayscale → sepia → scale_to_255 → chw_to_hwc → squeeze', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('10_ml_preprocess', 'batch_unsqueeze_nhwc'), params: {} },
        { model: loadModel('10_ml_preprocess', 'hwc_to_chw'), params: {} },
        { model: loadModel('10_ml_preprocess', 'scale_from_255'), params: {} },
        { model: loadModel('01_elementwise', 'grayscale'), params: {} },
        { model: loadModel('02_color_space', 'sepia'), params: {} },
        { model: loadModel('10_ml_preprocess', 'scale_to_255'), params: {} },
        { model: loadModel('10_ml_preprocess', 'chw_to_hwc'), params: {} },
        { model: loadModel('10_ml_preprocess', 'batch_squeeze_nhwc'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'full_pipeline'))

      expect(getInputNames(merged)).toEqual(['input'])
      expect(getOutputNames(merged)).toEqual(['output'])

      // 8モデル分のノードが全て含まれる
      const totalNodes = decoded.reduce((sum, d) => sum + d.model.graph!.node!.length, 0)
      expect(merged.graph!.node!.length).toBe(totalNodes)

      // 中間接続が正しい (m0_out → m1, m1_out → m2, ...)
      for (let i = 0; i < decoded.length - 1; i++) {
        const connTensor = `m${i}_out`
        const allTensors = collectAllTensorNames(merged)
        expect(allTensors.has(connTensor)).toBe(true)
      }

      expect(validateConnectivity(merged)).toEqual([])
    })
  })

  // ---- 16. 同一パラメータ名のモデル連続結合 ----

  describe('同一パラメータ名の衝突回避', () => {
    it('brightness → brightness: 同名パラメータが独立にベイクされる', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: 0.1 } },
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: -0.2 } },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_same_param'))

      const initNames = getInitializerNames(merged)
      expect(initNames).toContain('m0_brightness')
      expect(initNames).toContain('m1_brightness')

      const init0 = merged.graph!.initializer!.find((i) => i.name === 'm0_brightness')!
      const init1 = merged.graph!.initializer!.find((i) => i.name === 'm1_brightness')!
      expect(init0.floatData![0]).toBeCloseTo(0.1)
      expect(init1.floatData![0]).toBeCloseTo(-0.2)

      expect(validateConnectivity(merged)).toEqual([])
    })

    it('bakeParams=false: 同名パラメータが独立した外部入力として残る', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: 0.1 } },
        { model: loadModel('01_elementwise', 'brightness'), params: { brightness: -0.2 } },
      ]
      const merged = decodeMerged(
        mergeDecodedModels(decoded, 'test_same_ext', { bakeParams: false }),
      )

      const inputNames = getInputNames(merged)
      expect(inputNames).toContain('m0_brightness')
      expect(inputNames).toContain('m1_brightness')
      expect(inputNames.length).toBe(3) // input + 2 brightness
    })
  })

  // ---- 17. 長いチェーン (10モデル) ----

  describe('長いチェーン', () => {
    it('blur_3x3 × 10: 10段結合で接続性が保持される', () => {
      const decoded: DecodedModel[] = Array.from({ length: 10 }, () => ({
        model: loadModel('03_conv_filter', 'blur_3x3'),
        params: {},
      }))
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_long'))

      const singleNodeCount = loadModel('03_conv_filter', 'blur_3x3').graph!.node!.length
      expect(merged.graph!.node!.length).toBe(singleNodeCount * 10)

      // 全ノード名がユニーク
      const nodeNames = merged.graph!.node!.map((n) => n.name!)
      expect(new Set(nodeNames).size).toBe(nodeNames.length)

      // 全中間接続テンソルが存在
      const allTensors = collectAllTensorNames(merged)
      for (let i = 0; i < 9; i++) {
        expect(allTensors.has(`m${i}_out`)).toBe(true)
      }

      expect(validateConnectivity(merged)).toEqual([])
    })
  })

  // ---- 18. hsv_range (47ノード, 6パラメータ, 10イニシャライザー) ----

  describe('超大型モデル (hsv_range)', () => {
    it('47ノード + 6パラメータが正しく処理される', () => {
      const params = { h_min: 0.0, h_max: 1.0, s_min: 0.0, s_max: 1.0, v_min: 0.0, v_max: 1.0 }
      const decoded: DecodedModel[] = [
        { model: loadModel('02_color_space', 'hsv_range'), params },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_hsv'))

      expect(merged.graph!.node!.length).toBe(47)
      expect(getInputNames(merged)).toEqual(['input'])
      for (const key of Object.keys(params)) {
        expect(getInitializerNames(merged)).toContain(`m0_${key}`)
      }
      expect(validateConnectivity(merged)).toEqual([])
    })
  })

  // ---- 19. 複雑な複合チェーン (affine → hsv_range → crop) ----

  describe('複雑な複合チェーン', () => {
    it('affine(40) → hsv_range(47) → crop(28) = 115ノードが正しく結合される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('05_geometric', 'affine'), params: { a: 1, b: 0, tx: 0, c: 0, d: 1, ty: 0 } },
        { model: loadModel('02_color_space', 'hsv_range'), params: { h_min: 0, h_max: 1, s_min: 0, s_max: 1, v_min: 0, v_max: 1 } },
        { model: loadModel('05_geometric', 'crop'), params: { crop_top: 0, crop_left: 0, crop_h: 1, crop_w: 1 } },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_complex'))

      expect(merged.graph!.node!.length).toBe(40 + 47 + 28)
      expect(getInputNames(merged)).toEqual(['input'])
      expect(getOutputNames(merged)).toEqual(['output'])

      // パラメータ数: affine(6) + hsv_range(6) + crop(4) = 16
      const initParamCount = getInitializerNames(merged).filter((n) =>
        ['m0_a', 'm0_b', 'm0_tx', 'm0_c', 'm0_d', 'm0_ty',
         'm1_h_min', 'm1_h_max', 'm1_s_min', 'm1_s_max', 'm1_v_min', 'm1_v_max',
         'm2_crop_top', 'm2_crop_left', 'm2_crop_h', 'm2_crop_w'].includes(n),
      ).length
      expect(initParamCount).toBe(16)

      expect(validateConnectivity(merged)).toEqual([])
    })
  })

  // ---- 20. unsharp_mask (パラメータ + 属性の組み合わせ) ----

  describe('パラメータ + 属性の組み合わせ', () => {
    it('unsharp_mask_3x3 → unsharp_mask_5x5: パラメータと属性が独立して保持される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('03_conv_filter', 'unsharp_mask_3x3'), params: { amount: 2.0 } },
        { model: loadModel('03_conv_filter', 'unsharp_mask_5x5'), params: { amount: 1.5 } },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_unsharp'))

      // パラメータがそれぞれベイクされている
      expect(getInitializerNames(merged)).toContain('m0_amount')
      expect(getInitializerNames(merged)).toContain('m1_amount')
      const init0 = merged.graph!.initializer!.find((i) => i.name === 'm0_amount')!
      const init1 = merged.graph!.initializer!.find((i) => i.name === 'm1_amount')!
      expect(init0.floatData![0]).toBeCloseTo(2.0)
      expect(init1.floatData![0]).toBeCloseTo(1.5)

      // Conv 属性が 2 つ存在（異なるカーネルサイズ）
      const convNodes = merged.graph!.node!.filter((n) => n.opType === 'Conv')
      expect(convNodes.length).toBe(2)

      expect(validateConnectivity(merged)).toEqual([])
    })
  })

  // ---- 21. opset_import の保持 ----

  describe('メタデータの保持', () => {
    it('opset_import が最初のモデルから引き継がれる', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'grayscale'), params: {} },
        { model: loadModel('03_conv_filter', 'sharpen'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_opset'))

      const srcOpset = decoded[0].model.opsetImport!
      expect(merged.opsetImport!.length).toBe(srcOpset.length)
      expect(merged.opsetImport![0].domain).toBe(srcOpset[0].domain)
      expect(merged.opsetImport![0].version!.toString()).toBe(srcOpset[0].version!.toString())
    })

    it('irVersion が保持される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'grayscale'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_ir'))

      expect(merged.irVersion!.toString()).toBe(decoded[0].model.irVersion!.toString())
    })

    it('producerName が設定される', () => {
      const decoded: DecodedModel[] = [
        { model: loadModel('01_elementwise', 'grayscale'), params: {} },
      ]
      const merged = decodeMerged(mergeDecodedModels(decoded, 'test_producer'))

      expect(merged.producerName).toBe('onnx-cv-graph-node-editor')
    })
  })
})
