import { useEffect, useState } from 'react'
import type { ModelsMeta, ParamMeta } from '../types'

export function useModelsMeta(base: string) {
  const [meta, setMeta] = useState<ModelsMeta | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${base}models/models_meta.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((json: ModelsMeta) => setMeta(json))
      .catch((e: Error) => setError(e.message))
  }, [base])

  return { meta, error }
}

/**
 * models_meta.json の params エントリを ParamMeta 型に変換する
 */
export function toParamMeta(
  raw: Record<string, [number, number, number]>,
): Record<string, ParamMeta> {
  return Object.fromEntries(
    Object.entries(raw).map(([k, [min, max, def]]) => [
      k,
      { min, max, default: def },
    ]),
  )
}
