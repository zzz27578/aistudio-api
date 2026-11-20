import { request } from './client.ts'

export type ThinkingLevel = 'MINIMAL' | 'LOW' | 'MEDIUM' | 'HIGH'
export interface AliasPolicy { enabled: boolean; visible: boolean }
export interface ModelPolicy {
  enabled: boolean
  visible: boolean
  thinking_aliases: Partial<Record<ThinkingLevel, AliasPolicy>>
  text_wrapper: AliasPolicy
  wrapper_thinking_aliases: Partial<Record<ThinkingLevel, AliasPolicy>>
}
export interface CatalogModel {
  id: string
  metadata: {
    displayName?: string
    description?: string
    category?: string
    inputTokenLimit?: number
    outputTokenLimit?: number
    capabilities?: string[]
    defaultThinkingLevel?: number
  }
  policy: ModelPolicy
  supported_levels: ThinkingLevel[]
  aliases: Record<ThinkingLevel, string>
  supports_text_wrapper: boolean
  wrapper_alias: string
  wrapper_thinking_aliases: Record<ThinkingLevel, string>
}
export interface CatalogResponse {
  models: CatalogModel[]
  refreshed_at: number | null
  source: 'upstream' | 'fallback'
  refresh_error: string | null
}
export const modelsApi = {
  list(refresh = false): Promise<CatalogResponse> {
    return request(`/api/model-catalog?refresh=${refresh}`)
  },
  save(id: string, policy: ModelPolicy): Promise<{ ok: boolean }> {
    return request(`/api/model-catalog/${encodeURIComponent(id)}`, {
      method: 'PUT', body: JSON.stringify(policy),
    })
  },
}

export interface UpdateStatus {
  current_revision: string
  custom_build: string
  latest_revision: string
  branch: string
  status: string
  checked_at: number
  commit_message: string
  committed_at: string
  commit_url: string
  compare_url: string | null
  ahead_by: number | null
}
export function checkUpdate(): Promise<UpdateStatus> {
  return request('/api/system/check-update')
}
