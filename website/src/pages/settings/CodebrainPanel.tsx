import { useEffect, useMemo, useState } from 'react'
import { Cpu, Plus, RefreshCw, Trash2 } from 'lucide-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { SettingsSection, SettingsCard } from '../../components/settings'
import { Badge, Btn, IconButton, Input, PanelSectionHeader } from '../../components/ui'
import ErrorNotice from '../../components/ErrorNotice'
import { api, type CodebrainProvider, type CodebrainProviderInput } from '../../api/client'

/**
 * Settings > Codebrain — the direct-provider seam.
 *
 * Two different kinds of thing share this panel deliberately, because the user's
 * question is one question ("what can this install talk to?"):
 *
 *  - AUTO-DETECTED native CLIs. Nothing to configure: a profile exists exactly
 *    when its executable is on PATH, so the row states a fact and offers no
 *    controls. This is Codebrain's "virtual provider" idea.
 *  - STORED endpoint profiles (OpenRouter, DeepSeek, a team gateway). These are
 *    edited here and persisted to the provider store the resolver reads.
 *
 * Tokens are write-only: the backend reports presence, never the value, so an
 * existing key renders as "saved" and an untouched field saves nothing.
 */

/** Provider protocol types the direct seam knows how to route. */
const PROVIDER_TYPES = [
  'anthropic-compat',
  'mimo-compat',
  'openai-compat',
  'gemini-compat',
  'codex',
] as const

/** One-click starting points for the endpoints people actually add. Each is a
 *  DRAFT the user still edits and saves — nothing is written on click. */
const PRESETS: readonly CodebrainProviderInput[] = [
  {
    id: 'openrouter',
    label: 'OpenRouter',
    type: 'openai-compat',
    host: 'openclaude',
    baseUrl: 'https://openrouter.ai/api/v1',
    tokenEnvVar: 'OPENAI_API_KEY',
    models: ['anthropic/claude-sonnet-4.6', 'google/gemini-3.5-flash', 'deepseek/deepseek-chat'],
  },
  {
    id: 'deepseek',
    label: 'DeepSeek',
    type: 'openai-compat',
    host: 'openclaude',
    baseUrl: 'https://api.deepseek.com/v1',
    tokenEnvVar: 'OPENAI_API_KEY',
    models: ['deepseek-chat', 'deepseek-reasoner'],
  },
  {
    id: 'mimo',
    label: 'MIMO',
    type: 'mimo-compat',
    host: 'openclaude',
    baseUrl: 'https://token-plan-ams.xiaomimimo.com/anthropic',
    tokenEnvVar: 'ANTHROPIC_AUTH_TOKEN',
    models: ['mimo-v2.5-pro', 'mimo-v2.5'],
  },
  {
    id: 'groq',
    label: 'Groq',
    type: 'openai-compat',
    host: 'openclaude',
    baseUrl: 'https://api.groq.com/openai/v1',
    tokenEnvVar: 'OPENAI_API_KEY',
    models: ['llama-3.3-70b-versatile'],
  },
]

/** A row being edited. `token` is the only field that is write-only, and
 *  `hasToken` remembers that the STORE holds one even though we cannot read it. */
interface Draft extends CodebrainProviderInput {
  hasToken?: boolean
}

function toDraft(provider: CodebrainProvider): Draft {
  return {
    id: provider.id,
    label: provider.label,
    type: provider.type,
    host: provider.host,
    baseUrl: provider.baseUrl,
    tokenEnvVar: provider.tokenEnvVar,
    models: provider.models,
    hasToken: provider.hasToken,
  }
}

export function CodebrainPanel() {
  const queryClient = useQueryClient()
  const providers = useQuery({
    queryKey: ['codebrain-providers'],
    queryFn: () => api.codebrainProviders(),
  })
  const [drafts, setDrafts] = useState<Draft[] | null>(null)
  const [error, setError] = useState<string>('')

  // Server state seeds the form once per load; later edits are local until save.
  useEffect(() => {
    if (providers.data && drafts === null) {
      setDrafts(providers.data.providers.map(toDraft))
    }
  }, [providers.data, drafts])

  const save = useMutation({
    mutationFn: (rows: Draft[]) =>
      api.saveCodebrainProviders(
        rows.map(({ hasToken: _hasToken, token, ...rest }) => ({
          ...rest,
          // An untouched field must not blank the stored key.
          ...(token && token.trim() ? { token: token.trim() } : {}),
        })),
      ),
    onSuccess: () => {
      setError('')
      setDrafts(null)
      queryClient.invalidateQueries({ queryKey: ['codebrain-providers'] })
    },
    onError: (e: unknown) => setError(e instanceof Error ? e.message : String(e)),
  })

  const detectedCLIs = useMemo(
    () => (providers.data?.registry ?? []).filter(p => p.isVirtual),
    [providers.data],
  )

  // Sub-agent model, read from and written to the SAME config key the composer
  // chip and Settings > Chat use, so the three cannot disagree.
  const cfg = useQuery({ queryKey: ['kirocrew-config'], queryFn: () => api.kirocrewConfig() })
  const modelList = useQuery({ queryKey: ['models'], queryFn: () => api.models() })
  const subagentModel =
    (cfg.data as { agent?: { role_models?: Record<string, string> } } | undefined)?.agent
      ?.role_models?.subagent || 'auto'
  const subagentModelOptions = useMemo(() => {
    const list = Array.isArray(modelList.data)
      ? (modelList.data as { model_name?: string }[])
          .map(m => String(m.model_name || ''))
          .filter(Boolean)
      : []
    const out = list.includes('auto') ? [...list] : ['auto', ...list]
    // Keep a pin the backend no longer advertises selectable, or rendering the
    // select would silently move the stored value to the first option.
    if (!out.includes(subagentModel)) out.unshift(subagentModel)
    return out
  }, [modelList.data, subagentModel])
  const saveSubagentModel = useMutation({
    mutationFn: (model: string) =>
      api.patchConfig('agent.role_models.subagent', model === 'auto' ? '' : model),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['kirocrew-config'] }),
    onError: (e: unknown) => setError(e instanceof Error ? e.message : String(e)),
  })
  const rows = drafts ?? []
  const usedIds = new Set(rows.map(r => r.id))

  const update = (index: number, patch: Partial<Draft>) =>
    setDrafts(current => (current ?? []).map((row, i) => (i === index ? { ...row, ...patch } : row)))

  const addPreset = (preset: CodebrainProviderInput) => {
    // A second copy of the same preset would collide on id in the store, so the
    // duplicate gets a suffix the user can rename.
    let id = preset.id
    let n = 2
    while (usedIds.has(id)) id = `${preset.id}-${n++}`
    setDrafts(current => [...(current ?? []), { ...preset, id }])
  }

  return (
    <div className="space-y-4">
      {error && <ErrorNotice message={error} onDismiss={() => setError('')} />}

      <SettingsSection title="Native CLIs">
        <SettingsCard index={0}>
          <PanelSectionHeader
            label="Auto-detected"
            count={detectedCLIs.filter(p => p.detected).length}
            trailing={
              <IconButton
                aria-label="Re-scan"
                onClick={() => queryClient.invalidateQueries({ queryKey: ['codebrain-providers'] })}
              >
                <RefreshCw size={14} />
              </IconButton>
            }
          />
          <p className="text-[12px] text-muted mb-2">
            A native CLI becomes available the moment its executable is on PATH — there is nothing
            to configure. Select one per session with <code>agent.provider = codebrain</code>.
          </p>
          <ul className="space-y-1">
            {detectedCLIs.map(provider => (
              <li
                key={provider.id}
                className="flex items-center gap-2 py-1 border-b border-[var(--border)] last:border-0"
              >
                <Cpu size={14} className="text-muted" />
                <span className="font-medium">{provider.label}</span>
                <code className="text-[11px] text-muted">{provider.host}</code>
                {provider.detected ? (
                  <Badge variant="ok">detected</Badge>
                ) : (
                  <Badge variant="muted">not installed</Badge>
                )}
                <span className="ml-auto text-[11px] text-muted truncate max-w-[45%]">
                  {provider.models.slice(0, 3).join(', ')}
                </span>
              </li>
            ))}
            {detectedCLIs.length === 0 && (
              <li className="text-[12px] text-muted">No native CLI profiles in this build.</li>
            )}
          </ul>
        </SettingsCard>
      </SettingsSection>

      <SettingsSection title="Sub-agent model">
        <SettingsCard index={1}>
          <PanelSectionHeader label="Model for delegated work" />
          <p className="text-[12px] text-muted mb-2">
            The model sub-agents run on, written to <code>agent.role_models.subagent</code> — the
            same key the chip beside the composer's model picker edits. <code>auto</code> lets the
            provider choose; it does NOT inherit the chat model, so delegated work never silently
            rides the interactive flagship.
          </p>
          <select
            aria-label="Sub-agent model"
            className="bg-[var(--panel)] text-[var(--text)] border border-[var(--border)] rounded px-2 py-1 text-[13px] min-w-[220px]"
            value={subagentModel}
            onChange={e => saveSubagentModel.mutate(e.target.value)}
            disabled={saveSubagentModel.isPending}
          >
            {subagentModelOptions.map(model => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
          {saveSubagentModel.isError && (
            <p className="text-[12px] text-danger mt-1">Could not save the sub-agent model.</p>
          )}
        </SettingsCard>
      </SettingsSection>

      <SettingsSection title="Endpoint providers">
        <SettingsCard index={2}>
          <PanelSectionHeader label="Your providers" count={rows.length} />
          <p className="text-[12px] text-muted mb-2">
            An endpoint profile routes a CLI at another provider by environment — base URL plus a
            token. Stored in <code>{providers.data?.storePath ?? 'providers.json'}</code>. The token
            is write-only: a saved key shows as <em>saved</em> and an empty field leaves it alone.
          </p>

          <div className="flex flex-wrap gap-2 mb-3">
            {PRESETS.map(preset => (
              <Btn key={preset.id} onClick={() => addPreset(preset)}>
                <Plus size={13} /> {preset.label}
              </Btn>
            ))}
            <Btn
              onClick={() =>
                addPreset({ id: 'custom', label: 'Custom', type: 'openai-compat', host: 'openclaude' })
              }
            >
              <Plus size={13} /> Custom
            </Btn>
          </div>

          <div className="space-y-3">
            {rows.map((row, index) => (
              <div key={index} className="rounded-md border border-[var(--border)] p-2 space-y-2">
                <div className="flex items-center gap-2">
                  <Input
                    aria-label="Provider id"
                    value={row.id}
                    placeholder="id"
                    className="w-32"
                    onChange={e => update(index, { id: e.target.value })}
                  />
                  <Input
                    aria-label="Label"
                    value={row.label}
                    placeholder="Label"
                    onChange={e => update(index, { label: e.target.value })}
                  />
                  <select
                    aria-label="Provider type"
                    className="bg-[var(--panel)] text-[var(--text)] border border-[var(--border)] rounded px-2 py-1 text-[12px]"
                    value={row.type}
                    onChange={e => update(index, { type: e.target.value })}
                  >
                    {PROVIDER_TYPES.map(type => (
                      <option key={type} value={type}>
                        {type}
                      </option>
                    ))}
                  </select>
                  <IconButton
                    aria-label="Remove provider"
                    onClick={() => setDrafts(current => (current ?? []).filter((_, i) => i !== index))}
                  >
                    <Trash2 size={14} />
                  </IconButton>
                </div>
                <div className="flex items-center gap-2">
                  <Input
                    aria-label="CLI host"
                    value={row.host}
                    placeholder="cli host (claude / codex / openclaude)"
                    className="w-52"
                    onChange={e => update(index, { host: e.target.value })}
                  />
                  <Input
                    aria-label="Base URL"
                    value={row.baseUrl ?? ''}
                    placeholder="https://…/v1"
                    onChange={e => update(index, { baseUrl: e.target.value })}
                  />
                </div>
                <div className="flex items-center gap-2">
                  <Input
                    aria-label="Token env var"
                    value={row.tokenEnvVar ?? ''}
                    placeholder="OPENAI_API_KEY"
                    className="w-52"
                    onChange={e => update(index, { tokenEnvVar: e.target.value })}
                  />
                  <Input
                    aria-label="API token"
                    type="password"
                    autoComplete="off"
                    value={row.token ?? ''}
                    placeholder={row.hasToken ? 'saved — leave blank to keep' : 'API token'}
                    onChange={e => update(index, { token: e.target.value })}
                  />
                </div>
                <Input
                  aria-label="Models"
                  value={(row.models ?? []).join(', ')}
                  placeholder="model-a, model-b"
                  onChange={e =>
                    update(index, {
                      models: e.target.value
                        .split(',')
                        .map(m => m.trim())
                        .filter(Boolean),
                    })
                  }
                />
              </div>
            ))}
            {rows.length === 0 && (
              <p className="text-[12px] text-muted">
                No endpoint providers yet. Add one from a preset above.
              </p>
            )}
          </div>

          <div className="flex items-center gap-2 mt-3">
            <Btn primary disabled={save.isPending || drafts === null} onClick={() => save.mutate(rows)}>
              {save.isPending ? 'Saving…' : 'Save providers'}
            </Btn>
            <Btn disabled={drafts === null} onClick={() => setDrafts(null)}>
              Discard changes
            </Btn>
          </div>
        </SettingsCard>
      </SettingsSection>
    </div>
  )
}

export default CodebrainPanel
