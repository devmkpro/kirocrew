import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Users } from 'lucide-react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../api/client'
import { Input } from './ui'
import { i18nT } from '../i18n/t'

/**
 * Composer chip picking the model SUB-AGENTS run on.
 *
 * Sibling of the main model chip, and deliberately a separate control rather
 * than another row inside the model dropdown: the two answer different
 * questions ("what am I talking to" vs "what does the work I delegate run on")
 * and the second is the one a user changes to stop delegated work riding the
 * same expensive model as the conversation.
 *
 * Writes `agent.role_models.subagent`, the SAME key Settings > Chat and
 * Settings > Codebrain edit, so the quick pick and the settings row cannot
 * disagree. `auto` means "the provider picks" — NOT "inherit the chat model":
 * `RoleModels.resolve_model` never falls back to `agent.model`, precisely so
 * unattended work cannot silently ride the interactive flagship.
 *
 * Self-contained on purpose (own queries, own mutation) because the alternative
 * is threading four more props through a composer prop surface that is already
 * enormous, for a control that shares no state with it.
 */
export default function SubagentModelChip({ disabled }: { disabled?: boolean }) {
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const [rect, setRect] = useState<DOMRect | null>(null)
  const panelRef = useRef<HTMLDivElement | null>(null)

  const cfg = useQuery({ queryKey: ['kirocrew-config'], queryFn: () => api.kirocrewConfig() })
  // Only fetched while the menu is open: the list costs a backend read and the
  // chip's own label does not need it.
  const models = useQuery({
    queryKey: ['models'],
    queryFn: () => api.models(),
    enabled: open,
  })

  const current =
    (cfg.data as { agent?: { role_models?: Record<string, string> } } | undefined)?.agent
      ?.role_models?.subagent || 'auto'

  const save = useMutation({
    mutationFn: (model: string) =>
      api.patchConfig('agent.role_models.subagent', model === 'auto' ? '' : model),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['kirocrew-config'] }),
  })

  // Close on outside click / Escape, matching the other composer dropdowns.
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const rows: string[] = (() => {
    const list = Array.isArray(models.data)
      ? (models.data as { model_name?: string }[])
          .map(m => String(m.model_name || ''))
          .filter(Boolean)
      : []
    const out = list.includes('auto') ? [...list] : ['auto', ...list]
    // A pin the backend no longer advertises must stay visible, or opening the
    // menu would silently offer to replace it with nothing.
    if (!out.includes(current)) out.unshift(current)
    const q = filter.trim().toLowerCase()
    return q ? out.filter(m => m.toLowerCase().includes(q)) : out
  })()

  const label = current === 'auto' ? i18nT('pages.settings.chatPanel.role_model_auto') : current

  return (
    <>
      <button
        type="button"
        className="inline-flex items-center gap-1.5 h-7 min-w-0 text-[12px] text-muted hover:text-text px-2 rounded-md bg-transparent hover:bg-[color-mix(in_srgb,var(--bg-elevated)_84%,var(--text))] transition-colors border-none cursor-pointer disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-muted"
        disabled={disabled}
        data-testid="composer-subagent-model-chip"
        title={`Sub-agent model: ${label}`}
        aria-label={`Sub-agent model: ${label}`}
        onClick={e => {
          setRect(e.currentTarget.getBoundingClientRect())
          setFilter('')
          setOpen(v => !v)
        }}
      >
        <Users size={13} className="shrink-0" />
        <span className="truncate max-w-[110px]">{label}</span>
      </button>
      {open &&
        rect &&
        createPortal(
          <div
            ref={panelRef}
            role="dialog"
            aria-label="Sub-agent model"
            className="fixed z-[9999] bg-bg-elevated border border-border rounded-xl shadow-xl min-w-[240px] max-w-[320px] flex flex-col p-1 gap-0.5"
            style={(() => {
              const left = Math.max(8, Math.min(rect.left, window.innerWidth - 328))
              return { bottom: window.innerHeight - rect.top + 4, left }
            })()}
          >
            <div className="px-1.5 pt-1.5 pb-1">
              <Input
                type="text"
                autoFocus
                aria-label="Filter models"
                placeholder={i18nT('pages.chatPage.type_to_filter')}
                value={filter}
                onChange={e => setFilter(e.target.value)}
                className="w-full px-2 py-1 text-[13px]"
              />
            </div>
            <div className="px-2 pb-1 text-[11px] text-muted">
              Model for sub-agents you spawn. <code>auto</code> lets the provider choose — it does
              not inherit the chat model.
            </div>
            <div role="listbox" aria-label="Sub-agent models" className="overflow-y-auto max-h-[280px]">
              {models.isLoading && <div className="px-2 py-1.5 text-[12px] text-muted">Loading…</div>}
              {!models.isLoading && rows.length === 0 && (
                <div className="px-2 py-1.5 text-[12px] text-muted">No matches</div>
              )}
              {rows.map(model => (
                <button
                  key={model}
                  type="button"
                  role="option"
                  aria-selected={model === current}
                  className={`w-full text-left px-2 py-1.5 text-[13px] rounded-md bg-transparent border-none cursor-pointer hover:bg-[color-mix(in_srgb,var(--bg-elevated)_84%,var(--text))] ${
                    model === current ? 'text-accent' : 'text-text'
                  }`}
                  onClick={() => {
                    save.mutate(model)
                    setOpen(false)
                  }}
                >
                  {model === 'auto' ? i18nT('pages.settings.chatPanel.role_model_auto') : model}
                  {model === current && ' ✓'}
                </button>
              ))}
            </div>
            {save.isError && (
              <div className="px-2 py-1.5 text-[12px] text-danger">
                Could not save the sub-agent model.
              </div>
            )}
          </div>,
          document.body,
        )}
    </>
  )
}
