import { useQuery } from '@tanstack/react-query'
import { CheckCircle2, CircleSlash } from 'lucide-react'

import { SettingsCard } from '../../components/settings'
import { api } from '../../api/client'
import { i18nT } from '../../i18n/t'

/**
 * Developer > Agent backend — which vendor agent CLI this host can actually launch.
 *
 * Why this card exists beside the backend switch rather than inside it: the switch
 * answers "which harness should Kiro Crew drive", and this answers "what is
 * installed on this machine". They fail independently. An operator with no Kiro
 * account whose balance and config both fail to load needs to see that `codex` is
 * sitting right there in `~/.local/bin` — and the backend switch alone never says
 * so, because a backend can be selectable while its binary is absent.
 *
 * It reports EVERY declared CLI, including the absent ones, each with its install
 * hint. A picker that listed only what resolved could not tell the user what they
 * are missing, and "the option is not there" is the least diagnosable answer a
 * settings page can give.
 *
 * `credential_note` per row is the other half of the point. For Claude and Codex
 * there is deliberately no API-key field anywhere in this app: the vendor's CLI
 * holds the credential and Kiro Crew only invokes it. Saying that ON the row is
 * what stops a reader from hunting for a key field that does not exist.
 *
 * Refresh policy mirrors `AgentBackendTab`'s probe, and for its reason: this app
 * sets a global `staleTime: Infinity`, so inheriting it would freeze the answer for
 * the life of the page — an operator who follows the install hint would see the CLI
 * still missing with no way to re-ask short of a reload.
 */

const REFRESH_MS = 30_000

export function AgentCliCard() {
  const q = useQuery({
    queryKey: ['agentClis'],
    queryFn: () => api.agentClis(),
    // A 403 for a non-owner and a 404 on a gateway predating the endpoint are
    // permanent answers; retrying only delays the empty state below.
    retry: false,
    staleTime: 0,
    refetchInterval: REFRESH_MS,
  })

  const rows = q.data?.agent_clis ?? []
  const installed = rows.filter((r) => r.installed)

  return (
    <SettingsCard>
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1">
          <div className="text-sm font-medium">
            {i18nT('pages.developer.agentCliCard.title')}
          </div>
          <div className="text-xs opacity-70">
            {i18nT('pages.developer.agentCliCard.description')}
          </div>
        </div>

        {q.isError && (
          <div className="text-xs opacity-70">
            {i18nT('pages.developer.agentCliCard.could_not_probe')}
          </div>
        )}

        {!q.isError && rows.length === 0 && (
          <div className="text-xs opacity-70">
            {i18nT('pages.developer.agentCliCard.probing')}
          </div>
        )}

        {rows.length > 0 && (
          <>
            <div className="text-xs opacity-70">
              {i18nT('pages.developer.agentCliCard.found_count')}
              {': '}
              {installed.length}
              {' / '}
              {rows.length}
            </div>
            <ul className="flex flex-col gap-2">
              {rows.map((row) => (
                <li key={row.id} className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    {row.installed
                      ? <CheckCircle2 size={14} aria-hidden />
                      : <CircleSlash size={14} aria-hidden className="opacity-50" />}
                    <span className="text-sm">{row.display}</span>
                    <span className="text-xs font-mono opacity-60">{row.binary}</span>
                    <span className="text-xs opacity-70">
                      {row.installed
                        ? i18nT('pages.developer.agentCliCard.state_found')
                        : i18nT('pages.developer.agentCliCard.state_missing')}
                    </span>
                  </div>
                  {/* The absolute path, because it is what proves WHICH binary
                      would run — the same pinning the resolver does server-side. */}
                  {row.installed && row.path && (
                    <div className="text-xs font-mono opacity-60 break-all">{row.path}</div>
                  )}
                  {!row.installed && row.install_hint && (
                    <div className="text-xs opacity-70">{row.install_hint}</div>
                  )}
                  <div className="text-xs opacity-60">{row.credential_note}</div>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </SettingsCard>
  )
}
