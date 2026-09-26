# Codebrain → Kiro Crew: port plan

Source: `D:\Projetos\codebrain` — Electron + React 19 + TypeScript, 43,207 lines
across 234 TS/TSX files, plus `packages/mcp` (~60 JS handlers) and
`packages/memory` (12 files, SQLite). MIT licensed.

This is a **port**, not a copy. Each item below lands as its own PR with the
owning spec updated in the same commit, per `AGENTS.md`.

## Why not a file copy

| | Codebrain | Kiro Crew |
|---|---|---|
| Shape | Electron desktop app | Python backend + SPA it serves |
| Unit of work | a *pane* = a PTY running a CLI | a session/slot + subagent |
| Model access | multi-provider by API key | one provider, speaks ACP |
| Memory | SQLite + pure-JS TF-IDF | own embeddings / vectors |
| Governance | none | `POLICY ∩ PROFILE`, OS sandbox |
| MCP | 244 tools, module-global state | tools MUST be stateless |

Codebrain is a **terminal orchestrator**: its value is running N CLIs in visible
PTYs and coordinating them. Kiro Crew is **one agent with host capabilities**.
So most features do not "copy" — they either already exist here under another
name, or they depend on Electron/PTY machinery this repo does not have.

## Classification

### A. Already exists here — do not port

cron, skills (`builtin_skills/`), hooks, memory + lessons, sessions/history,
compaction/session-summary, token cost, voice (`stt/`, `voice_reply.py`),
subagents + spawn, crews (≈ squads), workflows, artifacts, connections,
messaging.

Porting any of these creates a second, competing system. Where Codebrain's
version is genuinely better, improve the existing subsystem instead.

### B. Worth porting

| Feature | Destination | Friction |
|---|---|---|
| Kanban / shared task board | dashboard + store | low |
| Work ledger (resumable state) | memory / session | low |
| Trajectory tracking + pattern extraction | memory | medium |
| Knowledge graph (PageRank, TF-IDF) | memory | medium — overlaps current vectors |
| Agent scoring (multi-factor) | `subagent_manager/admission/fairness.py` | medium |
| Pipeline fan-out / fan-in | pipeline-conductor (exists) | medium |
| Consensus (raft / gossip / byzantine) | new | high; may be over-engineering |
| Goal/Judge + task-gate pre-stop | autonudge (exists) | medium |
| Verify (visual proof of a front-end change) | e2e-gate | medium |
| LSP navigation (12 tools) | new MCP server | medium, high value |
| Max mode (best-of-N), Compose, Plan agent | workflows | high |

### C. Not portable — Electron/PTY bound

Tiling pane grid, focus mode, xterm.js, Discord Rich Presence, native desktop
notifications, Windows-registry auto-update, global push-to-talk, window
shortcuts. There is no Electron main process here to host them; `website/electron/`
exists for a different purpose.

### D. Blocked by a repo invariant

- **Browser automation (53 CDP tools)** — `computer_use/` already owns this and is
  deliberately ungoverned behind one operator opt-in. A second browser path
  outside that gate is what `AGENTS.md` forbids.
- **API-key providers / 140 models** — `agent.provider` is `enum=["acp"]`; a new
  harness arrives at `agent.acp_backend` and adapts to existing seams. Never
  hardcode a model id.
- **Background workers (7 daemons)** — must go through `monitor-architecture`,
  not a bare interval loop.
- `no-new-builtin-apps` and `internal-content-scan` block dropping a new tree in.

## Where the current branch already sits

`feat/providers-and-agent-inbox` is the leading edge of this port: `CodebrainProvider`
drives the codex/claude CLI directly, and the agent-to-agent inbox is
`message-bus.js` reimplemented idiomatically. The direction is already correct.

## Sequence

Ordered by dependency. Each step is one PR.

0. **Normalize the current branch.** 12 commits, three titled `codebrain`, no doc
   updates. `providers/` and `admission/gate.py` both have owning specs that must
   be written before more work stacks on top.
   - [x] `providers.md` — described the second provider and what it forfeits.
   - [x] `AGENTS.md` — recorded the fork divergence on `agent.provider`.
   - [x] `config/loader.py` — its factory docstring still claimed "KiroACP-only"
         three lines above the branch that contradicts it.
   - [ ] `admission/gate.py` — `_parent_project_root` is undocumented.
1. **Work ledger + kanban** — low friction, and the substrate the rest reports into.
2. **Trajectories + knowledge graph** — decide first: replace the current vector
   memory, or coexist with it? Most expensive decision to reverse.
3. **Agent scoring → admission/fairness** — a real, motivated reason to touch the gate.
4. **LSP tools** — self-contained, high value.
5. **Goal/Judge, verify, pipeline** — each touches an existing subsystem.

Items in C and D are out of scope until explicitly decided otherwise.

## Fork governance

This fork (`devmkpro/kirocrew`) diverges from upstream `kirodotdev/KiroCrew` on
one architectural invariant: **`agent.provider` is open**, and carries a second
value, `codebrain`. Upstream keeps the enum closed at `acp` and states so in
`providers.md`; that statement was already false in this tree before this plan
existed, and the specs now describe what the code does.

No entry was added to `docs/decisions/`. That ledger's rules require a maintainer
listed in `MAINTAINERS.md` — which names the three upstream maintainers, not this
fork's owner — and an on-record artifact rather than a conversation. The decision
is recorded here instead, which is the honest place for it. If this work is ever
proposed upstream, the divergence is the first thing to raise, and it needs a
real decision entry written by one of those maintainers.

What the divergence does **not** touch: governance (`POLICY ∩ PROFILE`), the
PreToolUse gate, the OS sandbox, credential masks, the no-hardcoded-model rule.
A foreign CLI child is governed as one.

## Open questions

- Does the knowledge graph **replace** Kiro Crew's vector memory, or **coexist**?
- Is consensus (raft/gossip/byzantine) actually wanted, or is it complexity
  without a caller here?
