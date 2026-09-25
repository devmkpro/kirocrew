# codebrain-tools — Codebrain como App MCP do fork

Traz as tools do Codebrain para dentro deste fork **sem mesclar código**. O
Codebrain roda como está, num processo próprio; o fork só o registra como
servidor MCP stdio.

## Instalar

```bash
mkdir -p ~/.kiro/crew/apps/codebrain-tools
cp app.json ~/.kiro/crew/apps/codebrain-tools/app.json
# ajuste o caminho de packages/mcp/stdio.js no manifest antes, se o checkout mudou
```

Depois reinicie o gateway. Confirme que subiu:

```bash
kirocrew app list | grep codebrain
```

## Por que stdio e não a URL HTTP

O daemon do Codebrain serve MCP em `http://127.0.0.1:<porta>/mcp` (Streamable
HTTP), e seria tentador registrar `url`. Mas MCP remoto HTTP/SSE **não roteia
pelo stub de MCP Apps** do KiroCrew. O launcher `packages/mcp/stdio.js` é o
caminho suportado — e ele já resolve descoberta e boot: procura um daemon vivo em
`~/.codebrain/daemon.json` e, se não achar, sobe um com `ELECTRON_RUN_AS_NODE=1`.

Consequência prática: **o Codebrain precisa estar instalado na máquina.** Este
App não empacota nada — não há campo de manifest que embuta um runtime Node no
instalador do KiroCrew.

## O que está ligado, e o que não

Ligado: o grupo `fetch` (`browser_fetch`, `_json`, `_html`, `_batch`,
`_cookies`) e as ops de browser via CDP. Os grupos avançados do Codebrain já
nascem desabilitados no próprio servidor (`packages/mcp/index.js:1805-1810`) e
sobem sob demanda com `enable_tool_group`.

`disabledTools` corta três classes, de propósito:

- **`memory_*`** — o Codebrain grava `~/.codebrain/memory.db`. Rodando junto com
  a Memory V2 do fork você teria dois sistemas que aprendem e nenhum sabendo do
  outro; cada busca devolveria metade da verdade. A memória autoritativa é a do
  fork.
- **`pane_*`** — dependem do modelo de panes/PTY do Codebrain e colidiriam com o
  orquestrador de subagents do fork.
- **`browser_eval` e `file_write`** — executam JS arbitrário numa página
  autenticada e escrevem no disco, entrando pelo gateway como tools externas.

## Ressalvas de segurança

O próprio `apps/manifest.py` observa que `command`/`args`/`env` de um
`mcpServers` são gravados na config que o kiro-cli **spawna**, e que esse é o
único trecho de um app assinado cujo programa poderia ser trocado com a
assinatura ainda validando. Trate o caminho no manifest como código.

Além disso: as tools de browser do Codebrain **não têm** o bloqueio de
loopback/IP privado que o `browser` nativo do fork aplica em
`_navigate_target_is_public`. É por isso que elas conseguem testar um dev server
— e é por isso que merecem revisão consciente de `autoApprove`.

## Limitação conhecida do modo browser

O modo webview embutido do Codebrain está **desativado**: `browserCmd` retorna
`"Webview mode is disabled"` quando não encontra CDP
(`packages/mcp/bridge/browser-handlers.js:169-174`). As tools de browser só
funcionam contra um Chrome real aberto com porta de depuração:

```bash
google-chrome --remote-debugging-port=9222
```

Sem isso, só o grupo `fetch` funciona — que é justamente o que não depende de
Electron nem de Chrome.
