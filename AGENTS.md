# rules

everything below is binding. it is short on purpose — read all of it, every session.

## writing convention

follow my casual writing convention: text in docs, comments, discussions, readmes and debugging output is **all lowercase**. only capitalize project, platform, or people's names, **if applicable**. electron-vite, for a contrary example, is a name written in lowercase, as the original namers intended. examples:

- Claude Code, Pi, OpenCode, Ollama, Electron, electron-vite, zustand, Windows, Linux

on the contrary, **in-app UI is cased as per usual**. "New Session", not "new session".

## the prose dash

which dash a sentence takes is decided by where the sentence lives, not by what it says.

|                                                |                                                              |
| ---------------------------------------------- | ------------------------------------------------------------ |
| **markdown documents, everywhere**             | `—`. private and public alike, this file included            |
| **commit messages, everywhere**                | ` -- `                                                       |
| **code comments and logging** | ` -- `. every language, and the strings a run prints as well as the comments around them |

**`--` is a build error inside an XML comment**. use a single `-` instead.

**and a prose dash beside a real flag reads as an argument.** *"name one with `--model qwen3.8`"* after a ` -- ` is a sentence nobody parses correctly; those take a colon.

**a user-facing string is neither a comment nor logging** and keeps normal typography.

## no manual wrapping

**never hard-wrap prose.** a paragraph is one line and the editor wraps it. this binds **commit messages** — subject line, then paragraphs each on a single line — and **every document**. a manual wrap is invisible work that has to be redone every time a sentence changes, and it turns a one-word edit into a reflowed diff hunk.

code is unaffected: comments and script headers wrap to the column the file around them uses.

## commits

i like commits across the whole repo nice and streamlined. try and follow these roles:

- **subjects are `<type>(<scope>): <subject>`**, Conventional Commits. types: `feat fix docs style aesthetics refactor perf test build ci chore revert`, plus **`diag`**, which is not standard. a `diag/` commit is never merged anywhere. scopes in the repo:
  - `host` — AgentHost and the session states above it: starting, restarting and killing harness processes, and each session's transcript and turn
  - `acp` — the ACP client side and harness profiles
  - `proxy` — the local HTTP proxy between the harnesses and Ollama
  - `ollama` — polling and managing Ollama
  - `ipc` — the typed channel between the main process and the renderer
  - `ui` — the renderer
  - `settings` — wrontend's settings file
  - `components` — the component engine: installing, versioning and resolving the parts wrontend downloads where they are first needed
  - `browser` — finding models: searching Hugging Face, a repo's quants and what fits on this machine
  - `updater` — checking harnesses, Ollama and models for updates, and applying them
  - `build` — packaging and the installer

  a commit that touches the whole repo or more than one scope takes no scope. the list grows when a new part of the app earns one.
- **three of those types are chosen wrongly more often than the rest, so the line between them is drawn here.** `fix` is the app breaking a promise it keeps elsewhere — something somebody could file as a bug. `feat` is the app doing something it did not do at all before, whose absence nobody would have called a bug. `style` is the same behaviour differently arranged: nothing was wrong, and it now looks better.
- `aesthetics` means a visual change to the app. that divergence is deliberate and this sentence is the whole of the warning it needs.
- **the test that settles the hard cases is whether a bisect could skip the commit.** a `aesthetics` commit is one that can be passed over while hunting a behavioural regression, because no code path moved — a margin, an order, a colour. a commit that adds machinery is not skippable however inert its pixels look, and calling it `aesthetics` tells a future bisect to ignore the one change that could have caused what it is hunting. a UI change with no behaviour in it is `aesthetics`; a UI change carried by new code is `feat`, or `fix` where the behaviour was already meant to be there.
- **commits to code are reasonably atomic, and a commit touching only comments and documents is one commit with its concerns stacked.** what atomicity buys is a bisect that can land on one change or pass over it, so separable code concerns — a mechanism and the feature using it, an optimisation, a pre-existing bug found on the way — each get a commit. **it can be taken too far**: a commit per file, per hunk or per step of one refactor costs review and buys a bisect nothing, and two changes that only make sense together are one concern however many files they touch. nothing bisects a comment or a document, so splitting a docs change has no benefit at all.
- **a commit message carries no markup.** no backticks around a symbol, no asterisks for emphasis, no bracketed links — a message is read by `git log`, `git show`, `shortlog`, `git blame` and every terminal and mail client that renders none of it, so a symbol wrapped in backticks arrives with the backticks still on it. name the symbol, the file or the flag bare; the prose has to carry the emphasis by itself, which is a fair constraint on a message this size. **markdown documents are unaffected** — they are rendered, and the table at the top of this file still decides their dash.
- **propose before committing, ask before pushing.** a branch name and every commit message are proposed in chat, complete, and approved before they are made, and a commit waits until the change it carries is verified, unless parking work on a local topic branch was asked for. **nothing is pushed without asking, every time**: naming the branch a commit belongs on is not permission to push it, and the same holds for a tag, a release or a post to an issue.
- **a rewritten message keeps `Co-Authored-By` as its last line.** GitHub credits a co-author only then, and `git commit-tree` stores a message verbatim where `git commit` cleans it up, so a rebuilt commit can end in a blank line that drops the credit while a check that the trailer merely exists still passes. assert that it is the last line.
- **branches: a `<type>/` prefix** means a topic; `<type>/<topic>`, e.g.: `fix/ndjson-line-split`
- **an instrument that measures a part which ships goes on `main`, behind a flag that is off by default.** a `diag/` branch is for an instrument bound to one bug, which will be closed. a branch cut to hold a counter rots against the very file it measures, and a capability that needs a checkout and a rebase before it can be used is one nobody reaches for.

### commit authors

for transparency and tool usage attribution reasons: **any commit** that **includes code written by any AI agent** properly displays as co-authored by said agent on GitHub. this is achieved by appending a functional `co-authored-by: ...` trailer:

```
{commit message}

co-authored-by: {agent}/{model with version} <{email address}>
```

three concrete examples:

```
fix(acp): split incoming records on newlines only

Node's readline also splits on U+2028 and U+2029, which are legal inside a JSON string. splitting on \n alone keeps a message chunk that carries either in one record.

co-authored-by: Claude Code/Claude Opus 5.5 <noreply@anthropic.com>
```

```
co-authored-by: Antigravity/Gemini 3.8 Flash <gemini@mircowuffwuff.com>
```

```
co-authored-by: Codex/GPT 6.1 Sol <gpt@mircowuffwuff.com>
```

what follows is a table of exact email's to pass for popular models. proper LLM model attribution in GitHub's UI can only be guaranteed this way. if the model that powers an agent is listed, the listed email is to be used, no matter what:

| model | email address |
| --- | --- |
| Claude | noreply@anthropic.com |
| Gemini | gemini@mircowuffwuff.com |
| GPT | gpt@mircowuffwuff.com |
| Qwen | qwen@mircowuffwuff.com |
| DeepSeek | deepseek@mircowuffwuff.com |
| Grok | grok@mircowuffwuff.com |
| Muse | muse@mircowuffwuff.com |
| MiMo | mimo@mircowuffwuff.com |
| GLM | glm@mircowuffwuff.com |
| Kimi | kimi@mircowuffwuff.com |

## public text

**public text** means anything that is committed, and every commit message. comments and script headers count. these rules bind it:

- **no reference that a reader outside this workspace cannot resolve.** no uncommitted document names, no internal roadmap labels (`M3`, `item 16`). this cost a whole session to clean up once, and the mistake was made *again* in the commit messages that cleaned it up, by quoting the references being removed. an illustration of an unresolvable reference is still an unresolvable reference. what a comment may cite is what conventionally resolves: an issue, a commit, a symbol, a released version. **a glossary is not the answer either** — defining the labels publicly makes an unusual practice official.
- **say the thing instead.** *"the main process owns every harness process"*, not *"see M1"*.
- **present tense. public text describes the software as it is, never how it got here.** no dates, no *"used to"*, no *"was removed on"*, no *"decided by the maintainer on"*. a reader outside this workspace cannot check any of it and does not need to: they want to know what the code does now. **this binds code comments exactly as it binds a document** — a comment is public documentation that happens to live beside the thing it describes
- **a future tense is the same violation**, and the one that survives a sweep: *"once this becomes a slider"* is wrong the day the future arrives, and no grep for a date or *used to* finds it. only reading does.
- **but a rejected alternative is not history — it is a reason, and it stays.** *"Tauri gives a lighter binary, but it adds a second language and still needs a Node sidecar for the agent side"* is what stops the next person doing it, and it costs nothing to say in the present. what goes is the diary around it: when it was tried, who decided, what the previous version looked like. **rewrite those into the reason; do not delete them.** and note that *"a build that is gone"* or *"a path the app can no longer open"* is runtime state rather than project history, and is not what this is about. **this carve-out stops at commit messages.** a document or a comment is read by somebody deciding what to do next, so the alternative that was passed over is a reason they need. a commit message describes one change in the state it is committed in and nothing else — no *“used to”*, no *“instead of”*, no alternative that is not in the diff. the reason still gets written down; it goes in the comment the commit is adding, where the next person reads it.
- **a rename is not finished until the old name is gone from comments and documents.** a section, row, screen or label that changes leaves its old name in doc comments and `docs/`, and nothing fails on it. grep for the old name in the same change that retires it.
- **no line numbers** into **reference projects** — those move, and the citation is wrong one commit later. name the file and the symbol.