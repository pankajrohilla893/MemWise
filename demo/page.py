"""The demo page's look and its fixed text: theme, CSS and the HTML sections that don't change.

app.py holds the logic and the parts that update as you chat.
"""

from __future__ import annotations

import gradio as gr

REPO = "https://github.com/pankajrohilla893/MemWise"

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)

HERO = f"""
<section class="mw-hero">
  <div class="mw-hero-main">
    <div class="mw-eyebrow"><span class="mw-live"></span>Live demo · open-source Python library</div>
    <h1>Long-term memory for AI products</h1>
    <p class="mw-lead"><b>memwise</b> decides what's worth keeping, notices when facts stop being
    true, and checks whether a memory should be used at all. Below, the same model answers twice:
    on its own, and with memwise.</p>
    <div class="mw-notjust">
      <span class="mw-notjust-tag">Not just chatbots</span>
      <span>Any product that should remember: coding agents that learn from <b>git commits</b>,
      support desks, shopping, tutoring, AI agents. See <b>Use it in any product</b>.</span>
    </div>
  </div>
  <aside class="mw-hero-side">
    <div class="mw-links">
      <a class="mw-btn mw-btn-primary" href="{REPO}" target="_blank" rel="noopener">GitHub</a>
      <a class="mw-btn" href="{REPO}#benchmark-results" target="_blank" rel="noopener">Benchmarks</a>
      <a class="mw-btn" href="{REPO}/blob/main/docs/how-it-works.md" target="_blank" rel="noopener">Design notes</a>
    </div>
  </aside>
</section>
"""

STEPS = """
<div class="mw-steps">
  <div class="mw-step"><span class="mw-num">1</span><div><b>Tell it about yourself</b>
    <small>or take the tour: one click per step</small></div></div>
  <div class="mw-step"><span class="mw-num">2</span><div><b>Start a new session</b>
    <small>even months later; both chats start empty</small></div></div>
  <div class="mw-step"><span class="mw-num">3</span><div><b>Ask again</b>
    <small>only one side still knows you</small></div></div>
</div>
"""

SIDE_PLAIN = """
<div class="mw-side mw-side-plain"><span class="mw-side-dot"></span><div><b>Without memory</b>
<small>The model on its own. It forgets you when the session ends.</small></div></div>
"""

SIDE_MEMWISE = """
<div class="mw-side mw-side-mem"><span class="mw-side-dot"></span><div><b>With memwise</b>
<small>Same model, same prompt, plus what memwise recalls about you.</small></div></div>
"""

PLACEHOLDER_PLAIN = ("<div class='mw-ph'><b>No long-term memory</b><br>Whatever you tell it is "
                     "gone in the next session.</div>")
PLACEHOLDER_MEMWISE = ("<div class='mw-ph'><b>memwise</b><br>Tell it about yourself, start a new "
                       "session, and ask again.</div>")


def _b(op: str, label: str) -> str:
    return f'<span class="mw-chip"><span class="mw-badge mw-{op}">{op}</span>{label}</span>'


LEGEND = f"""
<div class="mw-legend">
  <div><span class="mw-legend-title">When it writes</span>
    {_b("ADD", "new")}{_b("CONFIRM", "already known")}{_b("SUPERSEDE", "replaces an old one")}
    {_b("RETRACT", "no longer true")}{_b("IGNORE", "not kept")}</div>
  <div><span class="mw-legend-title">When it recalls</span>
    {_b("USE", "in the prompt")}{_b("VERIFY", "ask if still true")}{_b("MENTION", "say it")}
    {_b("HOLD", "left out")}{_b("PAST", "history")}</div>
</div>
"""

USE_CASES = f"""
<section class="mw-section">
  <h2>memwise works wherever a product should remember</h2>
  <p class="mw-sub">The chat above is one example. memwise is a library, not a chatbot: you give it
  whatever your product sees (messages, tickets, commits, events), and before each model call it
  gives back the few memories that matter, ready to add to the prompt. To use it somewhere new,
  you mostly change one thing: the instructions that say what's worth keeping.</p>
  <div class="mw-cards">
    <article class="mw-card mw-card-feature">
      <div class="mw-card-head"><span class="mw-icon">⌥</span><h3>Coding agents and developer tools</h3></div>
      <p>Reads a repository's <b>git commits</b> and keeps what a new developer should know:
      the stack, team rules, and lessons from reverts. Routine commits ("fix typo", "bump
      version") leave nothing behind.</p>
      <div class="mw-flow">
        <div><span class="mw-k">commit</span><code>Revert "Add Redis cache": it ran out of memory</code></div>
        <div><span class="mw-k">remembers</span>Redis ran out of memory at peak load; sessions went back to PostgreSQL</div>
        <div><span class="mw-k">months later</span>"How should I cache sessions?" → "Use PostgreSQL. Redis ran out of memory at peak load."</div>
        <div><span class="mw-k">also</span>"Migrate tests from Jest to Vitest" replaces Jest, and keeps it as history</div>
      </div>
      <a class="mw-try" href="{REPO}/blob/main/examples/git_commits.py" target="_blank" rel="noopener">examples/git_commits.py →</a>
    </article>
    <article class="mw-card">
      <div class="mw-card-head"><span class="mw-icon">✦</span><h3>Chat assistants and companions</h3></div>
      <p>Who the user is, how they like answers, and what's going on in their life. When a fact
      changes, the old one is closed and kept as history, never silently overwritten.</p>
      <div class="mw-eg">"I just moved to Mumbai" → Bangalore becomes history</div>
      <span class="mw-try">Live in this demo ↑</span>
    </article>
    <article class="mw-card">
      <div class="mw-card-head"><span class="mw-icon">☏</span><h3>Customer support</h3></div>
      <p>Each customer's address, sizes and past problems, so nobody has to explain twice.
      Card numbers, security codes and passwords are never stored, even when a customer asks.</p>
      <div class="mw-eg">"My last blender arrived broken" → shapes the next reply</div>
      <a class="mw-try" href="{REPO}/blob/main/evals/support_bot.py" target="_blank" rel="noopener">evals/support_bot.py →</a>
    </article>
    <article class="mw-card">
      <div class="mw-card-head"><span class="mw-icon">◈</span><h3>Shopping and recommendations</h3></div>
      <p>Tastes, sizes and budgets that make suggestions personal. Allergies are safety memories:
      they go into every prompt and are never held back.</p>
      <div class="mw-eg">Allergic to peanuts → no peanut snacks, ever</div>
    </article>
    <article class="mw-card">
      <div class="mw-card-head"><span class="mw-icon">◎</span><h3>Learning, health and coaching apps</h3></div>
      <p>Goals, level and what has worked. Sensitive details are kept only when the user asks,
      short-term plans expire on their own, and old facts get re-checked.</p>
      <div class="mw-eg">8 months on: "Still training for the marathon?"</div>
    </article>
    <article class="mw-card">
      <div class="mw-card-head"><span class="mw-icon">⟲</span><h3>AI agents and workflows</h3></div>
      <p>What was tried and how it went, so the next run doesn't repeat a mistake. These
      experiences fade unless they keep being useful.</p>
      <div class="mw-eg">"The CSV export timed out over 10k rows" → split the job next time</div>
    </article>
    <article class="mw-card">
      <div class="mw-card-head"><span class="mw-icon">◷</span><h3>Sales and CRM assistants</h3></div>
      <p>Each account's contacts, deal stage and what was promised. When a contact changes jobs
      or a deal moves on, the old fact becomes history instead of a wrong answer.</p>
      <div class="mw-eg">"Priya moved to Globex" → Acme contact closed, kept as history</div>
    </article>
  </div>
  <div class="mw-built">
    <h3>Built in, whatever the product</h3>
    <ul>
      <li><b>Any model:</b> any <code>/chat/completions</code> API, local models (Ollama, LM Studio), or your own function</li>
      <li><b>Users stay in control:</b> they can see what's stored, delete one memory, or delete everything, audit log included</li>
      <li><b>Contexts stay apart:</b> work memories stay out of personal chats, and the other way round</li>
      <li><b>Your data stays yours:</b> one SQLite file, nothing sent anywhere except to the model you chose</li>
    </ul>
  </div>
</section>
"""

CODE_TITLE = """
<div class="mw-section mw-code-title"><h2>Add it to your product</h2>
<p class="mw-sub">Three calls. Swap the messages for tickets, commits or events, and the user id for
a customer, a repository or an agent.</p></div>
"""

CODE = '''from memwise import MemoryManager, MemoryStore

mm = MemoryManager(MemoryStore("memory.db"))

# After a conversation, a support ticket, or a batch of commits: hand over what happened.
mm.ingest("user-42", messages)

# Before each model call: recall what matters, and add it to the system prompt.
recall = mm.recall("user-42", "Any snack ideas for my hike?")
system_prompt = "You are a helpful assistant.\\n\\n" + mm.build_context(recall)

# For a repository instead of a person:
#   mm.build_context(recall, about="the project")
'''

HOW = f"""
<section class="mw-section">
  <h2>How it works</h2>
  <p class="mw-sub">Every memory goes through the same pipeline, whatever product it comes from.
  Each step is a small, readable Python module.</p>
  <div class="mw-pipe">
    <div class="mw-stage"><span class="mw-phase">Write</span><h4>1 · Write gate</h4>
      <p>Drops small talk. Sensitive details need the user's consent. Secrets are never stored.</p></div>
    <div class="mw-stage"><span class="mw-phase">Write</span><h4>2 · Reconcile</h4>
      <p>Compares each new memory with what's stored: ADD, CONFIRM, SUPERSEDE or RETRACT.</p></div>
    <div class="mw-stage"><span class="mw-phase">Write</span><h4>3 · Store</h4>
      <p>SQLite rows with "true from … to …" dates. Old facts become history, not deletions.</p></div>
    <div class="mw-stage"><span class="mw-phase mw-phase-up">Upkeep</span><h4>4 · Maintain</h4>
      <p>Trips and plans expire, duplicates merge, unused experiences fade.</p></div>
    <div class="mw-stage"><span class="mw-phase mw-phase-read">Read</span><h4>5 · Retrieve</h4>
      <p>Ranks current memories by relevance, recency, importance and confidence.</p></div>
    <div class="mw-stage"><span class="mw-phase mw-phase-read">Read</span><h4>6 · Surface</h4>
      <p>Decides USE, MENTION, VERIFY or HOLD for each one. Held memories never reach the model.</p></div>
  </div>
  <div class="mw-bench">
    <h3>Measured, not just demoed</h3>
    <div class="mw-scroll"><table>
      <tr><th>Benchmark</th><th>memwise</th><th>Whole history in the prompt</th><th>Tokens saved</th></tr>
      <tr><td>LoCoMo, 382 questions</td><td><b>85.3</b></td><td>88.2</td><td><b>78%</b></td></tr>
      <tr><td>LongMemEval, 60-question sample</td><td><b>88.3</b></td><td>93.3</td><td><b>96%</b></td></tr>
    </table></div>
    <p class="mw-sub">On all 1,540 LoCoMo questions memwise scores <b>89.7</b>.
    <a href="{REPO}#benchmark-results" target="_blank" rel="noopener">How it was measured →</a></p>
  </div>
</section>
"""


def footer(version: str) -> str:
    return f"""
<footer class="mw-footer">
  <p><b>Your privacy in this demo:</b> your memory lives only in this browser tab's session, is never
  shared with other visitors, and is deleted when you leave. Messages are sent to the model
  provider to be answered, so please don't type anything private.</p>
  <p>memwise {version} · MIT license · <a href="{REPO}" target="_blank" rel="noopener">GitHub</a></p>
</footer>
"""


CSS = """
.gradio-container {max-width: 1220px !important; margin: 0 auto !important}

/* ---- hero: the pitch on the left, links on the right ---- */
.mw-hero {display:grid; grid-template-columns: minmax(0, 1fr) auto; gap: 32px;
  align-items: center; padding: 22px 26px; border-radius: 18px; margin: 4px 0 2px;
  background: radial-gradient(900px 240px at 0% 0%, rgba(99,102,241,.15), transparent 60%),
              radial-gradient(600px 220px at 100% 0%, rgba(168,85,247,.10), transparent 60%),
              var(--block-background-fill);
  border: 1px solid var(--border-color-primary)}
.mw-eyebrow {display:inline-flex; align-items:center; gap:8px; font-size:11.5px; font-weight:600;
  letter-spacing:.05em; text-transform:uppercase; color: var(--color-accent)}
.mw-live {width:8px; height:8px; border-radius:50%; background:#22c55e;
  box-shadow: 0 0 0 4px rgba(34,197,94,.18)}
.mw-hero h1 {font-size: clamp(24px, 3vw, 34px) !important; line-height:1.15; letter-spacing:-.02em;
  margin: 6px 0 8px !important; font-weight: 750}
.mw-lead {font-size: 15px; line-height: 1.55; margin: 0 0 12px; color: var(--body-text-color-subdued)}
.mw-lead b {color: var(--body-text-color)}
.mw-notjust {display:flex; gap:10px; align-items:baseline; padding: 8px 12px; border-radius: 10px;
  background: rgba(99,102,241,.08); border: 1px dashed rgba(99,102,241,.45); font-size: 13.5px;
  line-height:1.45}
.mw-notjust-tag {flex:none; font-size:10.5px; font-weight:700; text-transform:uppercase;
  letter-spacing:.04em; padding:2px 8px; border-radius:999px; background: var(--color-accent); color: white}
.mw-hero-side .mw-links {flex-direction: column; align-items: stretch; min-width: 150px}
.mw-hero-side .mw-btn {text-align: center}
.mw-links {display:flex; flex-wrap:wrap; gap:8px}
.mw-btn {display:inline-block; padding: 7px 13px; border-radius: 9px; font-weight:600; font-size:13px;
  text-decoration:none !important; border:1px solid var(--border-color-primary);
  color: var(--body-text-color) !important; background: var(--background-fill-primary)}
.mw-btn:hover {border-color: var(--color-accent)}
.mw-btn-primary {background: var(--color-accent); border-color: var(--color-accent); color:#fff !important}

/* ---- demo ---- */
.mw-steps {display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 6px 0 2px}
.mw-step {display:flex; gap:12px; align-items:center; padding: 12px 14px; border-radius: 14px;
  border: 1px solid var(--border-color-primary); background: var(--block-background-fill)}
.mw-step b {display:block; font-size:14.5px}
.mw-step small {color: var(--body-text-color-subdued); font-size:12.5px}
.mw-num {flex:none; width:28px; height:28px; border-radius:50%; display:grid; place-items:center;
  font-weight:700; font-size:14px; color:#fff; background: var(--color-accent)}
#mw-tour {font-size: 15px; padding: 12px; min-height: 48px}
.mw-side {display:flex; gap:10px; align-items:center; padding: 10px 14px; border-radius: 14px 14px 0 0;
  border: 1px solid var(--border-color-primary); border-bottom: 0}
.mw-side b {display:block; font-size:15px}
.mw-side small {color: var(--body-text-color-subdued); font-size:12.5px}
.mw-side-dot {flex:none; width:12px; height:12px; border-radius:50%}
.mw-side-plain {background: rgba(100,116,139,.10)}
.mw-side-plain .mw-side-dot {background:#94a3b8}
.mw-side-mem {background: rgba(99,102,241,.14)}
.mw-side-mem .mw-side-dot {background: var(--color-accent); box-shadow: 0 0 0 4px rgba(99,102,241,.2)}
.mw-chatcol {gap: 0 !important}
.mw-chatcol .html-container {padding: 0 !important}
/* The chats are rebuilt from the server on every message, so clearing one on screen would only confuse. */
.mw-chat button[aria-label="Clear"] {display: none !important}
.mw-chat {border: 1px solid var(--border-color-primary) !important; border-top: 0 !important;
  border-radius: 0 0 14px 14px !important; background: var(--block-background-fill) !important;
  box-shadow: 0 1px 2px rgba(15,23,42,.04)}
.mw-side-mem, .mw-chat-mem {border-color: rgba(99,102,241,.45) !important}
.mw-ph {text-align:center; color: var(--body-text-color-subdued); font-size:14px; line-height:1.5}
.mw-ph b {color: var(--body-text-color); font-size:15px}
.mw-statusbar {display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; font-size:13.5px;
  color: var(--body-text-color-subdued); padding: 4px 2px}
.mw-pill {display:inline-block; padding: 3px 10px; border-radius:999px; font-weight:700; font-size:12.5px;
  background: var(--color-accent); color:#fff}
.mw-statusbar b {color: var(--body-text-color)}
.mw-offline {color:#b45309; font-weight:600}
.mw-panel-title h3 {margin: 6px 0 2px !important; font-size: 17px !important}
.mw-panel-title p {margin: 0 0 6px !important; font-size: 13px; color: var(--body-text-color-subdued)}
.mw-panel {max-height: 520px; overflow-y: auto; padding-right: 4px}

/* ---- badges, legend, event feed ---- */
.mw-badge {display:inline-block; min-width:5.6em; text-align:center; padding:0 6px; border-radius:6px;
  font:600 11px/1.8 var(--font-mono); letter-spacing:.02em; background:rgba(120,120,120,.16);
  color:var(--body-text-color-subdued)}
.mw-ADD, .mw-USE {background:rgba(34,197,94,.16); color:#15803d}
.mw-CONFIRM, .mw-MERGE {background:rgba(59,130,246,.16); color:#1d4ed8}
.mw-SUPERSEDE, .mw-PAST {background:rgba(168,85,247,.16); color:#7e22ce}
.mw-RETRACT, .mw-VERIFY, .mw-EXPIRE {background:rgba(245,158,11,.18); color:#b45309}
.mw-MENTION {background:rgba(6,182,212,.16); color:#0e7490}
.mw-FORGET, .mw-PRUNE {background:rgba(239,68,68,.16); color:#b91c1c}
.mw-PROFILE {background:rgba(99,102,241,.16); color:#4338ca}
.dark .mw-ADD, .dark .mw-USE {color:#4ade80}
.dark .mw-CONFIRM, .dark .mw-MERGE {color:#93c5fd}
.dark .mw-SUPERSEDE, .dark .mw-PAST {color:#d8b4fe}
.dark .mw-RETRACT, .dark .mw-VERIFY, .dark .mw-EXPIRE {color:#fcd34d}
.dark .mw-MENTION {color:#67e8f9}
.dark .mw-FORGET, .dark .mw-PRUNE {color:#fca5a5}
.dark .mw-PROFILE {color:#a5b4fc}
.mw-legend {display:flex; flex-direction:column; gap:6px; padding: 10px 12px; border-radius:12px;
  border:1px solid var(--border-color-primary); background: var(--block-background-fill); margin-bottom:6px}
.mw-legend > div {display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center}
.mw-legend-title {font-size:12px; font-weight:700; min-width:108px; color: var(--body-text-color-subdued)}
.mw-chip {display:inline-flex; gap:5px; align-items:center; font-size:12.5px; color: var(--body-text-color-subdued)}
.mw-chip .mw-badge {min-width:0}
.mw-block {margin-bottom: 12px; padding: 10px 12px; border-radius: 12px;
  border:1px solid var(--border-color-primary); border-left: 4px solid #94a3b8;
  background: var(--block-background-fill)}
.mw-block-write {border-left-color: #22c55e}
.mw-block-recall {border-left-color: var(--color-accent)}
.mw-block-session {border-left-color: #f59e0b}
.mw-title {font-weight:650; font-size:13.5px; margin-bottom:6px}
.mw-table {width:100%; border-collapse:collapse; font-size:13px; margin:0; border:0 !important}
.mw-table tr {border:0 !important; background:transparent !important}
.mw-table td {padding:5px 6px; border:0 !important; border-top:1px solid var(--border-color-primary) !important;
  vertical-align:top; background:transparent !important}
.mw-table tr:first-child td {border-top:0 !important}
.mw-table td:first-child {width:1%; white-space:nowrap}
.mw-why {color:var(--body-text-color-subdued); font-size:12px; margin-top:1px}
.mw-empty {color:var(--body-text-color-subdued); font-size:13.5px; padding: 8px 2px}
.mw-kind {display:inline-block; font-size:11px; padding:0 6px; border-radius:6px; margin-right:4px;
  background: rgba(120,120,120,.14); color: var(--body-text-color-subdued)}
details.mw-history summary {cursor:pointer; font-size:13px; margin:12px 0 4px; font-weight:600}

/* ---- use cases, how it works ---- */
.mw-section {padding: 8px 4px 4px}
.mw-section a {color: var(--color-accent)}
.mw-section h2 {font-size: 26px !important; letter-spacing:-.01em; margin: 10px 0 6px !important}
.mw-sub {color: var(--body-text-color-subdued); font-size: 15px; line-height:1.6; max-width: 860px;
  margin: 0 0 18px}
.mw-cards {display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px}
.mw-card {padding: 18px; border-radius: 16px; border:1px solid var(--border-color-primary);
  background: var(--block-background-fill); display:flex; flex-direction:column; gap:10px}
.mw-card-feature {grid-column: span 3; background:
  linear-gradient(135deg, rgba(99,102,241,.10), rgba(168,85,247,.06)), var(--block-background-fill);
  border-color: rgba(99,102,241,.4)}
.mw-card-head {display:flex; gap:10px; align-items:center}
.mw-card h3 {margin:0 !important; font-size: 16.5px !important}
.mw-card p {margin:0 !important; font-size: 14px; line-height:1.55; color: var(--body-text-color-subdued)}
.mw-icon {flex:none; width:34px; height:34px; border-radius:10px; display:grid; place-items:center;
  font-size:17px; background: rgba(99,102,241,.14); color: var(--color-accent)}
.mw-eg {font-size:13px; padding: 8px 10px; border-radius:10px; background: var(--background-fill-secondary);
  font-family: var(--font-mono)}
.mw-flow {display:flex; flex-direction:column; gap:6px; font-size:13.5px}
.mw-flow > div {display:flex; gap:10px; align-items:baseline; padding: 7px 10px; border-radius:10px;
  background: var(--background-fill-primary); border:1px solid var(--border-color-primary)}
.mw-k {flex:none; width:96px; font-size:11px; font-weight:700; text-transform:uppercase;
  letter-spacing:.04em; color: var(--color-accent)}
.mw-try {font-size:13px; font-weight:600; color: var(--color-accent) !important; margin-top:auto;
  text-decoration:none !important}
.mw-built {margin-top: 18px; padding: 16px 18px; border-radius: 16px; border:1px solid var(--border-color-primary)}
.mw-built h3 {margin: 0 0 8px !important; font-size: 16px !important}
.mw-built ul {margin:0; padding-left: 18px; display:grid; grid-template-columns: 1fr 1fr; gap: 6px 24px;
  font-size: 14px; line-height:1.5}
.mw-code-title {margin-top: 10px}
.mw-pipe {display:grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px}
.mw-stage {position:relative; padding: 14px; border-radius: 14px; border:1px solid var(--border-color-primary);
  background: var(--block-background-fill)}
.mw-stage h4 {margin: 8px 0 6px !important; font-size: 14.5px !important}
.mw-stage p {margin:0 !important; font-size: 13px; line-height:1.5; color: var(--body-text-color-subdued)}
.mw-phase {font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; padding:2px 7px;
  border-radius:999px; background: rgba(34,197,94,.16); color:#15803d}
.mw-phase-up {background: rgba(245,158,11,.18); color:#b45309}
.mw-phase-read {background: rgba(99,102,241,.16); color:#4338ca}
.dark .mw-phase {color:#4ade80} .dark .mw-phase-up {color:#fcd34d} .dark .mw-phase-read {color:#a5b4fc}
.mw-bench {margin-top: 20px}
/* Wide content scrolls inside its own box instead of widening the page on a phone. */
.mw-scroll, .mw-code {contain: inline-size; width: 100%; overflow-x: auto}
.mw-bench h3 {font-size: 18px !important; margin: 0 0 8px !important}
.mw-bench table {width:100%; border-collapse: collapse; font-size: 14px; margin-bottom: 8px}
.mw-bench th, .mw-bench td {text-align:left; padding: 8px 10px; border-bottom:1px solid var(--border-color-primary)}
.mw-bench th {font-size:12.5px; color: var(--body-text-color-subdued); font-weight:600}
.mw-footer {margin-top: 18px; padding: 14px 4px; font-size: 13px; color: var(--body-text-color-subdued);
  border-top: 1px solid var(--border-color-primary)}
.mw-footer p {margin: 0 0 6px !important}

@media (max-width: 900px) {
  .mw-hero {grid-template-columns: minmax(0, 1fr); gap: 16px}
  .mw-hero-side .mw-links {flex-direction: row; flex-wrap: wrap; min-width: 0}
  .mw-cards {grid-template-columns: minmax(0, 1fr)}
  .mw-card-feature {grid-column: span 1}
  .mw-pipe {grid-template-columns: repeat(2, minmax(0, 1fr))}
  .mw-built ul {grid-template-columns: 1fr}
}
@media (max-width: 640px) {
  .main {padding-left: 12px !important; padding-right: 12px !important}
  .mw-hero {padding: 18px 16px}
  .mw-notjust {flex-direction: column; gap: 6px}
  .mw-steps {grid-template-columns: minmax(0, 1fr)}
  .mw-k {width: auto}
  .mw-flow > div {flex-direction: column; gap: 2px}
}
"""
