"""Turn benchmark summaries into one results page (a single HTML file, no dependencies).

    python evals/report.py evals/data/locomo-results/*.summary.json --out ~/Desktop/memwise-benchmarks.html

Options for the "What it saves" card: --price (an example price per million input tokens, to
turn tokens into money), --note (a line of your own, such as the cost of building the memory) and
--reference "LoCoMo|Mem0|92.5|6956|https://mem0.ai/research" (a result someone else published,
shown apart, as not a controlled comparison).

Every benchmark run (evals/locomo.py) saves a .summary.json next to its answers. Pass the ones to
show. For each benchmark, the first memwise set-up given is the headline. Others are listed
below it, and a run made with --baseline gives the full-context comparison: the whole
conversation in the prompt, on the same questions. Benchmarks without a run yet get a tab
marked "not run yet".
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

BENCHMARKS = {
    "LoCoMo": "Months of chats between two friends, then questions that need single facts (single hop), "
              "several facts together (multi-hop), inference beyond the chats (open-domain) and "
              "dates (temporal).",
    "LongMemEval": "Long chat histories with questions on recall, multi-session reasoning, "
                   "knowledge updates, temporal reasoning and abstention.",
}
ORDER = ["single-hop", "multi-hop", "open-domain", "temporal", "adversarial"]
NAMES = {"single-hop": "Single hop", "multi-hop": "Multi-hop", "open-domain": "Open-domain",
         "temporal": "Temporal", "adversarial": "Adversarial"}


def e(text: Any) -> str:
    return html.escape(str(text))


def tokens(n: float) -> str:
    return f"{n:,.0f}"


def same_questions(a: dict, b: dict) -> bool:
    return a["conversations"] == b["conversations"] and a["questions"] == b["questions"]


def collect(paths: list[Path]) -> dict[str, dict]:
    """Per benchmark: the headline set-up, the other set-ups and the full-context run."""
    out: dict[str, dict] = {}
    for p in paths:
        s = json.loads(p.read_text())
        b = out.setdefault(s["benchmark"], {"setups": [], "full": None})
        if "memwise" in s["modes"]:
            b["setups"].append(s)
        if "full" in s["modes"] and b["full"] is None:
            b["full"] = s
    for name, b in out.items():
        if not b["setups"]:
            sys.exit(f"{name}: no memwise run among the summaries given")
        head = b["setups"][0]
        if b["full"] and not same_questions(b["full"], head):
            print(f"note: {name}: the full-context run used different questions than "
                  f"{head['label']}; it is shown with its own question count", file=sys.stderr)
        b["setups"] = [s for s in b["setups"] if same_questions(s, head)] or [head]
    return out


# =========================================================================== page parts

def stat(value: str, unit: str, label: str, note: str = "", hero: bool = False) -> str:
    return (f'<div class="tile"><div class="value{" hero" if hero else ""}">{e(value)}'
            f'<span class="unit">{e(unit)}</span></div><div class="label">{e(label)}</div>'
            + (f'<div class="note">{e(note)}</div>' if note else "") + "</div>")


def split_label(label: str) -> tuple[str, str]:
    """"memwise (friends prompt, past recall)" -> ("memwise", "friends prompt, past recall")."""
    if label.endswith(")") and " (" in label:
        main, detail = label[:-1].split(" (", 1)
        return main, detail
    return label, ""


def name_html(label: str) -> str:
    main, detail = split_label(label)
    return e(main) + (f"<small>{e(detail)}</small>" if detail else "")


def bars(rows: list[tuple[str, float, str, str, str]], top: float, title: str, legend: str = "",
         grouped: bool = False) -> str:
    """Horizontal bars: (name, value, value as text, kind 'memwise'|'full', tooltip). With
    `grouped`, a row with no name belongs to the named row above it."""
    items = []
    for name, value, shown, kind, tip in rows:
        width = 0 if top <= 0 else max(0.6, 100 * value / top)
        starts = " starts" if grouped and name else ""
        items.append(
            f'<div class="row{starts}" tabindex="0" data-tip="{e(tip)}"><div class="name">{name_html(name)}</div>'
            f'<div class="track"><div class="bar {kind}" style="width:{width:.2f}%"></div></div>'
            f'<div class="num">{e(shown)}</div></div>')
    return (f'<div class="chart{" grouped" if grouped else ""}"><div class="chart-head"><h3>{e(title)}</h3>'
            f'{legend}</div>' + "".join(items) + "</div>")


LEGEND = ('<div class="legend"><span><i class="key memwise"></i>memwise</span>'
          '<span><i class="key full"></i>Full context</span></div>')


def coverage(s: dict) -> str:
    convs, total = len(s["conversations"]), s.get("total_conversations", len(s["conversations"]))
    which = "all conversations" if convs == total else f"{convs} of {total} conversations"
    return f"{s['questions']:,} questions · {which}"


def dollars(x: float) -> str:
    return f"${x:,.2f}"


def savings(versus: dict, full: dict | None, price: float, refs: list[list[str]], notes: list[str]) -> str:
    """What memwise saves against full context, in tokens and money, and published results."""
    parts = []
    if full:
        saved = 1 - versus["mean_tokens"] / full["mean_tokens"]
        per_k = [full["mean_tokens"] * 1000 * price / 1e6, versus["mean_tokens"] * 1000 * price / 1e6]
        parts.append(
            f'<p><b>{saved:.0%} fewer tokens</b> a question than sending the whole conversation: '
            f'{tokens(versus["mean_tokens"])} against {tokens(full["mean_tokens"])}, on the same '
            f'{full["n"]} questions, answered by the same model.</p>'
            f'<p>Models bill input by the token, so the memory part of every call costs {saved:.0%} '
            f'less. At an example price of {dollars(price)} per million input tokens, 1,000 questions '
            f'cost <b>{dollars(per_k[1])}</b> of context with memwise and {dollars(per_k[0])} with the '
            f'whole conversation. Use your model\'s price: the saving is the same share.</p>')
    for line in notes:
        parts.append(f"<p>{e(line)}</p>")
    for _, who, score, toks, url in refs:
        fewer = 1 - versus["tokens_all"] / float(toks)  # on every question, like the published one
        gap = float(score) - versus["score_all"]
        parts.append(
            f'<p class="ref">Published for reference: <a href="{e(url)}">{e(who)} reports {e(score)}</a> with '
            f'{tokens(float(toks))} tokens a query. memwise uses {abs(fewer):.0%} {"fewer" if fewer >= 0 else "more"} '
            f'tokens and scores {abs(gap):.1f} points {"lower" if gap >= 0 else "higher"}. {e(who)} used its own '
            f'models and grader, so this is not a controlled comparison.</p>')
    if not parts:
        return ""
    return '<div class="chart say"><div class="chart-head"><h3>What it saves</h3></div>' + "".join(parts) + "</div>"


def panel(name: str, b: dict, price: float = 1.0, refs: list[list[str]] | None = None,
          notes: list[str] | None = None) -> str:
    head = b["setups"][0]
    mw = head["modes"]["memwise"]
    full = b["full"]["modes"]["full"] if b["full"] else None
    cats = [c for c in ORDER if c in mw["categories"]]
    # Full context on fewer questions (a sample, to save tokens): compare it with memwise on
    # those same questions, from the same run, never with memwise's score on all of them.
    paired = (b["full"]["modes"].get("memwise") if full and not same_questions(b["full"], head) else None)
    versus = paired or mw

    main, detail = split_label(head["label"])
    tiles = [stat(f"{mw['score']:.1f}", "", f"{main} score",
                  (detail + " · " if detail else "") + "answers graded correct, out of 100", hero=True),
             stat(tokens(mw["mean_tokens"]), " tokens", "Mean context per question",
                  (f"about {mw['mean_memories']:.0f} memories"
                   + (" and messages" if head["settings"].get("keep_messages") else ""))
                  if "mean_memories" in mw else "")]
    if full:
        times = full["mean_tokens"] / max(1, versus["mean_tokens"])
        note = f"{tokens(full['mean_tokens'])} tokens a question, {times:.1f}× memwise"
        if paired:
            note += f" · on {full['n']} of the questions, where memwise scored {paired['score']:.1f}"
        tiles.append(stat(f"{full['score']:.1f}", "",
                          f"Full-context score, {full['n']:,} questions" if paired else "Full-context score", note))
        saved = 1 - versus["mean_tokens"] / full["mean_tokens"]
        tiles.append(stat(f"{100 * saved:.0f}", "%", "Fewer tokens than full context",
                          f"{tokens(versus['mean_tokens'])} against {tokens(full['mean_tokens'])} a question, "
                          "on the same questions"))

    setups = [(s["label"], s["modes"]["memwise"]) for s in b["setups"]]
    score_rows = [(lbl, m["score"], f"{m['score']:.1f}", "memwise",
                   f"{lbl}: {m['score']:.1f} on {m['n']} questions") for lbl, m in setups]
    token_rows = [(lbl, m["mean_tokens"], tokens(m["mean_tokens"]), "memwise",
                   f"{lbl}: {tokens(m['mean_tokens'])} tokens of context a question") for lbl, m in setups]
    if paired:
        lbl = f"memwise (same {full['n']} questions as full context)"
        score_rows.append((lbl, paired["score"], f"{paired['score']:.1f}", "memwise",
                           f"{head['label']}, on the {full['n']} questions full context answered: "
                           f"{paired['score']:.1f}"))
        token_rows.append((lbl, paired["mean_tokens"], tokens(paired["mean_tokens"]), "memwise",
                           f"memwise on the same {full['n']} questions: {tokens(paired['mean_tokens'])} tokens a question"))
    if full:
        score_rows.append((f"Full context ({full['n']} questions)" if paired else "Full context",
                           full["score"], f"{full['score']:.1f}", "full",
                           f"Whole conversation in the prompt: {full['score']:.1f} on {full['n']} questions"))
        token_rows.append((f"Full context ({full['n']} questions)" if paired else "Full context",
                           full["mean_tokens"], tokens(full["mean_tokens"]), "full",
                           f"Whole conversation in the prompt: {tokens(full['mean_tokens'])} tokens a question"))

    cat_rows = []
    for c in cats:
        m = versus["categories"][c]
        cat_rows.append((NAMES[c], m["score"], f"{m['score']:.1f}", "memwise",
                         f"memwise, {NAMES[c].lower()}: {m['score']:.1f} on {m['n']} questions"))
        if full and c in full["categories"]:
            f = full["categories"][c]
            cat_rows.append(("", f["score"], f"{f['score']:.1f}", "full",
                             f"Full context, {NAMES[c].lower()}: {f['score']:.1f} on {f['n']} questions"))

    table_rows = "".join(
        f"<tr><td>{e(lbl)}</td><td>{m['score']:.1f}</td>"
        + "".join(f"<td>{m['categories'][c]['score']:.1f}</td>" if c in m["categories"] else "<td>–</td>"
                  for c in cats)
        + f"<td>{tokens(m['mean_tokens'])}</td><td>{m['n']}</td></tr>"
        for lbl, m in setups + ([(f"memwise (same {full['n']} questions as full context)", paired)] if paired else [])
        + ([(f"Full context ({full['n']} questions)" if paired else "Full context", full)] if full else []))
    table = ('<details><summary>Table view</summary><table><thead><tr><th>Set-up</th><th>Score</th>'
             + "".join(f"<th>{e(NAMES[c])}</th>" for c in cats)
             + "<th>Mean tokens</th><th>Questions</th></tr></thead><tbody>" + table_rows
             + "</tbody></table></details>")

    st = head["settings"]
    facts = [f"Answers and grading: {st['chat_model']} ({st['grader']} grading, as in published results)"
             if st["grader"] == "lenient" else f"Answers and grading: {st['chat_model']} ({st['grader']} grading)",
             f"Embeddings: {st['embed_model']}",
             f"Headline set-up: extraction prompt {st['prompt']}, sensitive details {st['sensitive']}, "
             f"past recall {'on' if st['past'] else 'off'}, {st['top_k']} memories per person"
             + (f", messages kept and {st['message_top_k']} recalled per person" if st.get("keep_messages") else ""),
             "memwise and full context answer with the same model and prompt; only the context differs.",
             f"Run on {head['date']}. Score = share of answers graded correct (categories 1–4)."]
    if paired:
        facts.append(f"Full context was run on a sample to save tokens: {coverage(b['full'])} (up to 10 of "
                     "each question type per conversation). It is compared with memwise on those same questions.")

    saves = savings(dict(versus, score_all=mw["score"], tokens_all=mw["mean_tokens"]), full, price,
                    [r for r in (refs or []) if r[0] == name], notes or [])
    cards = (f'<div class="tiles">{"".join(tiles)}</div>{saves}'
             f'<div class="pair">{bars(score_rows, 100, "Score, out of 100", LEGEND if full else "")}'
             f'{bars(token_rows, max(r[1] for r in token_rows), "Mean context tokens per question")}</div>'
             + bars(cat_rows, 100, "Score by question type" + (f", on the {full['n']} questions both answered"
                                                               if paired else ""), LEGEND if full else "", grouped=True)
             + table + '<ul class="facts">' + "".join(f"<li>{e(f)}</li>" for f in facts) + "</ul>")
    return (f'<section class="panel" id="p-{e(name)}"><h2>{e(name)}</h2>'
            f'<p class="meta">{e(coverage(head))} · {len(cats)} categories</p>'
            f'<p class="desc">{e(BENCHMARKS.get(name, ""))}</p>{cards}</section>')


def empty_panel(name: str) -> str:
    return (f'<section class="panel" id="p-{e(name)}" hidden><h2>{e(name)}</h2>'
            f'<p class="desc">{e(BENCHMARKS.get(name, ""))}</p>'
            '<p class="empty">Not run yet.</p></section>')


CSS = """
:root { color-scheme: dark; --page:#0b1120; --surface:#111827; --line:#1f2937; --ink:#f1f5f9;
  --ink-2:#cbd5e1; --muted:#8b94a7; --memwise:#13a898; --full:#646d80; --accent:#5eead4; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--page); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
  padding:40px 20px 64px; }
main { max-width:920px; margin:0 auto; }
header { text-align:center; margin-bottom:28px; }
header .brand { font-size:13px; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); font-weight:600; }
header h1 { font-size:34px; font-weight:700; letter-spacing:-.02em; margin-top:6px; }
nav { display:flex; justify-content:center; gap:4px; background:var(--surface); border:1px solid var(--line);
  border-radius:999px; padding:5px; width:max-content; margin:22px auto 0; }
nav button { font:inherit; font-size:14px; color:var(--ink-2); background:none; border:0; border-radius:999px;
  padding:7px 16px; cursor:pointer; }
nav button[aria-selected="true"] { background:var(--memwise); color:#fff; font-weight:600; }
nav button:disabled { color:var(--muted); cursor:default; }
nav button small { font-size:11px; margin-left:6px; opacity:.8; }
.panel h2 { text-align:center; font-size:40px; font-weight:700; letter-spacing:-.02em; }
.meta { text-align:center; color:var(--accent); font:13px ui-monospace,SFMono-Regular,Menlo,monospace; margin-top:6px; }
.desc { text-align:center; color:var(--ink-2); max-width:620px; margin:8px auto 26px; font-size:14px; }
.empty { text-align:center; color:var(--muted); padding:48px 0; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; }
.tile { background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:22px 20px; text-align:center; }
.tile .value { font-size:40px; font-weight:650; letter-spacing:-.02em; }
.tile .value.hero { font-size:56px; }
.tile .unit { font-size:16px; font-weight:500; color:var(--ink-2); margin-left:2px; }
.tile .label { color:var(--ink); font-weight:600; margin-top:2px; }
.tile .note { color:var(--muted); font-size:13px; margin-top:2px; }
.pair { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; margin-top:14px; }
.chart { background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:18px 20px 12px; margin-top:14px; }
.pair .chart { margin-top:0; }
.chart-head { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:10px; flex-wrap:wrap; }
.chart h3 { font-size:13px; font-weight:600; color:var(--ink-2); letter-spacing:.06em; text-transform:uppercase; }
.legend { display:flex; gap:14px; font-size:13px; color:var(--ink-2); }
.key { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.row { display:grid; grid-template-columns:minmax(90px,38%) 1fr 64px; align-items:center; gap:12px; padding:5px 0;
  border-radius:8px; outline:none; }
.row:hover .track, .row:focus-visible .track { background:#1a2333; }
.row .name { color:var(--ink); font-size:14px; line-height:1.3; }
.row .name small { display:block; color:var(--muted); font-size:12px; }
.grouped .row { grid-template-columns:110px 1fr 64px; padding:3px 0; }
.grouped .row.starts { margin-top:10px; }
.grouped .row.starts:first-of-type { margin-top:0; }
.track { height:18px; border-radius:4px; }
.bar { height:18px; border-radius:0 4px 4px 0; }
.bar.memwise, .key.memwise { background:var(--memwise); }
.bar.full, .key.full { background:var(--full); }
.row .num { text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); font-size:14px; }
details { margin-top:14px; background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:12px 20px; }
summary { cursor:pointer; color:var(--ink-2); font-size:14px; }
table { width:100%; border-collapse:collapse; margin-top:10px; font-size:13px; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:6px 8px; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; }
.say p { color:var(--ink-2); font-size:14px; margin-top:8px; text-align:left; }
.say p:first-of-type { margin-top:0; }
.say b { color:var(--ink); }
.say a { color:var(--accent); }
.say .ref { color:var(--muted); font-size:13px; }
.facts { list-style:none; margin-top:16px; color:var(--muted); font-size:13px; text-align:center; }
.facts li + li { margin-top:2px; }
#tip { position:fixed; pointer-events:none; background:#0b1120; border:1px solid #334155; color:var(--ink);
  font-size:13px; padding:6px 10px; border-radius:8px; opacity:0; transition:opacity .1s; max-width:320px; }
"""

JS = """
const tabs = document.querySelectorAll('nav button');
tabs.forEach(b => b.addEventListener('click', () => {
  tabs.forEach(t => t.setAttribute('aria-selected', t === b));
  document.querySelectorAll('.panel').forEach(p => p.hidden = p.id !== 'p-' + b.dataset.name);
}));
const tip = document.getElementById('tip');
function show(el, x, y) { tip.textContent = el.dataset.tip; tip.style.left = (x + 14) + 'px';
  tip.style.top = (y + 14) + 'px'; tip.style.opacity = 1; }
document.querySelectorAll('.row[data-tip]').forEach(r => {
  r.addEventListener('mousemove', ev => show(r, ev.clientX, ev.clientY));
  r.addEventListener('mouseleave', () => tip.style.opacity = 0);
  r.addEventListener('focus', () => { const b = r.getBoundingClientRect(); show(r, b.left + b.width / 2, b.bottom); });
  r.addEventListener('blur', () => tip.style.opacity = 0);
});
"""


def page(data: dict[str, dict], price: float = 1.0, refs: list[list[str]] | None = None,
         notes: list[str] | None = None) -> str:
    names = list(dict.fromkeys([*data, *BENCHMARKS]))
    first = next(n for n in names if n in data)
    buttons = "".join(
        f'<button role="tab" data-name="{e(n)}" aria-selected="{str(n == first).lower()}">{e(n)}</button>'
        if n in data else f'<button role="tab" disabled>{e(n)}<small>not run yet</small></button>'
        for n in names)
    panels = "".join(panel(n, data[n], price, refs, notes).replace('class="panel"', 'class="panel"' + ("" if n == first else " hidden"), 1)
                     if n in data else empty_panel(n) for n in names)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>memwise benchmarks</title><style>{CSS}</style></head><body><main>"
            f'<header><div class="brand">memwise</div><h1>Benchmarks</h1><nav role="tablist">{buttons}</nav></header>'
            f'{panels}</main><div id="tip" role="tooltip"></div><script>{JS}</script></body></html>')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("summaries", nargs="+", type=Path, help=".summary.json files; the first memwise one leads")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "benchmarks.html",
                        help="where to write the page (default: evals/data/benchmarks.html)")
    parser.add_argument("--price", type=float, default=1.0,
                        help="an example price per million input tokens, in dollars (default 1)")
    parser.add_argument("--note", action="append", default=[], help="a line of your own for 'What it saves'")
    parser.add_argument("--reference", action="append", default=[], metavar="BENCHMARK|NAME|SCORE|TOKENS|URL",
                        help="a published result, shown apart as not a controlled comparison")
    args = parser.parse_args()
    refs = [r.split("|") for r in args.reference]
    if any(len(r) != 5 for r in refs):
        sys.exit("--reference takes BENCHMARK|NAME|SCORE|TOKENS|URL")
    out = args.out.expanduser()
    out.write_text(page(collect(args.summaries), args.price, refs, args.note))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
