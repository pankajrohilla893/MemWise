---
title: MemWiseDemo
emoji: 🏆
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: A live demo web app. Two chats side by side
---

# memwise live demo

The same model in two chats side by side: on the left on its own, on the right with
[memwise](https://github.com/pankajrohilla893/MemWise), a long-term memory library for AI
products. Tell it about yourself, start a new session (and skip ahead in time), and see what each
one remembers. Panels show what memwise stored or ignored, what it recalled for each message
(USE, MENTION, VERIFY or HOLD, with the reason), and the exact memory it added to the prompt.

memwise isn't only for chatbots: the page's **Use it in any product** tab shows it in coding
agents (project memory from git commits), customer support, shopping, tutoring apps and AI
agents, and **How it works** walks through the pipeline and the benchmark results.

Each visitor gets their own memory, kept in RAM and deleted when they leave.

## Run it yourself

From a clone of the repository:

```bash
pip install gradio
OPENAI_API_KEY=... python demo/app.py
```

With only `OPENAI_API_KEY` set, it uses OpenAI with `gpt-6-luna` and `text-embedding-3-small`,
the models the benchmarks used. `MEMWISE_MODEL` and `MEMWISE_EMBED_MODEL` pick other models;
`MEMWISE_BASE_URL` (with `MEMWISE_API_KEY`) points it at any other `/chat/completions` API. With
no key at all, the demo runs offline: memwise uses its rule-based extractor and the chats say
there's no model to answer.

## Deploy to a Hugging Face Space

`.github/workflows/deploy-demo.yml` copies this folder and the `memwise` package to the Space
[pankajrohilla899/MemWiseDemo](https://huggingface.co/spaces/pankajrohilla899/MemWiseDemo) on
every push to `main` that changes this folder. A change to the `memwise` package alone doesn't
deploy: it goes out with the next change here, or run the workflow by hand from the **Actions**
tab. To set it up:

1. On Hugging Face, create a Space with the **Gradio** SDK and the **Blank** template.
2. In the Space's **Settings → Variables and secrets**, add a **secret** named `OPENAI_API_KEY`
   (a secret, not a variable: variables are visible to anyone who views the Space).
3. On Hugging Face, create an access token with write access
   (**Settings → Access Tokens**).
4. In the GitHub repository's **Settings → Secrets and variables → Actions**, add that token as
   the secret `HF_TOKEN`. To deploy to a different Space, also add a variable `HF_SPACE` with its
   name, like `your-name/memwise`.
5. Run the workflow from the **Actions** tab (or push to `main`).

To deploy from your own machine instead, after `hf auth login`:

```bash
rm -rf /tmp/space && mkdir /tmp/space && cp -r demo/. memwise /tmp/space/
hf upload pankajrohilla899/MemWiseDemo /tmp/space . --repo-type space --exclude "*__pycache__*" --exclude "*.ruff_cache*"
```

Model calls cost money, so the demo limits them. Set these as Space variables to change the
limits:

| Variable | What it limits | Default |
|---|---|---|
| `DEMO_VISITOR_LIMIT` | messages and new sessions per visitor | 40 |
| `DEMO_DAILY_LIMIT` | messages and new sessions per day, all visitors together | 1500 |
| `DEMO_REPLY_OPTIONS` | extra fields for the chat replies, as JSON, e.g. `{"max_completion_tokens": 600}` | `{"reasoning_effort": "none"}` with the default model, so replies start in under a second |
