"""The whole API in one page. Runs offline, no API key needed:

    python examples/quickstart.py
"""

import os
import sys

# Lets you run this from a clone without installing: `python examples/quickstart.py`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from memwise import MemoryManager, MemoryStore

# 1. Create a manager. In production: MemoryStore("memory.db") and an LLMExtractor.
mm = MemoryManager(MemoryStore(":memory:"))
user = "user-42"

# 2. WRITE: after a conversation, hand over what the user said.
decisions = mm.ingest(user, [
    "Hi, my name is Ana and I live in Berlin.",
    "I prefer short answers.",
    "Remember that I'm allergic to peanuts.",
    "I was diagnosed with asthma last year.",       # sensitive, not explicit -> not stored
])
for d in decisions:
    print(f"{d.op:<9} {d.candidate.text:<38} {d.reason}")

# 3. Facts change: the old one is closed, not overwritten.
for d in mm.ingest(user, ["I just moved to Lisbon!"]):
    print(f"{d.op:<9} {d.candidate.text:<38} {d.reason}")

# 4. READ: before each model call, recall and build the memory section of your prompt.
recall = mm.recall(user, "Any snack ideas for this weekend?")
system_prompt = "You are a friendly assistant.\n\n" + mm.build_context(recall)
print("\n--- system prompt your model would receive ---")
print(system_prompt)

# 5. What the user can see and control.
print("\n--- what we know about the user ---")
for m in mm.current(user):
    print(f"{m.id:<4} {m.text}")

allergy = next(m for m in mm.current(user) if m.predicate == "allergy")
mm.forget(user, allergy.id)          # hard delete + scrubbed from the audit log
print(f"\nforgot {allergy.id}; {len(mm.current(user))} memories left")

# 6. UPKEEP: run this nightly for every user (expire, merge, prune).
mm.maintain_all()
