"""memwise: long-term memory for AI products that decides what to keep,
notices when facts stop being true, and checks whether to speak up.

Quick start:

    from memwise import MemoryManager, MemoryStore

    mm = MemoryManager(MemoryStore("memory.db"))
    mm.ingest("user-1", ["I live in Berlin.", "Remember that I'm allergic to peanuts."])

    recall = mm.recall("user-1", "Any snack ideas?")
    system_prompt = "You are a helpful assistant.\\n\\n" + mm.build_context(recall)

In production, connect any model provider (see memwise.llm):

    from memwise import LLMExtractor, MemoryConfig, OpenAICompatible, OpenAICompatibleEmbedder

    llm = OpenAICompatible(base_url="...", model="...", api_key="...")
    embedder = OpenAICompatibleEmbedder(base_url="...", model="...", api_key="...")
    mm = MemoryManager(MemoryStore("memory.db"), extractor=LLMExtractor(llm),
                       embedder=embedder, config=MemoryConfig.for_embedding_model())

The pipeline, in the order a memory travels through it:

    extractor   -> candidates ("this might be worth remembering")
    reconciler  -> ADD / CONFIRM / SUPERSEDE / RETRACT / IGNORE against what's stored
    store       -> SQLite rows with validity windows, never silently overwritten
    maintenance -> expires situational facts, merges duplicates, prunes unused episodes
    retriever   -> ranks current memories (relevance + recency + importance + confidence)
    surfacer    -> USE / MENTION / VERIFY / HOLD for each retrieved memory
    manager     -> ties it together and builds the context block the LLM sees
"""

from .clock import Clock
from .config import MemoryConfig
from .embed import Embedder, HashingEmbedder
from .extractor import Extractor, RuleExtractor
from .llm import (LLM, LLMError, LLMExtractor, LLMJudge, LLMQueryRewriter, OpenAICompatible,
                  OpenAICompatibleEmbedder)
from .manager import MemoryManager, Recall
from .models import Candidate, Memory, SavedMessage
from .reconciler import Decision, Judge, JudgeRequest, Verdict
from .retriever import Rewriter, ScoredMessage
from .store import MemoryStore
from .surfacer import SurfaceDecision

__version__ = "0.5.0"

__all__ = [
    "Candidate",
    "Clock",
    "Decision",
    "Embedder",
    "Extractor",
    "HashingEmbedder",
    "Judge",
    "JudgeRequest",
    "LLM",
    "LLMError",
    "LLMExtractor",
    "LLMJudge",
    "LLMQueryRewriter",
    "Memory",
    "MemoryConfig",
    "MemoryManager",
    "MemoryStore",
    "OpenAICompatible",
    "OpenAICompatibleEmbedder",
    "Recall",
    "Rewriter",
    "RuleExtractor",
    "SavedMessage",
    "ScoredMessage",
    "SurfaceDecision",
    "Verdict",
    "__version__",
]
