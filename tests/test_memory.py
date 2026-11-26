"""Run with:  python -m unittest discover -s tests -v

Each test pins down one memory behaviour. When you change a threshold or a rule,
these tell you which promise you just broke. They double as an eval starter:
real products grow this into multi-session scripts with planted facts.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memwise import Candidate, Clock, MemoryConfig, MemoryManager, MemoryStore  # noqa: E402
from memwise.llm import (JUDGE_SYSTEM, LLMError, LLMExtractor, LLMJudge,  # noqa: E402
                         OpenAICompatible, OpenAICompatibleEmbedder, parse_memories,
                         parse_verdicts)

U = "u1"
_STORES: list[MemoryStore] = []


def store(path: str = ":memory:") -> MemoryStore:
    s = MemoryStore(path)
    _STORES.append(s)
    return s


def tearDownModule():
    for s in _STORES:
        s.close()


def manager() -> MemoryManager:
    clock = Clock(fixed_start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    return MemoryManager(store(), clock=clock)


def say(mm: MemoryManager, *texts: str):
    return mm.ingest(U, [(f"t{i}", t) for i, t in enumerate(texts)])


def current_values(mm: MemoryManager, predicate: str) -> list[str]:
    return [m.value for m in mm.current(U) if m.predicate == predicate]


class WritePath(unittest.TestCase):
    def test_small_talk_stores_nothing(self):
        mm = manager()
        self.assertEqual(say(mm, "lol the weather is terrible today"), [])
        self.assertEqual(mm.current(U), [])

    def test_new_value_supersedes_old_and_keeps_history(self):
        mm = manager()
        say(mm, "I live in Berlin.")
        mm.clock.advance(10)
        [d] = say(mm, "I just moved to Lisbon!")
        self.assertEqual(d.op, "SUPERSEDE")
        self.assertEqual(current_values(mm, "lives_in"), ["Lisbon"])
        berlin = next(m for m in mm.history(U) if m.value == "Berlin")
        self.assertIsNotNone(berlin.valid_to)
        self.assertEqual(berlin.superseded_by, d.result.id)

    def test_restating_confirms_instead_of_duplicating(self):
        mm = manager()
        say(mm, "My name is Sam.")
        mm.clock.advance(30)
        [d] = say(mm, "My name is Sam, by the way.")
        self.assertEqual(d.op, "CONFIRM")
        self.assertEqual(len(mm.history(U)), 1)
        self.assertEqual(d.result.last_confirmed, mm.clock.now())

    def test_a_changed_number_replaces_instead_of_confirming(self):
        mm = manager()
        [old] = [d.result for d in mm.add(U, Candidate("fact", "note", "boot size", "Hiking boot size is EU 42"))]
        [d] = mm.add(U, Candidate("fact", "note", "boot size", "Hiking boot size is EU 43"))
        self.assertEqual((d.op, d.target.id), ("SUPERSEDE", old.id))
        self.assertIn("42 -> 43", d.reason)
        self.assertEqual([m.text for m in mm.current(U)], ["Hiking boot size is EU 43"])

    def test_numbers_alone_dont_replace_episodes_or_restatements(self):
        mm = manager()
        mm.add(U, Candidate("episode", "experience", "late delivery", "Order 5531 arrived late"))
        [d] = mm.add(U, Candidate("episode", "experience", "late delivery", "Order 6120 arrived late"))
        self.assertEqual(d.op, "ADD")                     # two separate events
        mm.add(U, pref("likes", "running", "Runs 5k every morning"))
        [d] = mm.add(U, pref("likes", "running", "Runs every morning"))
        self.assertEqual(d.op, "CONFIRM")                 # no numbers on one side: a restatement

    def test_negation_retracts(self):
        mm = manager()
        say(mm, "I love coffee.")
        ops = [d.op for d in say(mm, "I don't drink coffee anymore, switched to tea.")]
        self.assertEqual(ops, ["RETRACT", "ADD"])
        self.assertEqual(current_values(mm, "likes"), ["tea"])

    def test_negation_filed_under_another_predicate_still_retracts(self):
        mm = manager()
        say(mm, "I love coffee.")
        [d] = mm.add(U, [Candidate(kind="preference", predicate="diet", value="coffee",
                                   text="No longer drinks coffee", negated=True)])
        self.assertEqual(d.op, "RETRACT")
        self.assertEqual(current_values(mm, "likes"), [])

    def test_negation_never_retracts_a_sensitive_memory_by_another_name(self):
        mm = manager()
        say(mm, "Remember that I'm allergic to peanuts.")
        [d] = mm.add(U, [Candidate(kind="preference", predicate="diet", value="peanuts",
                                   text="Doesn't eat peanuts anymore", negated=True)])
        self.assertEqual(d.op, "IGNORE")
        self.assertEqual(current_values(mm, "allergy"), ["peanuts"])

    def test_lasting_facts_never_expire(self):
        mm = manager()
        [d] = mm.add(U, [Candidate(kind="fact", predicate="employer", value="Acme",
                                   text="Works at Acme", ttl_days=30)])
        self.assertIsNone(d.result.expires_at)

    def test_one_diet_at_a_time(self):
        mm = manager()
        mm.add(U, [Candidate(kind="fact", predicate="diet", value="vegetarian", text="Is vegetarian")])
        [d] = mm.add(U, [Candidate(kind="fact", predicate="diet", value="eats meat", text="Eats meat")])
        self.assertEqual(d.op, "SUPERSEDE")
        self.assertEqual(current_values(mm, "diet"), ["eats meat"])

    def test_sensitive_needs_explicit_consent(self):
        mm = manager()
        [d] = say(mm, "I was diagnosed with diabetes last year.")
        self.assertEqual(d.op, "IGNORE")
        self.assertEqual(mm.current(U), [])
        # ...and the refused detail doesn't leak into the audit log either.
        self.assertNotIn("diabetes", " ".join(r["detail"] for r in mm.store.logs(U)))

        [d] = say(mm, "Remember that I'm allergic to peanuts.")
        self.assertEqual(d.op, "ADD")
        self.assertTrue(d.result.sensitive)
        self.assertGreaterEqual(d.result.confidence, 0.95)

    def test_explicit_note_without_pattern_is_kept(self):
        mm = manager()
        [d] = say(mm, "Remember that the spare key is under the blue pot.")
        self.assertEqual((d.op, d.result.predicate), ("ADD", "note"))


class Maintenance(unittest.TestCase):
    def test_situational_fact_expires(self):
        mm = manager()
        say(mm, "I'm traveling to Tokyo next week.")
        self.assertEqual(current_values(mm, "traveling_to"), ["Tokyo"])
        mm.clock.advance(15)
        self.assertEqual(current_values(mm, "traveling_to"), [])  # hidden even before maintenance
        self.assertTrue(any("EXPIRE" in line for line in mm.maintain(U)))

    def test_unused_episode_is_pruned(self):
        mm = manager()
        say(mm, "We tried Redis for the session cache and it kept running out of memory.")
        mm.clock.advance(200)
        self.assertTrue(any("PRUNE" in line for line in mm.maintain(U)))
        self.assertEqual(mm.history(U), [])


class ReadPath(unittest.TestCase):
    def setUp(self):
        self.mm = manager()
        say(self.mm,
            "My name is Sam.",
            "I live in Berlin and I work at Acme as a backend engineer.",
            "I prefer short answers with code examples.",
            "Remember that I'm allergic to peanuts.",
            "We tried Redis for the session cache and it kept running out of memory.")

    def actions(self, query: str, scope: str = "general") -> dict[str, str]:
        r = self.mm.recall(U, query, scope=scope)
        return {d.scored.memory.predicate: d.action for d in r.items}

    def test_profile_is_always_loaded(self):
        r = self.mm.recall(U, "unrelated question about astronomy")
        self.assertEqual({m.predicate for m in r.profile}, {"name", "response_style"})

    def test_sensitive_memory_used_when_relevant(self):
        self.assertEqual(self.actions("Any snack ideas?").get("allergy"), "USE")

    def test_sensitive_memory_absent_when_unrelated(self):
        self.mm.add(U, [Candidate(kind="fact", predicate="health", value="diabetes", text="Has diabetes",
                                  sensitive=True, explicit=True)])
        r = self.mm.recall(U, "Write a Python function to parse dates")
        self.assertNotIn("health", {d.scored.memory.predicate for d in r.used})
        self.assertNotIn("diabetes", self.mm.build_context(r))

    def test_episode_is_mentioned(self):
        self.assertEqual(self.actions("How should I cache sessions?", "work").get("experience"),
                         "MENTION")

    def test_scope_mismatch_is_held(self):
        acts = self.actions("I'm stressed about work, how do I relax this weekend?", "personal")
        self.assertEqual(acts.get("employer"), "HOLD")

    def test_context_can_be_about_something_other_than_a_person(self):
        r = self.mm.recall(U, "Any snack ideas?")
        self.assertIn("## About the user (always loaded)", self.mm.build_context(r))
        self.assertIn("## About the project (always loaded)", self.mm.build_context(r, about="the project"))

    def test_held_memories_never_reach_the_prompt(self):
        r = self.mm.recall(U, "I'm stressed about work, how do I relax?", scope="personal")
        self.assertNotIn("Acme", self.mm.build_context(r))

    def test_stale_fact_is_verified(self):
        self.mm.clock.advance(200)
        self.assertEqual(self.actions("Help me prep for my performance review at work", "work")
                         .get("employer"), "VERIFY")

    def test_using_a_memory_does_not_confirm_it(self):
        before = next(m for m in self.mm.current(U) if m.predicate == "employer").last_confirmed
        self.mm.clock.advance(5)
        self.mm.recall(U, "my job at work", scope="work")
        after = next(m for m in self.mm.current(U) if m.predicate == "employer")
        self.assertEqual(after.last_confirmed, before)
        self.assertEqual(after.use_count, 1)

    def test_superseded_memory_is_never_retrieved(self):
        self.mm.clock.advance(1)
        say(self.mm, "I just moved to Lisbon.")
        r = self.mm.recall(U, "What's the weather like in my city?")
        self.assertNotIn("Berlin", [d.scored.memory.value for d in r.items])


class SafetyMemories(unittest.TestCase):
    """Allergies reach the prompt even when the question looks unrelated: holding one back
    because "dinner ideas" scored low once led a real model to suggest peanut noodles."""

    UNRELATED = "Write a Python function to parse dates"

    def setUp(self):
        self.mm = manager()
        say(self.mm, "I live in Berlin.", "Remember that I'm allergic to peanuts.")

    def context(self, query: str, scope: str = "general", mm: MemoryManager | None = None) -> str:
        mm = mm or self.mm
        return mm.build_context(mm.recall(U, query, scope=scope))

    def test_allergy_is_used_for_an_unrelated_question(self):
        r = self.mm.recall(U, self.UNRELATED)
        self.assertEqual({d.scored.memory.predicate: d.action for d in r.items}.get("allergy"), "USE")

    def test_allergy_goes_in_its_own_section_with_a_safety_instruction(self):
        ctx = self.context(self.UNRELATED)
        self.assertIn("## Always respect (safety)\n- Allergic to peanuts", ctx)
        self.assertIn("never suggest anything that conflicts with this", ctx)
        self.assertNotIn("## Memories relevant to this message", ctx)   # Berlin isn't relevant here

    def test_old_allergy_is_still_applied_not_questioned(self):
        self.mm.clock.advance(400)
        r = self.mm.recall(U, "Any snack ideas?")
        self.assertEqual({d.scored.memory.predicate: d.action for d in r.items}.get("allergy"), "USE")

    def test_allergies_do_not_take_top_k_slots(self):
        mm = MemoryManager(store(), clock=self.mm.clock, config=MemoryConfig(top_k=1))
        say(mm, "I live in Berlin.", "Remember that I'm allergic to peanuts.",
            "Remember that I'm allergic to shellfish.")
        r = mm.recall(U, "What's the weather like in Berlin?")
        self.assertEqual(sorted(d.scored.memory.value for d in r.used), ["Berlin", "peanuts", "shellfish"])

    def test_allergy_filed_as_personal_still_reaches_other_scopes(self):
        mm = manager()
        mm.add(U, [Candidate(kind="fact", predicate="allergy", value="peanuts", text="Allergic to peanuts",
                             scope="personal", sensitive=True, explicit=True)])
        for scope in ("work", "general", "personal"):
            self.assertIn("Allergic to peanuts", self.context("Order lunch for the team", scope, mm))

    def test_other_memories_still_respect_scope(self):
        mm = manager()
        mm.add(U, [Candidate(kind="fact", predicate="allergy", value="peanuts", text="Allergic to peanuts",
                             scope="personal", sensitive=True, explicit=True),
                   Candidate(kind="fact", predicate="relationship", value="divorce",
                             text="Going through a divorce", scope="personal")])
        self.assertNotIn("divorce", self.context("Going through my divorce paperwork", "work", mm))

    def test_retracted_allergy_is_no_longer_included(self):
        self.mm.add(U, [Candidate(kind="fact", predicate="allergy", value="peanuts",
                                  text="No longer allergic to peanuts", negated=True)])
        self.assertNotIn("peanuts", self.context("Any snack ideas?"))

    def test_can_be_turned_off(self):
        mm = MemoryManager(store(), clock=self.mm.clock, config=MemoryConfig(safety_predicates=set()))
        say(mm, "Remember that I'm allergic to peanuts.")
        self.assertNotIn("peanuts", self.context(self.UNRELATED, mm=mm))


class Secrets(unittest.TestCase):
    """Card numbers, passwords and keys never reach the store or the log, even when asked."""

    def add(self, mm: MemoryManager, predicate: str, value: str, text: str, **kw):
        [d] = mm.add(U, [Candidate(kind="fact", predicate=predicate, value=value, text=text, **kw)])
        return d

    def test_card_number_is_refused_even_when_the_user_asks(self):
        mm = manager()
        d = self.add(mm, "payment_card", "4242 4242 4242 4242", "Card number 4242 4242 4242 4242",
                     explicit=True, sensitive=True)
        self.assertEqual(d.op, "IGNORE")
        self.assertIn("never stored", d.reason)
        self.assertEqual(mm.current(U), [])
        self.assertNotIn("4242", log_text(mm))

    def test_password_is_refused_by_its_predicate_alone(self):
        mm = manager()
        self.assertEqual(self.add(mm, "password", "Sunflower!2024", "Account password", explicit=True).op,
                         "IGNORE")
        self.assertNotIn("sunflower", log_text(mm))

    def test_api_key_and_bank_details_are_refused(self):
        mm = manager()
        for value in ("sk-proj-abcdefghijklmnop1234", "GB82 WEST 1234 5698 7654 32"):
            self.assertEqual(self.add(mm, "note", value, f"Remember {value}", explicit=True).op, "IGNORE")

    def test_a_retracted_secret_is_not_logged_either(self):
        mm = manager()
        self.add(mm, "note", "Sunflower!2024", "Old password is Sunflower!2024", negated=True)
        self.assertNotIn("sunflower", log_text(mm))

    def test_lookalikes_are_kept(self):
        mm = manager()
        for text in ("Never commit API keys to the repo", "Prefers a password manager",
                     "Uses a rolling pin for pastry", "Prefers the Visa card ending 4242"):
            self.assertNotEqual(self.add(mm, "note", text, text).op, "IGNORE", text)

    def test_can_be_turned_off(self):
        mm = MemoryManager(store(), config=MemoryConfig(block_secrets=False))
        self.assertEqual(self.add(mm, "note", "PIN 4821", "Door PIN 4821", explicit=True).op, "ADD")


class Addresses(unittest.TestCase):
    def test_a_new_address_replaces_the_old_one(self):
        mm = manager()
        mm.add(U, [Candidate(kind="fact", predicate="address", value="14 Rose Street, Leeds",
                             text="Address is 14 Rose Street, Leeds")])
        [d] = mm.add(U, [Candidate(kind="fact", predicate="address", value="7 Kings Road, York",
                                   text="Address is 7 Kings Road, York")])
        self.assertEqual(d.op, "SUPERSEDE")
        self.assertEqual(current_values(mm, "address"), ["7 Kings Road, York"])


def log_text(mm: MemoryManager) -> str:
    return " ".join(str(dict(row)) for row in mm.store.logs(U)).lower()


class UserControl(unittest.TestCase):
    def test_asking_to_forget_in_chat_deletes_like_the_button(self):
        mm = manager()
        say(mm, "I live in Berlin.", "I like jazz.")
        say(mm, "I just moved to Lisbon!")
        [d] = say(mm, "Please forget that I live in Lisbon.")
        self.assertEqual(d.op, "FORGET")
        self.assertEqual([m.text for m in mm.history(U)], ["Likes jazz"])  # Berlin went too
        self.assertNotIn("lisbon", log_text(mm))

    def test_asking_to_forget_works_for_sensitive_memories(self):
        mm = manager()
        say(mm, "Remember that I'm allergic to peanuts.")
        [d] = mm.add(U, [Candidate(kind="fact", predicate="allergy", value="peanuts",
                                   text="Allergic to peanuts", forget=True, sensitive=True)])
        self.assertEqual(d.op, "FORGET")
        self.assertEqual(mm.history(U), [])

    def test_asking_to_forget_something_unknown_logs_nothing_about_it(self):
        mm = manager()
        [d] = say(mm, "Please forget that I live in Rome.")
        self.assertEqual(d.op, "IGNORE")
        self.assertNotIn("rome", log_text(mm))

    def test_forget_removes_row_history_and_log_text(self):
        mm = manager()
        say(mm, "I live in Berlin.")
        mm.clock.advance(1)
        [d] = say(mm, "I moved to Lisbon.")
        removed = mm.forget(U, d.result.id)
        self.assertEqual(sorted(removed), sorted([d.result.id, d.target.id]))
        self.assertEqual(mm.history(U), [])
        log_text = " ".join(r["detail"] for r in mm.store.logs(U))
        self.assertNotIn("Lisbon", log_text)
        self.assertNotIn("Berlin", log_text)

    def test_cannot_forget_another_users_memory(self):
        mm = manager()
        say(mm, "I live in Berlin.")
        mid = mm.current(U)[0].id
        self.assertEqual(mm.forget("someone-else", mid), [])
        self.assertEqual(len(mm.current(U)), 1)


ITEM = {
    "kind": "fact", "predicate": "lives_in", "value": "Porto", "text": "Lives in Porto",
    "scope": "general", "negated": False, "explicit": False, "source_kind": "stated",
    "confidence": 0.9, "importance": 0.7, "sensitive": False, "ttl_days": 0, "source_ref": "s1:1",
}


class LLMExtractorParsing(unittest.TestCase):
    """Checks how we read a model's reply, with a fake model function (no network)."""

    def fake_llm(self, reply: str):
        self.calls = []

        def llm(system, messages):
            self.calls.append((system, messages))
            return reply

        return llm

    def test_candidates_flow_into_the_same_pipeline(self):
        mm = manager()
        mm.extractor = LLMExtractor(self.fake_llm(json.dumps({"memories": [ITEM]})))
        [d] = mm.ingest(U, [("s1:1", "Finally settled in Porto")])
        self.assertEqual((d.op, d.result.value, d.result.expires_at), ("ADD", "Porto", None))
        system, messages = self.calls[0]
        self.assertIn('{"memories": []}', system)          # the prompt asks for JSON
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("[s1:1] user: Finally settled in Porto", messages[0]["content"])

    def test_finds_the_json_inside_prose_and_code_fences(self):
        reply = "Sure! Here you go:\n```json\n" + json.dumps({"memories": [ITEM]}) + "\n```\nDone."
        [c] = parse_memories(reply)
        self.assertEqual(c.value, "Porto")

    def test_forgives_loose_types_and_missing_fields(self):
        [c] = parse_memories(json.dumps([{
            "kind": "Fact", "predicate": "traveling_to", "value": "Rome",
            "explicit": "true", "confidence": "1.7", "scope": "holidays", "ttl_days": "14",
        }]))
        self.assertEqual((c.kind, c.explicit, c.confidence, c.scope, c.ttl_days),
                         ("fact", True, 1.0, "general", 14))
        self.assertEqual(c.text, "traveling_to: Rome")

    def test_reads_forget_requests(self):
        [c] = parse_memories(json.dumps([{**ITEM, "forget": True}]))
        self.assertTrue(c.forget)

    def test_skips_items_it_cannot_use(self):
        reply = json.dumps({"memories": [{"kind": "rumour", "predicate": "x", "value": "y"},
                                         {"kind": "fact", "value": "no predicate"}, "junk", ITEM]})
        self.assertEqual([c.value for c in parse_memories(reply)], ["Porto"])

    def test_empty_reply_means_nothing_to_remember(self):
        self.assertEqual(parse_memories(""), [])
        self.assertEqual(LLMExtractor(self.fake_llm("")).extract([("s1:1", "hi")]), [])

    def test_reply_without_json_raises_a_clear_error(self):
        with self.assertRaises(ValueError):
            parse_memories("I'm sorry, I can't help with that.")


class FakeServer:
    """A local HTTP server that answers like a model provider (no network)."""

    def serve(self, status: int, reply):
        """`reply` is a dict, or a function of the request body that returns one."""
        received: dict = {"bodies": []}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received["path"] = self.path
                received["auth"] = self.headers.get("Authorization")
                received["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received["bodies"].append(received["body"])
                payload = json.dumps(reply(received["body"]) if callable(reply) else reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}/v1/", received


class ConversationContext(unittest.TestCase):
    """What the extractor gets to see besides the user's own words."""

    def fake_llm(self, reply: dict | None = None):
        self.prompts = []

        def llm(system, messages):
            self.prompts.append(messages[0]["content"])
            return json.dumps(reply or {"memories": []})

        return llm

    def test_assistant_turns_are_shown_as_context(self):
        mm = manager()
        mm.extractor = LLMExtractor(self.fake_llm())
        mm.ingest(U, [{"role": "user", "content": "Book me a flight to Rome."},
                      {"role": "assistant", "content": "Window or aisle?"},
                      {"role": "user", "content": "Window, always.", "ref": "s1:3"},
                      {"role": "system", "content": "You are a travel agent."}])
        prompt = self.prompts[0]
        self.assertIn("[1] user: Book me a flight to Rome.", prompt)
        self.assertIn("assistant: Window or aisle?", prompt)
        self.assertIn("[s1:3] user: Window, always.", prompt)
        self.assertNotIn("travel agent", prompt)

    def test_offline_extractor_never_learns_from_the_assistant(self):
        mm = manager()
        say_turns = [{"role": "assistant", "content": "I live in Paris and I like opera."},
                     {"role": "user", "content": "I like jazz."}]
        mm.ingest(U, say_turns)
        self.assertEqual([m.text for m in mm.current(U)], ["Likes jazz"])

    def test_stored_memories_are_shown_but_never_sensitive_ones(self):
        mm = manager()
        say(mm, "I live in Berlin.", "Remember that I'm allergic to peanuts.")
        mm.extractor = LLMExtractor(self.fake_llm())
        mm.ingest(U, ["I left last week."])
        self.assertIn("lives_in: Berlin", self.prompts[0])
        self.assertNotIn("peanut", self.prompts[0].lower())

    def test_only_the_most_relevant_stored_memories_are_shown(self):
        mm = manager()
        say(mm, "I live in Berlin.", "I love hiking.", "I work at Acme.")
        mm.config.extraction_context_memories = 1
        mm.extractor = LLMExtractor(self.fake_llm())
        mm.ingest(U, ["Thinking about a hiking trip in the Alps."])
        self.assertIn("likes: hiking", self.prompts[0])
        self.assertNotIn("Berlin", self.prompts[0])

    def test_echoed_memories_do_not_count_as_confirmed(self):
        mm = manager()
        say(mm, "I live in Berlin.")
        berlin = mm.current(U)[0]
        mm.clock.advance(30)
        echo = {**ITEM, "predicate": "lives_in", "value": "Berlin", "text": "Lives in Berlin",
                "source_ref": ""}
        new = {**ITEM, "predicate": "likes", "value": "jazz", "text": "Likes jazz", "source_ref": "s2:2"}
        mm.extractor = LLMExtractor(self.fake_llm({"memories": [echo, new]}))
        decisions = mm.ingest(U, [("s2:1", "What's on this weekend?"), ("s2:2", "I like jazz.")])
        self.assertEqual([d.op for d in decisions], ["ADD"])
        self.assertEqual(mm.store.get(berlin.id).last_confirmed, berlin.last_confirmed)

    def test_extractors_without_the_new_arguments_still_work(self):
        class OldStyle:
            def extract(self, messages):
                return [Candidate(kind="fact", predicate="lives_in", value="Oslo",
                                  text="Lives in Oslo", source_ref=ref) for ref, _ in messages]

        mm = manager()
        mm.extractor = OldStyle()
        [d] = mm.ingest(U, [{"role": "assistant", "content": "Hi!"},
                            {"role": "user", "content": "I'm in Oslo."}])
        self.assertEqual((d.op, d.result.source_ref), ("ADD", "2"))


class OpenAICompatibleClient(FakeServer, unittest.TestCase):
    """The built-in chat client against a local fake server."""

    def test_sends_a_chat_completions_request(self):
        url, got = self.serve(200, {"choices": [{"message": {"content": "Hello!"}}]})
        llm = OpenAICompatible(base_url=url, model="any-model", api_key="test-key",
                               options={"temperature": 0})
        self.assertEqual(llm("Be brief.", [{"role": "user", "content": "Hi"}]), "Hello!")
        self.assertEqual(got["path"], "/v1/chat/completions")
        self.assertEqual(got["auth"], "Bearer test-key")
        self.assertEqual(got["body"]["model"], "any-model")
        self.assertEqual(got["body"]["temperature"], 0)
        self.assertEqual(got["body"]["messages"], [{"role": "system", "content": "Be brief."},
                                                   {"role": "user", "content": "Hi"}])
        self.assertNotIn("test-key", repr(llm))

    def test_local_servers_need_no_key(self):
        url, got = self.serve(200, {"choices": [{"message": {"content": "ok"}}]})
        OpenAICompatible(base_url=url, model="local")("s", [])
        self.assertIsNone(got["auth"])

    def test_http_errors_carry_the_status(self):
        url, _ = self.serve(401, {"error": {"message": "invalid api key"}})
        with self.assertRaises(LLMError) as caught:
            OpenAICompatible(base_url=url, model="m", api_key="bad")("s", [])
        self.assertEqual(caught.exception.status, 401)
        self.assertIn("invalid api key", str(caught.exception))

    def test_dropped_connections_are_retried_and_reported_as_llm_errors(self):
        import memwise.llm as llm_module

        class Reply:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, *args):
                return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

        calls = []

        def flaky(request, timeout):
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionResetError(54, "Connection reset by peer")
            return Reply()

        real_open, real_sleep = llm_module.urllib.request.urlopen, llm_module.time.sleep
        llm_module.urllib.request.urlopen, llm_module.time.sleep = flaky, lambda s: None
        try:
            self.assertEqual(OpenAICompatible(base_url="http://x/v1", model="m")("s", []), "ok")
            self.assertEqual(len(calls), 3)                 # two drops, then success
            calls.clear()
            with self.assertRaises(LLMError):               # never a raw ConnectionResetError
                OpenAICompatible(base_url="http://x/v1", model="m", retries=1)("s", [])
            self.assertEqual(len(calls), 2)
        finally:
            llm_module.urllib.request.urlopen, llm_module.time.sleep = real_open, real_sleep

    def test_client_errors_are_not_retried(self):
        url, got = self.serve(401, {"error": {"message": "invalid api key"}})
        with self.assertRaises(LLMError):
            OpenAICompatible(base_url=url, model="m", api_key="bad")("s", [])
        self.assertEqual(len(got["bodies"]), 1)

    def test_from_env_explains_what_to_set(self):
        with self.assertRaises(LLMError) as caught:
            OpenAICompatible.from_env(prefix="MEMWISE_TEST_UNSET_")
        self.assertIn("MEMWISE_TEST_UNSET_BASE_URL", str(caught.exception))


def fake_embeddings(body):
    """Embeds each text as [len, 1, 0] and returns the rows in reverse order, to test re-ordering."""
    rows = [{"index": i, "embedding": [len(t), 1.0, 0.0]} for i, t in enumerate(body["input"])]
    return {"data": rows[::-1]}


class Embedder3D:
    """An in-process embedder with 3 dimensions, counting its batch calls."""

    def __init__(self):
        self.batches = 0

    def embed(self, text):
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        self.batches += 1
        return [[1.0, 0.0, 0.0] for _ in texts]


class EmbeddingModels(FakeServer, unittest.TestCase):
    """The built-in embeddings client, and switching embedders safely."""

    def test_sends_an_embeddings_request_and_normalizes(self):
        url, got = self.serve(200, {"data": [{"index": 0, "embedding": [3.0, 4.0]}]})
        e = OpenAICompatibleEmbedder(base_url=url, model="embed-model", api_key="test-key",
                                     options={"dimensions": 2})
        self.assertEqual(e.embed("Lives in Lisbon"), [0.6, 0.8])
        self.assertEqual(got["path"], "/v1/embeddings")
        self.assertEqual(got["auth"], "Bearer test-key")
        self.assertEqual(got["body"], {"model": "embed-model", "input": ["Lives in Lisbon"],
                                       "dimensions": 2})
        self.assertNotIn("test-key", repr(e))

    def test_remembers_texts_it_already_embedded(self):
        url, got = self.serve(200, fake_embeddings)
        e = OpenAICompatibleEmbedder(base_url=url, model="m")
        first = e.embed("tea")
        self.assertEqual(e.embed("tea"), first)
        e.embed_many(["tea", "coffee", "tea"])
        self.assertEqual([b["input"] for b in got["bodies"]], [["tea"], ["coffee"]])

    def test_batches_requests_and_keeps_the_order(self):
        url, got = self.serve(200, fake_embeddings)
        e = OpenAICompatibleEmbedder(base_url=url, model="m", batch_size=2)
        texts = ["a", "bb", "ccc", "dddd", "eeeee"]
        vectors = e.embed_many(texts)
        self.assertEqual(len(got["bodies"]), 3)
        self.assertEqual([round(v[0] / v[1]) for v in vectors], [1, 2, 3, 4, 5])

    def test_blank_text_needs_no_request(self):
        url, got = self.serve(200, fake_embeddings)
        self.assertEqual(OpenAICompatibleEmbedder(base_url=url, model="m").embed("  "), [])
        self.assertEqual(got["bodies"], [])

    def test_from_env_reuses_the_chat_settings(self):
        env = {"MWT_BASE_URL": "http://chat.test/v1", "MWT_EMBED_MODEL": "e1", "MWT_API_KEY": "k"}
        old = {k: os.environ.get(k) for k in [*env, "MWT_EMBED_BASE_URL"]}
        os.environ.update(env)
        try:
            e = OpenAICompatibleEmbedder.from_env(prefix="MWT_")
            self.assertEqual((e.base_url, e.model), ("http://chat.test/v1", "e1"))
            os.environ["MWT_EMBED_BASE_URL"] = "http://embed.test/v1"
            self.assertEqual(OpenAICompatibleEmbedder.from_env(prefix="MWT_").base_url,
                             "http://embed.test/v1")
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_switching_embedders_fails_loudly_until_reembedded(self):
        mm = manager()
        say(mm, "I live in Berlin.", "I love hiking.")
        mm.embedder = Embedder3D()
        with self.assertRaisesRegex(ValueError, "reembed"):
            mm.recall(U, "weekend plans?")
        mm.embedder.batches = 0
        self.assertEqual(mm.reembed(), 2)
        self.assertEqual(mm.embedder.batches, 1)       # one batch for the user's memories
        mm.recall(U, "weekend plans?")

    def test_embedding_model_preset(self):
        c = MemoryConfig.for_embedding_model(top_k=3)
        self.assertLess(c.relevant_enough, MemoryConfig().relevant_enough)
        self.assertGreater(c.duplicate_similarity, 0.81)   # keeps "peanuts" and "tree nuts" apart
        self.assertEqual(c.top_k, 3)


class FakeJudgeLLM:
    """Stands in for the model behind `LLMJudge`. `answer` gives the decisions for a prompt."""

    def __init__(self, answer=None, raw: str | None = None, error: Exception | None = None):
        self.answer, self.raw, self.error = answer, raw, error
        self.prompts: list[str] = []
        self.during = None   # called while the model is "thinking"

    def __call__(self, system, messages):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if self.during:
            self.during()
        if self.error:
            raise self.error
        if self.raw is not None:
            return self.raw
        return json.dumps({"decisions": self.answer(prompt) if self.answer else []})


def judged(llm: FakeJudgeLLM, **config) -> MemoryManager:
    config.setdefault("judge_min_similarity", -1.0)   # show every stored memory to the judge
    clock = Clock(fixed_start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    return MemoryManager(store(), clock=clock, config=MemoryConfig(**config), judge=LLMJudge(llm))


def pref(predicate: str, value: str, text: str, kind: str = "preference", **kw) -> Candidate:
    return Candidate(kind, predicate, value, text, **kw)


class Judging(unittest.TestCase):
    """The optional judge: asked only where the rules can't tell, and checked before it's used."""

    def test_without_a_judge_the_rules_keep_both(self):
        mm = manager()
        mm.add(U, pref("likes", "steak", "Loves steak"))
        [d] = mm.add(U, pref("diet", "vegetarian", "Went vegetarian", kind="fact"))
        self.assertEqual(d.op, "ADD")
        self.assertEqual(len(mm.current(U)), 2)

    def test_update_supersedes_and_keeps_history(self):
        llm = FakeJudgeLLM()
        mm = judged(llm)
        [steak] = [d.result for d in mm.add(U, pref("likes", "steak", "Loves steak"))]
        llm.answer = lambda p: [{"item": 1, "op": "UPDATE", "ids": [steak.id], "reason": "no meat now"}]
        [d] = mm.add(U, pref("diet", "vegetarian", "Went vegetarian", kind="fact"))
        self.assertEqual((d.op, d.reason), ("SUPERSEDE", "model: no meat now"))
        self.assertEqual([m.text for m in mm.current(U)], ["Went vegetarian"])
        old = mm.store.get(steak.id)
        self.assertIsNotNone(old.valid_to)                 # closed, not deleted
        self.assertEqual(old.superseded_by, d.result.id)
        self.assertIn("Loves steak", llm.prompts[0])

    def test_update_can_close_several_memories(self):
        llm = FakeJudgeLLM()
        mm = judged(llm)
        ids = [d.result.id for d in mm.add(U, [pref("likes", "steak", "Loves steak"),
                                                pref("likes", "cheese", "Likes cheese"),
                                                pref("likes", "tea", "Likes tea")])]
        llm.answer = lambda p: [{"item": 1, "op": "UPDATE", "ids": ids[:2]}]
        [d] = mm.add(U, pref("diet", "vegan", "Went vegan", kind="fact"))
        self.assertEqual(d.op, "SUPERSEDE")
        self.assertEqual(sorted(m.text for m in mm.current(U)), ["Likes tea", "Went vegan"])
        self.assertEqual(sum(1 for row in mm.store.logs(U) if row["op"] == "SUPERSEDE"), 2)

    def test_none_confirms_instead_of_adding(self):
        llm = FakeJudgeLLM()
        mm = judged(llm)
        [hiking] = [d.result for d in mm.add(U, pref("likes", "hiking", "Loves hiking"))]
        mm.clock.advance(30)
        llm.answer = lambda p: [{"item": 1, "op": "NONE", "ids": [hiking.id]}]
        [d] = mm.add(U, pref("likes", "hill walking", "Enjoys walks in the hills"))
        self.assertEqual((d.op, d.target.id), ("CONFIRM", hiking.id))
        self.assertEqual(len(mm.history(U)), 1)
        self.assertEqual(mm.store.get(hiking.id).last_confirmed, mm.clock.now())

    def test_add_keeps_both(self):
        llm = FakeJudgeLLM(lambda p: [{"item": 1, "op": "ADD"}])
        mm = judged(llm)
        mm.add(U, pref("likes", "tea", "Likes tea"))
        [d] = mm.add(U, pref("likes", "coffee", "Likes coffee in the afternoon"))
        self.assertEqual(d.op, "ADD")
        self.assertEqual(len(mm.current(U)), 2)

    def test_delete_retracts_what_stopped_in_other_words(self):
        llm = FakeJudgeLLM()
        mm = judged(llm)
        [running] = [d.result for d in mm.add(U, pref("likes", "running", "Goes running every morning"))]
        llm.answer = lambda p: [{"item": 1, "op": "DELETE", "ids": [running.id]}]
        [d] = mm.add(U, pref("likes", "jogging", "No longer jogs", negated=True))
        self.assertEqual(d.op, "RETRACT")
        self.assertEqual(mm.current(U), [])
        self.assertIn("(no longer true)", llm.prompts[0])

    def test_new_wording_of_the_same_value_can_replace_it(self):
        llm = FakeJudgeLLM()
        mm = judged(llm)
        [old] = [d.result for d in mm.add(U, pref("likes", "running", "Runs 5k every morning"))]
        llm.answer = lambda p: [{"item": 1, "op": "UPDATE", "ids": [old.id]}]
        [d] = mm.add(U, pref("likes", "running", "Runs 10k every morning"))
        self.assertEqual(d.op, "SUPERSEDE")
        self.assertEqual([m.text for m in mm.current(U)], ["Runs 10k every morning"])

    def test_new_wording_must_replace_the_memory_the_rules_matched(self):
        llm = FakeJudgeLLM()
        mm = judged(llm)
        run, tea = [d.result for d in mm.add(U, [pref("likes", "running", "Goes running most mornings"),
                                                 pref("likes", "tea", "Likes tea")])]
        llm.answer = lambda p: [{"item": 1, "op": "UPDATE", "ids": [tea.id]}]
        [d] = mm.add(U, pref("likes", "running", "Running is how every day starts"))
        self.assertEqual((d.op, d.target.id), ("CONFIRM", run.id))   # the rules' answer stands
        self.assertTrue(mm.store.get(tea.id).is_current(mm.clock.now()))

    def test_ids_it_was_not_shown_are_ignored(self):
        llm = FakeJudgeLLM()
        mm = judged(llm, judge_related=1)
        mm.add("someone-else", pref("likes", "steak", "Loves steak"))
        a, b = [d.result for d in mm.add(U, [pref("likes", "steak", "Loves steak"),
                                             pref("likes", "tea", "Likes tea")])]
        theirs = mm.current("someone-else")[0]
        # Only one memory is shown (judge_related=1); name everything except that one.
        llm.answer = lambda p: [{"item": 1, "op": "UPDATE",
                                 "ids": [m for m in (a.id, b.id) if f"  {m}:" not in p] + [theirs.id, "m999"]}]
        [d] = mm.add(U, pref("diet", "vegetarian", "Went vegetarian", kind="fact"))
        self.assertEqual(sum(f"  {m}:" in llm.prompts[-1] for m in (a.id, b.id)), 1)
        self.assertEqual(d.op, "ADD")
        self.assertEqual(len(mm.current(U)), 3)
        self.assertEqual(len(mm.current("someone-else")), 1)

    def test_never_sees_or_closes_sensitive_and_safety_memories(self):
        llm = FakeJudgeLLM()
        mm = judged(llm)
        allergy, health = [d.result for d in mm.add(U, [
            Candidate("fact", "allergy", "shellfish", "Allergic to shellfish", explicit=True),
            Candidate("fact", "health", "diabetes", "Has type 1 diabetes", sensitive=True, explicit=True)])]
        mm.add(U, pref("likes", "beaches", "Loves beach holidays"))
        llm.answer = lambda p: [{"item": 1, "op": "UPDATE", "ids": [allergy.id, health.id]}]
        [d] = mm.add(U, pref("likes", "seafood", "Loves seafood platters"))
        self.assertEqual(d.op, "ADD")
        self.assertNotIn("Allergic to shellfish", llm.prompts[0])
        self.assertNotIn("diabetes", llm.prompts[0])
        now = mm.clock.now()
        self.assertTrue(mm.store.get(allergy.id).is_current(now))
        self.assertTrue(mm.store.get(health.id).is_current(now))

    def test_secrets_sensitive_safety_and_forget_never_reach_the_judge(self):
        llm = FakeJudgeLLM(lambda p: [{"item": 1, "op": "NONE", "ids": ["m1"]}])
        mm = judged(llm)
        mm.add(U, pref("likes", "peanut butter", "Loves peanut butter"))
        llm.prompts.clear()
        ds = mm.add(U, [
            Candidate("fact", "note", "card", "Card is 4111 1111 1111 1111", explicit=True),
            Candidate("fact", "health", "asthma", "Has asthma", sensitive=True, explicit=True),
            Candidate("fact", "allergy", "peanuts", "Allergic to peanuts", explicit=True),
            pref("likes", "peanut butter", "Loves peanut butter", forget=True)])
        self.assertEqual(llm.prompts, [])
        self.assertEqual([d.op for d in ds], ["IGNORE", "ADD", "ADD", "FORGET"])

    def test_a_single_valued_change_stands_whatever_the_judge_says(self):
        llm = FakeJudgeLLM()
        mm = judged(llm)
        berlin, cafe = [d.result for d in mm.add(U, [
            Candidate("fact", "lives_in", "Berlin", "Lives in Berlin"),
            pref("likes", "Cafe Lotte", "Favourite cafe is Cafe Lotte in Berlin")])]
        llm.answer = lambda p: [{"item": 1, "op": "NONE", "ids": [cafe.id]}]
        [d] = mm.add(U, Candidate("fact", "lives_in", "Lisbon", "Lives in Lisbon"))
        self.assertEqual((d.op, d.target.id, d.also), ("SUPERSEDE", berlin.id, []))
        self.assertTrue(mm.store.get(cafe.id).is_current(mm.clock.now()))

    def test_the_judge_can_keep_both_when_only_numbers_differ(self):
        llm = FakeJudgeLLM(lambda p: [{"item": 1, "op": "ADD", "reason": "two different children"}])
        mm = judged(llm)
        mm.add(U, Candidate("fact", "note", "child", "Has a 7-year-old child"))
        [d] = mm.add(U, Candidate("fact", "note", "child", "Has a 9-year-old child"))
        self.assertEqual((d.op, d.reason), ("ADD", "model: two different children"))
        self.assertEqual(len(mm.current(U)), 2)

    def test_the_judge_can_close_more_alongside_a_single_valued_change(self):
        llm = FakeJudgeLLM()
        mm = judged(llm)
        veggie, cheese, tea = [d.result for d in mm.add(U, [
            Candidate("fact", "diet", "vegetarian", "Is vegetarian"),
            pref("likes", "cheese", "Likes cheese"), pref("likes", "tea", "Likes tea")])]
        llm.answer = lambda p: [{"item": 1, "op": "UPDATE", "ids": [veggie.id, cheese.id]}]
        [d] = mm.add(U, Candidate("fact", "diet", "vegan", "Is vegan"))
        self.assertEqual((d.op, d.target.id, [m.id for m in d.also]), ("SUPERSEDE", veggie.id, [cheese.id]))
        self.assertIn("one diet at a time", d.reason)
        self.assertEqual(sorted(m.text for m in mm.current(U)), ["Is vegan", "Likes tea"])
        self.assertEqual(mm.store.get(cheese.id).superseded_by, d.result.id)

    def test_not_asked_when_nothing_similar_is_stored(self):
        llm = FakeJudgeLLM(lambda p: [{"item": 1, "op": "ADD"}])
        clock = Clock(fixed_start=datetime(2026, 1, 1, tzinfo=timezone.utc))
        mm = MemoryManager(store(), clock=clock, judge=LLMJudge(llm))   # default threshold
        mm.add(U, pref("likes", "tea", "Likes tea"))                   # nothing stored yet
        mm.add(U, pref("likes", "jazz", "Listens to jazz records"))    # nothing similar
        self.assertEqual(llm.prompts, [])

    def test_one_call_for_everything_in_one_ingest(self):
        llm = FakeJudgeLLM(lambda p: [{"item": n, "op": "ADD"} for n in (1, 2, 3)])
        mm = judged(llm)
        mm.add(U, pref("likes", "tea", "Likes tea"))
        mm.add(U, [pref("likes", "coffee", "Likes coffee"), pref("likes", "jazz", "Likes jazz"),
                   pref("likes", "chess", "Plays chess")])
        self.assertEqual(len(llm.prompts), 1)   # the first add had nothing to compare with
        self.assertIn("Item 3 (new): Plays chess", llm.prompts[0])

    def test_failures_fall_back_to_the_rules(self):
        for llm in (FakeJudgeLLM(error=LLMError("timed out")), FakeJudgeLLM(raw="Sure! Let me think.")):
            mm = judged(llm)
            mm.add(U, pref("likes", "steak", "Loves steak"))
            with self.assertLogs("memwise", "WARNING"):
                [d] = mm.add(U, pref("diet", "vegetarian", "Went vegetarian", kind="fact"))
            self.assertEqual(d.op, "ADD")

    def test_the_model_is_asked_outside_the_lock(self):
        llm = FakeJudgeLLM(lambda p: [{"item": 1, "op": "ADD"}])
        mm = judged(llm)
        mm.add(U, pref("likes", "tea", "Likes tea"))
        got = []

        def another_thread_takes_the_lock():
            t = threading.Thread(target=lambda: got.append(mm._lock.acquire(timeout=2) and mm._lock.release() is None))
            t.start()
            t.join()

        llm.during = another_thread_takes_the_lock
        mm.add(U, pref("likes", "coffee", "Likes coffee"))
        self.assertEqual(got, [True])

    def test_a_memory_that_changed_while_judging_is_left_alone(self):
        llm = FakeJudgeLLM()
        mm = judged(llm)
        [steak] = [d.result for d in mm.add(U, pref("likes", "steak", "Loves steak"))]
        llm.answer = lambda p: [{"item": 1, "op": "UPDATE", "ids": [steak.id]}]
        llm.during = lambda: mm.forget(U, steak.id)   # the user deletes it meanwhile
        [d] = mm.add(U, pref("diet", "vegetarian", "Went vegetarian", kind="fact"))
        self.assertEqual(d.op, "ADD")
        self.assertEqual([m.text for m in mm.history(U)], ["Went vegetarian"])

    def test_ingest_uses_the_judge_after_extraction(self):
        def llm(system, messages):
            if system == JUDGE_SYSTEM:
                steak = next(m for m in mm.current(U) if m.value == "steak")
                return json.dumps({"decisions": [{"item": 1, "op": "UPDATE", "ids": [steak.id]}]})
            return json.dumps({"memories": [{"kind": "fact", "predicate": "diet", "value": "vegetarian",
                                             "text": "Is vegetarian", "source_ref": "t0"}]})

        clock = Clock(fixed_start=datetime(2026, 1, 1, tzinfo=timezone.utc))
        mm = MemoryManager(store(), extractor=LLMExtractor(llm), clock=clock, judge=LLMJudge(llm),
                           config=MemoryConfig(judge_min_similarity=-1.0))
        mm.add(U, pref("likes", "steak", "Loves steak"))
        [d] = mm.ingest(U, [("t0", "I went vegetarian last month.")])
        self.assertEqual(d.op, "SUPERSEDE")
        self.assertEqual([m.value for m in mm.current(U)], ["vegetarian"])

    def test_verdicts_are_read_forgivingly(self):
        reply = 'Here you go:\n```json\n{"decisions": [{"item": 2, "op": "update", "id": "m3"},' \
                ' {"item": 1, "op": "none", "ids": ["m1"]}, {"item": 9, "op": "ADD"},' \
                ' {"item": 3, "op": "MERGE", "ids": ["m2"]}]}\n```'
        v = parse_verdicts(reply, 3)
        self.assertEqual([(x.op, x.ids) if x else None for x in v],
                         [("NONE", ["m1"]), ("UPDATE", ["m3"]), None])
        with self.assertRaises(ValueError):
            parse_verdicts("I can't decide.", 1)


class LibraryAPI(unittest.TestCase):
    """The extension points other products rely on."""

    def test_ingest_accepts_plain_strings(self):
        mm = manager()
        [d] = mm.ingest(U, ["I live in Berlin."])
        self.assertEqual((d.op, d.result.source_ref), ("ADD", "1"))

    def test_add_reconciles_candidates_you_build(self):
        mm = manager()
        mm.add(U, Candidate("fact", "lives_in", "Berlin", "Lives in Berlin"))
        [d] = mm.add(U, Candidate("fact", "lives_in", "Lisbon", "Lives in Lisbon"))
        self.assertEqual(d.op, "SUPERSEDE")

    def test_config_adds_a_single_valued_predicate(self):
        config = MemoryConfig()
        config.single_valued.add("timezone")
        mm = MemoryManager(store(), config=config)
        mm.add(U, Candidate("fact", "timezone", "CET", "Timezone is CET"))
        [d] = mm.add(U, Candidate("fact", "timezone", "PST", "Timezone is PST"))
        self.assertEqual(d.op, "SUPERSEDE")
        self.assertNotIn("timezone", MemoryConfig().single_valued)  # defaults stay untouched

    def test_custom_embedder_is_used_and_reembed_refreshes_vectors(self):
        class FixedEmbedder:
            def __init__(self, vec):
                self.vec, self.calls = vec, 0

            def embed(self, text):
                self.calls += 1
                return self.vec

        mm = MemoryManager(store(), embedder=FixedEmbedder([1.0, 0.0]))
        mm.ingest(U, ["I live in Berlin."])
        self.assertEqual(mm.current(U)[0].embedding, [1.0, 0.0])
        mm.embedder = FixedEmbedder([0.0, 1.0])
        self.assertEqual(mm.reembed(U), 1)
        self.assertEqual(mm.current(U)[0].embedding, [0.0, 1.0])

    def test_build_context_is_empty_when_there_is_nothing_to_say(self):
        mm = manager()
        self.assertEqual(mm.build_context(mm.recall(U, "hello")), "")

    def test_maintain_all_covers_every_user(self):
        mm = manager()
        mm.ingest("a", ["I'm traveling to Tokyo next week."])
        mm.ingest("b", ["I'm traveling to Rome tomorrow."])
        mm.clock.advance(20)
        self.assertEqual(sorted(mm.maintain_all()), ["a", "b"])

    def test_forget_all_removes_memories_and_log(self):
        mm = manager()
        mm.ingest(U, ["I live in Berlin.", "I love coffee."])
        mm.ingest("other", ["I live in Paris."])
        self.assertEqual(mm.forget_all(U), 2)
        self.assertEqual((mm.history(U), list(mm.store.logs(U))), ([], []))
        self.assertEqual(len(mm.current("other")), 1)


class Concurrency(unittest.TestCase):
    def test_threads_can_share_one_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = MemoryManager(MemoryStore(os.path.join(tmp, "m.db")))
            errors, added = [], []

            def worker(n):
                try:
                    for i in range(10):
                        for d in mm.ingest(f"user{n}", [f"I love hobby{n}x{i}."]):
                            if d.op in ("ADD", "SUPERSEDE"):   # both write a new memory
                                added.append(d.result.id)
                        mm.recall(f"user{n}", "what are my hobbies?")
                except Exception as e:  # pragma: no cover - surfaced below
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            stored = [m.id for u in mm.store.user_ids() for m in mm.history(u)]
            self.assertGreater(len(added), 70)            # a few may CONFIRM as near-duplicates
            self.assertEqual(sorted(stored), sorted(added))
            self.assertEqual(len(set(stored)), len(stored))  # no id handed out twice
            mm.store.close()

    def test_two_stores_on_one_file_never_reuse_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.db")
            a, b = MemoryStore(path), MemoryStore(path)  # like two worker processes
            ids = [s.next_id() for _ in range(20) for s in (a, b)]
            self.assertEqual(len(set(ids)), 40)
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
