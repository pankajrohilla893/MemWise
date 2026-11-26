"""Judge evals: what the optional LLMJudge adds, and what it must not break.

    python evals/judge.py               # same environment variables as evals/scenarios.py
    python evals/judge.py --no-judge    # the same scenarios with the rules alone, to compare
    python evals/judge.py --repeat 3

The rules can't see links between different predicates: "Went vegetarian" doesn't match
"Loves steak", and "Runs 10k every morning" only confirms "Runs 5k every morning". The first
ones need the judge to pass. The others check it stays careful: things that can be true
together are kept, a specific favourite doesn't erase a general preference, and a bad
experience doesn't erase a preference.
Run the general and product evals with `--judge` too, to check nothing else changes.
"""

from __future__ import annotations

import re

from scenarios import Run, main


def a_new_diet_ends_a_food_preference(t: Run):
    t.session("I love a good steak, it's my favourite dinner.", "I live in Leeds.")
    t.later(30)
    t.session("I went vegetarian last month.")
    t.check("vegetarian is current", t.find("vegetarian"), t.memory_dump(closed=True))
    t.check("steak is no longer current", not t.find("steak"), t.memory_dump(closed=True))
    t.check("steak is kept as history", t.closed("steak"))
    t.check("unrelated memories are untouched", t.find("leeds"), t.memory_dump(closed=True))


def one_change_can_end_several_memories(t: Run):
    t.session("I love steak.", "I put cheese on everything.", "I drink a lot of green tea.")
    t.later(30)
    t.session("I've gone fully vegan now.")
    t.check("vegan is current", t.find("vegan"), t.memory_dump(closed=True))
    t.check("steak and cheese are no longer current", not t.find("steak|cheese"), t.memory_dump(closed=True))
    t.check("green tea is still current", t.find(r"\btea"), t.memory_dump(closed=True))


def details_change(t: Run):
    t.session("I run 5k every morning before work.")
    t.later(40)
    t.session("I've built up to 10k runs every morning now.")
    t.check("the 10k run is current", t.find("10 ?k"), t.memory_dump(closed=True))
    old = [m for m in t.find(r"\b5 ?k") if not re.search(r"10 ?k", f"{m.value} {m.text}", re.I)]
    t.check("the 5k run is no longer current", not old, t.memory_dump(closed=True))


def work_changes(t: Run):
    t.session("I'm a backend engineer on the payments team.")
    t.later(60)
    t.session("I moved to the platform team last week.")
    t.check("the platform team is current", t.find("platform"), t.memory_dump(closed=True))
    # The new memory may say where they moved from ("from payments to platform"): that's fine.
    old = [m for m in t.find("payments") if "platform" not in f"{m.value} {m.text}".lower()]
    t.check("the payments team is no longer current", not old, t.memory_dump(closed=True))


def a_specific_favourite_keeps_the_general_preference(t: Run):
    t.session("I like Italian food.")
    t.later(20)
    t.session("Neapolitan pizza is my absolute favourite Italian food.")
    t.check("Neapolitan pizza is current", t.find("neapolitan"), t.memory_dump(closed=True))
    t.check("liking Italian food in general isn't lost", t.find("italian"), t.memory_dump(closed=True))


def things_that_add_up_are_kept(t: Run):
    t.session("I love tea.", "I go hiking most weekends.")
    t.later(30)
    t.session("I've started drinking coffee in the afternoons too.", "I also took up rock climbing.")
    for what in (r"\btea", "hik", "coffee", "climb"):
        t.check(f"{what} is current", t.find(what), t.memory_dump(closed=True))


def a_bad_experience_does_not_erase_a_preference(t: Run):
    t.session("I love Thai food.")
    t.later(10)
    t.session("Had a terrible Thai takeaway last night, way too oily.")
    t.check("still likes Thai food", t.find("thai", "likes") or
            any("thai" in m.text.lower() and m.kind == "preference" for m in t.current()),
            t.memory_dump(closed=True))


def stopping_in_other_words(t: Run):
    t.session("I go jogging every morning.")
    t.later(30)
    t.session("My knees gave out, so no more running for me.")
    t.check("jogging is no longer current", not t.find("jog"), t.memory_dump(closed=True))


JUDGE_SCENARIOS = [
    a_new_diet_ends_a_food_preference, one_change_can_end_several_memories, details_change,
    work_changes, a_specific_favourite_keeps_the_general_preference, things_that_add_up_are_kept,
    a_bad_experience_does_not_erase_a_preference, stopping_in_other_words,
]

if __name__ == "__main__":
    main(JUDGE_SCENARIOS, __doc__, judge=True)
