"""Deterministic synthetic mini-corpus.

The corpus mixes three flavours that are simple enough for a tiny GPT to learn
something coherent, but expressive enough that next-token loss has meaningful
variance:

* short fairy-tale style fragments (mirrors public TinyStories);
* simple arithmetic statements ("two plus three equals five");
* templated factoids ("the color of the sky is blue").

It is entirely procedural — no external LLM, no scraped text. The exact text is
reproducible from a seed so smoke runs are deterministic.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List, Sequence

_NAMES = [
    "Maya", "Leo", "Iris", "Oren", "Talia", "Bran", "Cora", "Dax",
    "Elara", "Finn", "Greta", "Hugo", "Inka", "Jules", "Kai", "Luna",
    "Milo", "Nia", "Otto", "Pia", "Quinn", "Rex", "Sage", "Tara",
]

_PLACES = [
    "forest", "village", "river", "mountain", "garden", "harbor",
    "library", "workshop", "orchard", "meadow", "valley", "lighthouse",
]

_OBJECTS = [
    "lantern", "key", "compass", "letter", "telescope", "feather",
    "ribbon", "map", "shell", "flute", "kite", "coin",
]

_VERBS = [
    "found", "lost", "carried", "shared", "drew", "built",
    "repaired", "discovered", "painted", "planted", "borrowed", "returned",
]

_ADJ = [
    "small", "bright", "quiet", "rusty", "shiny", "patient",
    "clever", "wooden", "silver", "ancient", "warm", "tiny",
]

_FACT_TEMPLATES = [
    "the color of the {obj} is {color}.",
    "a {obj} can be used to {verb}.",
    "the {place} is full of {plural}.",
    "every morning, {name} visits the {place}.",
    "the {name1} and the {name2} are good friends.",
]

_COLORS = ["red", "blue", "green", "yellow", "purple", "silver", "gold", "white"]
_PLURALS = ["birds", "leaves", "stones", "flowers", "songs", "stories", "lights"]

_NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
]


def _arithmetic_sentence(rng: random.Random) -> str:
    # Keep the result inside the [0, 10] range of `_NUMBER_WORDS`.
    op = rng.choice(["+", "-"])
    if op == "+":
        a = rng.randint(0, 10)
        b = rng.randint(0, 10 - a)
        c = a + b
        return f"{_NUMBER_WORDS[a]} plus {_NUMBER_WORDS[b]} equals {_NUMBER_WORDS[c]}."
    a = rng.randint(0, 10)
    b = rng.randint(0, a)
    c = a - b
    return f"{_NUMBER_WORDS[a]} minus {_NUMBER_WORDS[b]} equals {_NUMBER_WORDS[c]}."


def _factoid(rng: random.Random) -> str:
    template = rng.choice(_FACT_TEMPLATES)
    return template.format(
        obj=rng.choice(_OBJECTS),
        color=rng.choice(_COLORS),
        verb=rng.choice(_VERBS),
        place=rng.choice(_PLACES),
        plural=rng.choice(_PLURALS),
        name=rng.choice(_NAMES),
        name1=rng.choice(_NAMES),
        name2=rng.choice(_NAMES),
    ).capitalize()


def _story(rng: random.Random) -> str:
    name = rng.choice(_NAMES)
    friend = rng.choice([n for n in _NAMES if n != name])
    place = rng.choice(_PLACES)
    obj = rng.choice(_OBJECTS)
    adj = rng.choice(_ADJ)
    verb = rng.choice(_VERBS)
    color = rng.choice(_COLORS)
    return (
        f"{name} walked through the {adj} {place} with {friend}. "
        f"They {verb} a {color} {obj}. "
        f"It made them happy. "
        f"{name} said, \"Let us share it with everyone.\" "
        f"And so they did."
    )


def build_tiny_corpus(n_examples: int = 4000, seed: int = 1337) -> List[str]:
    """Build a list of newline-separated documents.

    `n_examples` documents are generated. The mix is roughly:
    * 60% short stories
    * 25% factoids
    * 15% arithmetic
    """
    rng = random.Random(seed)
    out: List[str] = []
    for _ in range(n_examples):
        roll = rng.random()
        if roll < 0.60:
            out.append(_story(rng))
        elif roll < 0.85:
            out.append(_factoid(rng))
        else:
            out.append(_arithmetic_sentence(rng))
    return out


def iter_tiny_corpus_lines(n_examples: int = 4000, seed: int = 1337) -> Iterator[str]:
    yield from build_tiny_corpus(n_examples=n_examples, seed=seed)


def write_corpus_to_file(path: str | Path, lines: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
