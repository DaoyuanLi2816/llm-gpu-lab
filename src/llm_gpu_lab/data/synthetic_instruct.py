"""Deterministic synthetic instruction-tuning dataset.

The dataset is generated procedurally and covers five task families:

1. arithmetic        — "What is 3 + 4?" → "7"
2. JSON formatting   — "Return a JSON with keys ..." → strict JSON string
3. rewriting         — "Rewrite in past tense" → rewritten sentence
4. summarization     — "Summarize: ..." → one-sentence summary
5. classification    — "Classify the sentiment: ..." → "positive"/"negative"

The goal isn't to teach a small model to be smart — it's to exercise the SFT
plumbing on a real instruction signal whose correctness can be checked by
simple string operations during evaluation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class InstructExample:
    instruction: str
    input: str
    output: str
    task: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "task": self.task,
        }

    def to_chat(self) -> List[Dict[str, str]]:
        prompt = self.instruction if not self.input else f"{self.instruction}\n\n{self.input}"
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": self.output},
        ]


def _arithmetic(rng: random.Random) -> InstructExample:
    op = rng.choice(["+", "-", "*"])
    a = rng.randint(0, 9)
    b = rng.randint(0, 9)
    if op == "+":
        ans = a + b
        op_word = "plus"
    elif op == "-":
        if b > a:
            a, b = b, a
        ans = a - b
        op_word = "minus"
    else:
        ans = a * b
        op_word = "times"
    instr = f"What is {a} {op_word} {b}?"
    return InstructExample(
        instruction=instr,
        input="",
        output=str(ans),
        task="arithmetic",
    )


_KEYS = ["name", "color", "count", "city", "fruit"]
_FRUITS = ["apple", "banana", "pear", "mango", "kiwi", "lime"]
_CITIES = ["Lyon", "Osaka", "Quito", "Hobart", "Tromsø", "Cusco"]
_NAMES = ["Iris", "Leo", "Maya", "Oren", "Tara", "Quinn"]
_COLORS = ["red", "blue", "green", "yellow", "purple"]


def _json_format(rng: random.Random) -> InstructExample:
    chosen_keys = rng.sample(_KEYS, k=rng.randint(2, 4))
    payload: Dict[str, object] = {}
    for k in chosen_keys:
        if k == "name":
            payload[k] = rng.choice(_NAMES)
        elif k == "color":
            payload[k] = rng.choice(_COLORS)
        elif k == "count":
            payload[k] = rng.randint(1, 9)
        elif k == "city":
            payload[k] = rng.choice(_CITIES)
        elif k == "fruit":
            payload[k] = rng.choice(_FRUITS)
    parts = [f"{k}={payload[k]}" for k in chosen_keys]
    instr = "Return a single-line JSON object with these key=value pairs."
    inp = "; ".join(parts)
    out = json.dumps(payload, separators=(", ", ": "))
    return InstructExample(instruction=instr, input=inp, output=out, task="json_format")


_SUBJECT = ["She", "He", "The cat", "The teacher", "My friend"]
_VERB_PRES = ["walks", "writes", "paints", "reads", "bakes"]
_VERB_PAST = {"walks": "walked", "writes": "wrote", "paints": "painted", "reads": "read", "bakes": "baked"}
_OBJECT = ["a letter", "a picture", "a book", "some bread", "a story"]


def _rewrite_past(rng: random.Random) -> InstructExample:
    subj = rng.choice(_SUBJECT)
    verb = rng.choice(_VERB_PRES)
    obj = rng.choice(_OBJECT)
    sentence = f"{subj} {verb} {obj}."
    rewritten = f"{subj} {_VERB_PAST[verb]} {obj}."
    return InstructExample(
        instruction="Rewrite the following sentence in the past tense.",
        input=sentence,
        output=rewritten,
        task="rewrite_past_tense",
    )


_PARAGRAPHS = [
    (
        "A small fox lived in the woods. Every morning the fox visited the river to drink. "
        "It often saw a kind rabbit there.",
        "A small fox visits the river each morning and meets a kind rabbit.",
    ),
    (
        "The lighthouse keeper kept a journal. She wrote down the weather and the names of "
        "every passing ship.",
        "A lighthouse keeper records the weather and passing ships in her journal.",
    ),
    (
        "The librarian arranged the new books by topic. She placed the children's books on "
        "the lower shelves so they were easy to reach.",
        "The librarian sorted new books by topic and put children's books on lower shelves.",
    ),
    (
        "A young inventor built a small kite from paper and string. He launched it from a hill "
        "near the orchard at sunset.",
        "A young inventor built a paper kite and flew it from a hill at sunset.",
    ),
    (
        "Two friends rebuilt an old wooden bridge across the stream. They worked all weekend "
        "until the bridge was safe again.",
        "Two friends spent the weekend repairing an old wooden bridge until it was safe.",
    ),
]


def _summarize(rng: random.Random) -> InstructExample:
    paragraph, summary = rng.choice(_PARAGRAPHS)
    return InstructExample(
        instruction="Summarize the paragraph in one short sentence.",
        input=paragraph,
        output=summary,
        task="summarize",
    )


_SENTIMENT_POS = [
    "I love this book, it is wonderful.",
    "What a great day, everything went perfectly.",
    "The food was delicious and the staff were kind.",
    "I am so happy with the new design.",
    "This is the best gift I have ever received.",
]

_SENTIMENT_NEG = [
    "I dislike this book, it is boring.",
    "What a terrible day, everything went wrong.",
    "The food was cold and the staff were rude.",
    "I am disappointed with the new design.",
    "This is the worst gift I have ever received.",
]


def _classify_sentiment(rng: random.Random) -> InstructExample:
    if rng.random() < 0.5:
        text = rng.choice(_SENTIMENT_POS)
        label = "positive"
    else:
        text = rng.choice(_SENTIMENT_NEG)
        label = "negative"
    return InstructExample(
        instruction="Classify the sentiment of the following sentence as 'positive' or 'negative'.",
        input=text,
        output=label,
        task="sentiment",
    )


_BUILDERS = [
    _arithmetic,
    _json_format,
    _rewrite_past,
    _summarize,
    _classify_sentiment,
]


def build_synthetic_instruct_dataset(
    n_examples: int = 64,
    seed: int = 1337,
) -> List[InstructExample]:
    rng = random.Random(seed)
    examples: List[InstructExample] = []
    # Round-robin over task families to keep the mix balanced.
    for i in range(n_examples):
        builder = _BUILDERS[i % len(_BUILDERS)]
        examples.append(builder(rng))
    rng.shuffle(examples)
    return examples
