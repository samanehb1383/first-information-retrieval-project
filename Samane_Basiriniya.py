#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text preprocessing for Information Retrieval
============================================
Course  : Foundations of Information Retrieval and Web Search
Part 2  : Boolean retrieval - building the inverted index (tokenizer + linguistic modules)
Author  : Samane Basiriniya

Pipeline (always in this order):
    1. input         - built-in test texts, a file, the command line, stdin or the keyboard
    2. tokenization  - punctuation is separated from words and dropped (dots inside
                       abbreviations are kept), hyphenated words stay one token, the
                       possessive 's / ’s is removed
    3a. normalization - lower-casing; U.S.A. and USA both become "usa"
    3b. stop words    - one fixed English list (the 198-word NLTK list, embedded below)
    3c. stemming      - Porter stemmer, implemented from scratch in this file
Evaluation:
    Precision and Recall of each text's terms against a gold standard that was built
    by hand (GOLD_STANDARD), computed on sets of distinct terms.

Usage:
    python3 Samane_Basiriniya.py                  # the 3 test texts + evaluation
    python3 Samane_Basiriniya.py -t "Any text."   # text(s) from the command line
    python3 Samane_Basiriniya.py -f texts.txt     # a file, one text per non-empty line
    python3 Samane_Basiriniya.py -f -             # read the texts from stdin
    python3 Samane_Basiriniya.py -i               # type a text interactively
    python3 Samane_Basiriniya.py --ablation       # error analysis: disable one rule at a time
    python3 Samane_Basiriniya.py --help           # all options

Only the Python standard library is used (Python 3.8 or newer).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import textwrap
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache


# ═══════════════════════════════════════════════════════════════════════════════════
#  TEST TEXTS AND THE HAND-MADE GOLD STANDARD
# ═══════════════════════════════════════════════════════════════════════════════════

TEST_TEXTS = {
    "text1": "Friends, Romans, countrymen, lend me your ears;",
    "text2": ("John's new U.S.A. laptop is a state-of-the-art device. "
              "The price is high, but the quality is amazing!"),
    "text3": ("The authorization for this authorized project is pending. "
              "He authorizes the payments regularly."),
}

# Written by hand, stage by stage, from the rules of the assignment - NOT produced by
# running this program. Stem shapes follow the Porter algorithm (derived step by step;
# the derivations are in the report). "terms" is the gold standard used for Precision
# and Recall; the earlier stages are used to find which stage an error comes from.
GOLD_STANDARD = {
    "text1": {
        "tokens":     ["Friends", "Romans", "countrymen", "lend", "me", "your", "ears"],
        "normalized": ["friends", "romans", "countrymen", "lend", "me", "your", "ears"],
        "no_stop":    ["friends", "romans", "countrymen", "lend", "ears"],
        "terms":      ["friend", "roman", "countrymen", "lend", "ear"],
    },
    "text2": {
        "tokens":     ["John", "new", "U.S.A.", "laptop", "is", "a", "state-of-the-art",
                       "device", "The", "price", "is", "high", "but", "the", "quality",
                       "is", "amazing"],
        "normalized": ["john", "new", "usa", "laptop", "is", "a", "state-of-the-art",
                       "device", "the", "price", "is", "high", "but", "the", "quality",
                       "is", "amazing"],
        "no_stop":    ["john", "new", "usa", "laptop", "state-of-the-art", "device",
                       "price", "high", "quality", "amazing"],
        "terms":      ["john", "new", "usa", "laptop", "state-of-the-art", "devic",
                       "price", "high", "qualiti", "amaz"],
    },
    "text3": {
        "tokens":     ["The", "authorization", "for", "this", "authorized", "project", "is",
                       "pending", "He", "authorizes", "the", "payments", "regularly"],
        "normalized": ["the", "authorization", "for", "this", "authorized", "project", "is",
                       "pending", "he", "authorizes", "the", "payments", "regularly"],
        "no_stop":    ["authorization", "authorized", "project", "pending", "authorizes",
                       "payments", "regularly"],
        "terms":      ["author", "author", "project", "pend", "author", "payment",
                       "regularli"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION - every rule can be switched off (used only by the ablation study)
# ═══════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Config:
    """The default values ARE the submitted pipeline. The ablation study (--ablation)
    flips exactly one switch at a time to show which stage each kind of error comes from."""
    split_punctuation: bool = True    # 2: separate punctuation from words, then drop it
    keep_abbreviations: bool = True   # 2: U.S.A. stays one token
    keep_hyphenated: bool = True      # 2: state-of-the-art stays one token
    strip_possessive: bool = True     # 2: John's / John’s -> John
    lowercase: bool = True            # 3a
    unify_abbreviations: bool = True  # 3a: u.s.a. -> usa, so it matches USA
    remove_stop_words: bool = True    # 3b
    stemmer: str = "porter"           # 3c: "porter", "s" (plural-only S-stemmer) or "none"
    stop_before_stem: bool = True     # 3b before 3c, as the assignment requires


SUBMITTED = Config()


# ═══════════════════════════════════════════════════════════════════════════════════
#  STEP 2 - TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════════════

_LETTER = r"[^\W\d_]"   # one letter of any alphabet
_ALNUM = r"[^\W_]"      # one letter or digit

# Abbreviation: single letters joined by dots (U.S.A.  U.S.A  e.g.  i.e.), optionally
# followed by a possessive ('s). It may not start or end in the middle of a word.
_ABBREVIATION = (rf"(?<!{_ALNUM}){_LETTER}(?:\.{_LETTER})+\.?"
                 rf"(?:['’][sS])?(?!{_ALNUM})")
# Numbers with a decimal point or thousands separators stay whole: 3.14  1,000,000
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+"
# Word: letters/digits, with inner hyphens or apostrophes: state-of-the-art  John's  don't
_WORD = rf"{_ALNUM}+(?:[-'’]{_ALNUM}+)*"
_WORD_NO_HYPHEN = rf"{_ALNUM}+(?:['’]{_ALNUM}+)*"
# Anything else that is not a space is ONE punctuation unit: . , ; : ! ? ( ) " - …
_PUNCTUATION = r"[^\w\s]|_"

_PUNCTUATION_RE = re.compile(rf"(?:{_PUNCTUATION})+")
_POSSESSIVE_RE = re.compile(r"^(.+?)['’][sS]$")


@lru_cache(maxsize=None)
def _unit_regex(keep_abbreviations: bool, keep_hyphenated: bool) -> re.Pattern:
    """Alternatives are tried left to right, so abbreviations win over words and
    punctuation, and numbers win over words."""
    parts = [_ABBREVIATION] if keep_abbreviations else []
    parts += [_NUMBER, _WORD if keep_hyphenated else _WORD_NO_HYPHEN, _PUNCTUATION]
    return re.compile("|".join(f"(?:{p})" for p in parts))


def tokenize(text: str, cfg: Config = SUBMITTED, trace: dict | None = None) -> list[str]:
    """Step 2. Splits the text into units (words AND punctuation marks, separated from each
    other), removes the possessive 's, then drops the punctuation units.
    If a dict is passed as `trace`, the intermediate results are stored in it."""
    if cfg.split_punctuation:
        units = _unit_regex(cfg.keep_abbreviations, cfg.keep_hyphenated).findall(text)
    else:                                   # naive baseline, only for the ablation study
        units = text.split()

    tokens, possessives, punctuation = [], [], []
    for unit in units:
        if cfg.split_punctuation and _PUNCTUATION_RE.fullmatch(unit):
            punctuation.append(unit)        # separated from the words, now removed
            continue
        if cfg.strip_possessive:
            match = _POSSESSIVE_RE.match(unit)
            if match:
                possessives.append((unit, match.group(1)))
                unit = match.group(1)
        tokens.append(unit)

    if trace is not None:
        trace.update(units=units, possessives=possessives, punctuation=punctuation)
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════════
#  STEP 3a - NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════════

_ABBREVIATION_TOKEN_RE = re.compile(rf"{_LETTER}(?:\.{_LETTER})+\.?")


def normalize(token: str, cfg: Config = SUBMITTED) -> str:
    """Maps different spellings of a word to one equivalence class:
    - Unicode compatibility forms (NFKC) and curly apostrophes are unified;
    - every letter becomes lower case;
    - an abbreviation loses its dots, so U.S.A. -> usa matches USA -> usa;
    - accents are removed (café -> cafe), as users often type words without them."""
    t = unicodedata.normalize("NFKC", token).replace("’", "'").replace("‘", "'")
    if cfg.lowercase:
        t = t.lower()
    if cfg.unify_abbreviations and _ABBREVIATION_TOKEN_RE.fullmatch(t):
        t = t.replace(".", "")
    decomposed = unicodedata.normalize("NFD", t)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch)))


def normalize_all(tokens: list[str], cfg: Config = SUBMITTED) -> list[str]:
    return [normalize(t, cfg) for t in tokens]


# ═══════════════════════════════════════════════════════════════════════════════════
#  STEP 3b - STOP-WORD REMOVAL
# ═══════════════════════════════════════════════════════════════════════════════════

# A fixed list: the English stop list of NLTK (corpus "stopwords", 198 words). It is
# copied here so that no download is needed and the list cannot change between runs.
STOP_WORDS = frozenset("""
a about above after again against ain all am an and any are aren aren't as at be because
been before being below between both but by can couldn couldn't d did didn didn't do does
doesn doesn't doing don don't down during each few for from further had hadn hadn't has
hasn hasn't have haven haven't having he he'd he'll her here hers herself he's him himself
his how i i'd if i'll i'm in into is isn isn't it it'd it'll it's its itself i've just ll m
ma me mightn mightn't more most mustn mustn't my myself needn needn't no nor not now o of
off on once only or other our ours ourselves out over own re s same shan shan't she she'd
she'll she's should shouldn shouldn't should've so some such t than that that'll the their
theirs them themselves then there these they they'd they'll they're they've this those
through to too under until up ve very was wasn wasn't we we'd we'll we're were weren
weren't we've what when where which while who whom why will with won won't wouldn wouldn't
y you you'd you'll your you're yours yourself yourselves you've
""".split())


def remove_stop_words(tokens: list[str], cfg: Config = SUBMITTED) -> list[str]:
    """Step 3b - runs BEFORE stemming: a stemmed stop word may no longer be found in the
    list (this -> thi), see the ablation study."""
    if not cfg.remove_stop_words:
        return list(tokens)
    return [t for t in tokens if t not in STOP_WORDS]


# ═══════════════════════════════════════════════════════════════════════════════════
#  STEP 3c - STEMMING: the Porter stemmer, written from scratch
# ═══════════════════════════════════════════════════════════════════════════════════


class PorterStemmer:
    """Porter, M.F. (1980) "An algorithm for suffix stripping", Program 14(3): 130-137,
    in the reference form published by Martin Porter (tartarus.org/martin/PorterStemmer),
    which differs from the paper in three small points:
        * words of one or two letters are not stemmed;
        * step 2 uses (m>0) BLI -> BLE instead of (m>0) ABLI -> ABLE;
        * step 2 has the extra rule (m>0) LOGI -> LOG.
    Notation: C = run of consonants, V = run of vowels; every word has the form
    [C](VC)^m[V], and m is called the measure of the word or word part."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    # ---------- helper predicates --------------------------------------------------

    @staticmethod
    def _is_consonant(word: str, i: int) -> bool:
        """Not a, e, i, o, u - and y only when it is not preceded by a consonant."""
        ch = word[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return i == 0 or not PorterStemmer._is_consonant(word, i - 1)
        return True

    def _measure(self, stem: str) -> int:
        """m = number of VC sequences, e.g. tree 0, trouble 1, troubles 2."""
        m, previous_is_vowel = 0, False
        for i in range(len(stem)):
            consonant = self._is_consonant(stem, i)
            if consonant and previous_is_vowel:
                m += 1
            previous_is_vowel = not consonant
        return m

    def _has_vowel(self, stem: str) -> bool:                 # *v*
        return any(not self._is_consonant(stem, i) for i in range(len(stem)))

    def _ends_double_consonant(self, word: str) -> bool:     # *d
        return (len(word) >= 2 and word[-1] == word[-2]
                and self._is_consonant(word, len(word) - 1))

    def _ends_cvc(self, word: str) -> bool:                  # *o
        """consonant-vowel-consonant, the last one not w, x or y (hop, but not snow)."""
        return (len(word) >= 3
                and self._is_consonant(word, len(word) - 3)
                and not self._is_consonant(word, len(word) - 2)
                and self._is_consonant(word, len(word) - 1)
                and word[-1] not in "wxy")

    def _replace_first(self, word: str, rules, condition) -> str:
        """Finds the FIRST rule whose suffix the word ends with. If the condition holds
        for the remaining stem the suffix is replaced; either way no other rule is tried."""
        for suffix, replacement in rules:
            if word.endswith(suffix):
                stem = word[: len(word) - len(suffix)]
                return stem + replacement if condition(stem) else word
        return word

    # ---------- the five steps ---------------------------------------------------

    def _step1a(self, word: str) -> str:      # plurals: caresses->caress, ponies->poni
        return self._replace_first(
            word, [("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")], lambda s: True)

    def _step1b(self, word: str) -> str:      # -ed / -ing: agreed->agree, hopping->hop
        if word.endswith("eed"):
            stem = word[:-3]
            return stem + "ee" if self._measure(stem) > 0 else word
        for suffix in ("ed", "ing"):
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                return self._step1b_tidy(stem) if self._has_vowel(stem) else word
        return word

    def _step1b_tidy(self, stem: str) -> str:
        if stem.endswith(("at", "bl", "iz")):                   # conflat(ed) -> conflate
            return stem + "e"
        if self._ends_double_consonant(stem) and stem[-1] not in "lsz":
            return stem[:-1]                                    # hopp(ing) -> hop
        if self._measure(stem) == 1 and self._ends_cvc(stem):
            return stem + "e"                                   # fil(ing) -> file
        return stem

    def _step1c(self, word: str) -> str:      # happy->happi, sky->sky
        if word.endswith("y") and self._has_vowel(word[:-1]):
            return word[:-1] + "i"
        return word

    _STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
              ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"),
              ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
              ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
              ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
              ("logi", "log")]

    def _step2(self, word: str) -> str:       # relational->relate, authorization->authorize
        return self._replace_first(word, self._STEP2, lambda s: self._measure(s) > 0)

    _STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
              ("ical", "ic"), ("ful", ""), ("ness", "")]

    def _step3(self, word: str) -> str:       # electrical->electric, goodness->good
        return self._replace_first(word, self._STEP3, lambda s: self._measure(s) > 0)

    _STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
              "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"]

    def _step4(self, word: str) -> str:       # revival->reviv, adoption->adopt
        def condition(stem: str) -> bool:
            if word.endswith("ion") and stem[-1:] not in ("s", "t"):
                return False                  # (m>1 and (*S or *T)) ION ->
            return self._measure(stem) > 1
        return self._replace_first(word, [(s, "") for s in self._STEP4], condition)

    def _step5(self, word: str) -> str:       # probate->probat, rate->rate, controll->control
        if word.endswith("e"):
            stem = word[:-1]
            m = self._measure(stem)
            if m > 1 or (m == 1 and not self._ends_cvc(stem)):
                word = stem
        if self._measure(word) > 1 and self._ends_double_consonant(word) and word.endswith("l"):
            word = word[:-1]
        return word

    # ---------- public -----------------------------------------------------------

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word
        if word not in self._cache:
            result = word
            for step in (self._step1a, self._step1b, self._step1c,
                         self._step2, self._step3, self._step4, self._step5):
                result = step(result)
            self._cache[word] = result
        return self._cache[word]


PORTER = PorterStemmer()


def s_stem(word: str) -> str:
    """The 'S-stemmer' (Harman, 1991): removes plural endings only. It is NOT used by the
    pipeline; the ablation study compares it with Porter."""
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        return word[:-1]
    if word.endswith("s") and not word.endswith(("us", "ss")):
        return word[:-1]
    return word


def stem_all(tokens: list[str], cfg: Config = SUBMITTED) -> list[str]:
    if cfg.stemmer == "porter":
        return [PORTER.stem(t) for t in tokens]
    if cfg.stemmer == "s":
        return [s_stem(t) for t in tokens]
    return list(tokens)


# ═══════════════════════════════════════════════════════════════════════════════════
#  THE WHOLE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════════


@dataclass
class Trace:
    """Everything the pipeline produced for one text, stage by stage."""
    text: str
    units: list            # words and punctuation, separated
    possessives: list      # (before, after) pairs
    punctuation: list      # punctuation units that were dropped
    tokens: list           # result of step 2
    normalized: list       # result of step 3a
    no_stop: list | None   # result of step 3b (None if stemming ran first)
    terms: list            # result of step 3c = the final output


def preprocess(text: str, cfg: Config = SUBMITTED) -> Trace:
    info: dict = {}
    tokens = tokenize(text, cfg, info)
    normalized = normalize_all(tokens, cfg)
    if cfg.stop_before_stem:
        no_stop = remove_stop_words(normalized, cfg)
        terms = stem_all(no_stop, cfg)
    else:                  # wrong order - only used to show why the assignment forbids it
        no_stop = None
        terms = remove_stop_words(stem_all(normalized, cfg), cfg)
    return Trace(text, info["units"], info["possessives"], info["punctuation"],
                 tokens, normalized, no_stop, terms)


# ═══════════════════════════════════════════════════════════════════════════════════
#  EVALUATION - Precision and Recall against the gold standard
# ═══════════════════════════════════════════════════════════════════════════════════


@dataclass
class Metrics:
    n_output: int          # |O|  distinct terms produced by the program
    n_gold: int            # |G|  distinct terms of the gold standard
    n_common: int          # |O ∩ G|
    precision: float
    recall: float
    f1: float
    extra: list            # O − G : produced but not in the gold standard
    missed: list           # G − O : in the gold standard but not produced
    same_sequence: bool    # identical lists (order and repetitions included)


def _unique_in_order(items) -> list:
    return list(dict.fromkeys(items))


def evaluate(output: list[str], gold: list[str]) -> Metrics:
    """Precision = |O ∩ G| / |O|,  Recall = |O ∩ G| / |G|, on sets of DISTINCT terms
    (repetitions do not count). An empty denominator gives 0, as the assignment says."""
    o_set, g_set = set(output), set(gold)
    common = len(o_set & g_set)
    precision = common / len(o_set) if o_set else 0.0
    recall = common / len(g_set) if g_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(len(o_set), len(g_set), common, precision, recall, f1,
                   _unique_in_order(t for t in output if t not in g_set),
                   _unique_in_order(t for t in gold if t not in o_set),
                   list(output) == list(gold))


STAGES = ("tokenization", "normalization", "stop words", "stemming")


def stage_check(text: str, gold: dict, cfg: Config = SUBMITTED) -> dict:
    """Error attribution. Each stage is run on the GOLD output of the previous stage and
    compared with its own gold output, so a mistake is charged to the stage that makes
    it instead of also showing up in every later stage."""
    produced = {
        "tokenization": tokenize(text, cfg),
        "normalization": normalize_all(gold["tokens"], cfg),
        "stop words": remove_stop_words(gold["normalized"], cfg),
        "stemming": stem_all(gold["no_stop"], cfg),
    }
    expected = {"tokenization": gold["tokens"], "normalization": gold["normalized"],
                "stop words": gold["no_stop"], "stemming": gold["terms"]}
    return {stage: evaluate(produced[stage], expected[stage]) for stage in STAGES}


def error_source(text: str, gold: dict, cfg: Config) -> list[str]:
    """Stages that make errors on their own; if none does but the final output is still
    wrong, the stages are fine and the problem is the order in which they run."""
    failing = [s for s, m in stage_check(text, gold, cfg).items() if not m.same_sequence]
    if not failing and preprocess(text, cfg).terms != gold["terms"]:
        failing = ["order of stages"]
    return failing


# ═══════════════════════════════════════════════════════════════════════════════════
#  CONSOLE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════════


class Out:
    color = False
    width = 100

    @classmethod
    def setup(cls, no_color: bool) -> None:
        try:                                    # Windows consoles: UTF-8 and ANSI colours
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
        if os.name == "nt":
            os.system("")
        wants = (sys.stdout.isatty() or "PYCHARM_HOSTED" in os.environ
                 or "FORCE_COLOR" in os.environ)
        cls.color = wants and not no_color and "NO_COLOR" not in os.environ
        cls.width = max(72, min(120, shutil.get_terminal_size((100, 24)).columns))

    @classmethod
    def c(cls, text: str, *codes: str) -> str:
        if not cls.color or not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"


BOLD, DIM, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN = "1", "2", "31", "32", "33", "34", "35", "36"
LABEL_WIDTH = 22


def rule(char: str = "─", color: str = DIM) -> None:
    print(Out.c(char * Out.width, color))


def _pack(items: list[str], sep: str, width: int) -> list[str]:
    """Greedy line filling that never splits an item; a separator ends the broken line."""
    lines, current = [], ""
    for item in items:
        candidate = item if not current else current + sep + item
        if current and len(candidate) > width:
            lines.append(current + sep.rstrip())
            current = item
        else:
            current = candidate
    return lines + [current]


def line(label: str, value, label_color: str = YELLOW, value_color: str = "",
         indent: int = 1, label_width: int = LABEL_WIDTH) -> None:
    """Prints `label  value`, wrapping a long value under itself. `value` is a string or
    a list of items (shown separated by ' | '); items and words are never split."""
    pad = " " * indent
    room = Out.width - indent - label_width
    if isinstance(value, list):
        wrapped = _pack(value, " | ", room) if value else ["(none)"]
    else:
        wrapped = textwrap.wrap(value, width=room, break_long_words=False,
                                break_on_hyphens=False) or [""]
    print(pad + Out.c(label.ljust(label_width), label_color)
          + (Out.c(wrapped[0], value_color) if value_color else wrapped[0]))
    for rest in wrapped[1:]:
        print(pad + " " * label_width + (Out.c(rest, value_color) if value_color else rest))


def as_json(items) -> str:
    return json.dumps(list(items), ensure_ascii=False)


def to_json(outputs: dict) -> str:
    """{"text1": [...], ...} with one text per line, as suggested in the assignment."""
    body = ",\n".join(f"  {json.dumps(k, ensure_ascii=False)}: {as_json(v)}"
                      for k, v in outputs.items())
    return "{\n" + body + "\n}\n"


def changes(before: list[str], after: list[str]) -> str:
    pairs = _unique_in_order(f"{b} → {a}" for b, a in zip(before, after) if b != a)
    return ",  ".join(pairs) if pairs else "(none)"


def print_trace(text_id: str, tr: Trace) -> None:
    title = text_id.replace("text", "TEXT ", 1) if text_id.startswith("text") else text_id
    print()
    rule("═", CYAN)
    print(Out.c(f" {title}", BOLD, CYAN))
    rule("═", CYAN)
    line("[1] Input", tr.text, value_color=BOLD)
    print(Out.c(" [2] Tokenization", YELLOW))
    line("    units (%d)" % len(tr.units), tr.units, YELLOW)
    line("    possessive 's", ",  ".join(f"{a} → {b}" for a, b in tr.possessives) or "(none)",
         YELLOW)
    line("    punctuation", (" ".join(tr.punctuation) + "   (removed)")
         if tr.punctuation else "(none)", YELLOW)
    line("    tokens (%d)" % len(tr.tokens), tr.tokens, YELLOW)
    line("[3a] Normalization", tr.normalized)
    line("    changed", changes(tr.tokens, tr.normalized), YELLOW, DIM)
    line("[3b] Stop words", tr.no_stop)
    removed = [t for t in tr.normalized if t in STOP_WORDS]
    line("    removed (%d)" % len(removed), ", ".join(removed) or "(none)", YELLOW, DIM)
    line("[3c] Stemming", tr.terms)
    line("    changed", changes(tr.no_stop, tr.terms), YELLOW, DIM)
    print()
    distinct = len(set(tr.terms))
    line(f"FINAL OUTPUT ({len(tr.terms)})", as_json(tr.terms), BOLD + ";" + GREEN, BOLD)
    if distinct != len(tr.terms):
        line("", f"({len(tr.terms)} terms, {distinct} distinct)", YELLOW, DIM)


def score_color(x: float) -> str:
    return GREEN if x >= 0.999 else (YELLOW if x >= 0.7 else RED)


def print_metrics(m: Metrics, gold_terms: list[str], text: str, gold: dict | None) -> None:
    line(f"GOLD STANDARD ({len(gold_terms)})", as_json(gold_terms), BOLD + ";" + MAGENTA)
    line("Precision", f"|O ∩ G| / |O| = {m.n_common} / {m.n_output} = {m.precision:.3f}",
         BOLD, BOLD + ";" + score_color(m.precision))
    line("Recall", f"|O ∩ G| / |G| = {m.n_common} / {m.n_gold} = {m.recall:.3f}",
         BOLD, BOLD + ";" + score_color(m.recall))
    line("Extra   (O − G)", ", ".join(m.extra) or "none", YELLOW, RED if m.extra else GREEN)
    line("Missed  (G − O)", ", ".join(m.missed) or "none", YELLOW, RED if m.missed else GREEN)
    line("Order & repeats", "same sequence as the gold standard ✓" if m.same_sequence
         else "sequence differs from the gold standard ✗", YELLOW,
         GREEN if m.same_sequence else RED)
    if gold is not None:
        checks = stage_check(text, gold)
        marks = "   ".join(
            f"{s} " + (Out.c("✓", GREEN, BOLD) if r.same_sequence else Out.c("✗", RED, BOLD))
            for s, r in checks.items())
        print(" " + Out.c("Stage check".ljust(LABEL_WIDTH), YELLOW) + marks)
        print(" " + " " * LABEL_WIDTH
              + Out.c("(each stage fed the gold output of the previous stage)", DIM))
        bad = [s for s, r in checks.items() if not r.same_sequence]
        if bad:
            line("Errors come from", ", ".join(bad), YELLOW, RED)


def print_summary(rows: list[tuple[str, int, Metrics]]) -> None:
    print()
    rule("═", CYAN)
    print(Out.c(" SUMMARY  (Precision and Recall on distinct terms)", BOLD, CYAN))
    rule("═", CYAN)
    header = f" {'text':<10}{'terms':>6}{'|O|':>6}{'|G|':>6}{'|O∩G|':>8}" \
             f"{'Precision':>12}{'Recall':>9}{'F1':>8}"
    print(Out.c(header, BOLD))
    for text_id, n_terms, m in rows:
        print(f" {text_id:<10}{n_terms:>6}{m.n_output:>6}{m.n_gold:>6}{m.n_common:>8}"
              + Out.c(f"{m.precision:>12.3f}", score_color(m.precision), BOLD)
              + Out.c(f"{m.recall:>9.3f}", score_color(m.recall), BOLD)
              + f"{m.f1:>8.3f}")
    if len(rows) > 1:
        p = sum(m.precision for _, _, m in rows) / len(rows)
        r = sum(m.recall for _, _, m in rows) / len(rows)
        f = sum(m.f1 for _, _, m in rows) / len(rows)
        print(Out.c(f" {'average':<36}{p:>12.3f}{r:>9.3f}{f:>8.3f}", BOLD))


# ═══════════════════════════════════════════════════════════════════════════════════
#  ABLATION STUDY - which stage causes which errors?
# ═══════════════════════════════════════════════════════════════════════════════════

VARIANTS = [
    ("none (submitted pipeline)", "-", SUBMITTED),
    ("whitespace split, punctuation kept", "tokenization",
     replace(SUBMITTED, split_punctuation=False)),
    ("no abbreviation rule", "tokenization", replace(SUBMITTED, keep_abbreviations=False)),
    ("no possessive rule", "tokenization", replace(SUBMITTED, strip_possessive=False)),
    ("hyphenated words split", "tokenization", replace(SUBMITTED, keep_hyphenated=False)),
    ("no lower-casing", "normalization", replace(SUBMITTED, lowercase=False)),
    ("no abbreviation unification", "normalization",
     replace(SUBMITTED, unify_abbreviations=False)),
    ("no stop-word removal", "stop words", replace(SUBMITTED, remove_stop_words=False)),
    ("stemming BEFORE stop words", "order", replace(SUBMITTED, stop_before_stem=False)),
    ("no stemming", "stemming", replace(SUBMITTED, stemmer="none")),
    ("S-stemmer instead of Porter", "stemming", replace(SUBMITTED, stemmer="s")),
]


def run_ablation() -> None:
    ids = list(TEST_TEXTS)
    print()
    rule("═", CYAN)
    print(Out.c(" ABLATION STUDY - switch off ONE rule at a time and measure the damage",
                BOLD, CYAN))
    rule("═", CYAN)
    for text in (" Each row is the submitted pipeline with exactly one change, evaluated on the",
                 " three test texts. \"Error born in\" = stages that are wrong even when fed the",
                 " gold output of the previous stage."):
        print(Out.c(text, DIM))
    print()
    head = f" {'#':>2}  {'change':<35}" + "".join(f"{t + ' P/R':>12}" for t in ids) \
        + f"{'avg P':>7}{'avg R':>7}  error born in"
    print(Out.c(head, BOLD))
    results = []
    for k, (name, _stage, cfg) in enumerate(VARIANTS):
        metrics = {t: evaluate(preprocess(TEST_TEXTS[t], cfg).terms, GOLD_STANDARD[t]["terms"])
                   for t in ids}
        avg_p = sum(m.precision for m in metrics.values()) / len(ids)
        avg_r = sum(m.recall for m in metrics.values()) / len(ids)
        born = _unique_in_order(s for t in ids
                                for s in error_source(TEST_TEXTS[t], GOLD_STANDARD[t], cfg))
        results.append((k, name, metrics, avg_p, avg_r))
        cells = "".join(
            " " * 3 + Out.c(f"{m.precision:.2f}", score_color(m.precision)) + "/"
            + Out.c(f"{m.recall:.2f}", score_color(m.recall)) for m in metrics.values())
        print(f" {k:>2}  {name:<35}{cells}"
              + Out.c(f"{avg_p:>7.2f}", score_color(avg_p), BOLD)
              + Out.c(f"{avg_r:>7.2f}", score_color(avg_r), BOLD)
              + "  " + Out.c(", ".join(born) or "-", RED if born else GREEN))

    print()
    print(Out.c(" Wrong terms per change   extra = O − G (lowers precision),"
                "  missed = G − O (lowers recall)", BOLD))
    for k, name, metrics, _p, _r in results[1:]:
        print(Out.c(f" {k:>2}  {name}", YELLOW))
        for t, m in metrics.items():
            if m.extra:
                line(f"      {t} extra", as_json(m.extra), DIM, RED, indent=0, label_width=20)
            if m.missed:
                line(f"      {t} missed", as_json(m.missed), DIM, MAGENTA, indent=0,
                     label_width=20)

    by_name = {name: (p, r) for _k, name, _m, p, r in results}
    worst_p = min(results[1:], key=lambda r: r[3])
    worst_r = min(results[1:], key=lambda r: r[4])
    no_stop_p, no_stop_r = by_name["no stop-word removal"]
    order_p, _ = by_name["stemming BEFORE stop words"]
    print()
    print(Out.c(" Conclusions", BOLD))
    print(f"  * Largest loss of precision: {worst_p[1]} (avg P {worst_p[3]:.2f});"
          f" largest loss of recall: {worst_r[1]} (avg R {worst_r[4]:.2f}).")
    print(f"  * Without stop-word removal precision falls to {no_stop_p:.2f} but recall stays"
          f" {no_stop_r:.2f}: stop words only add noise terms.")
    print(f"  * Every stage is right on its own, yet stemming before stop-word removal gives"
          f" avg P {order_p:.2f}:")
    print("    'this' becomes 'thi', which is not in the stop list - hence the required order.")
    print("  * Tokenization mistakes are inherited by every later stage, which is why the")
    print("    'error born in' column feeds each stage the gold input of the previous one.")


# ═══════════════════════════════════════════════════════════════════════════════════
#  STEP 1 - INPUT, AND THE MAIN PROGRAM
# ═══════════════════════════════════════════════════════════════════════════════════


def _canonical(text: str) -> str:
    return " ".join(text.split())


_BUILT_IN_BY_TEXT = {_canonical(t): k for k, t in TEST_TEXTS.items()}


def read_texts(args: argparse.Namespace) -> dict[str, str]:
    """Step 1. Returns {id: text}. Without options the three test texts are used."""
    texts: list[str] = []
    if args.file:
        if args.file == "-":
            raw = sys.stdin.read()
        else:
            with open(args.file, encoding="utf-8-sig") as fh:
                raw = fh.read()
        texts += [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if args.text:
        texts += [t for t in args.text if t.strip()]
    if args.interactive:
        print("Type or paste a text; finish with an empty line (Ctrl+D also works):")
        lines = []
        while True:
            try:
                ln = input("> ")
            except EOFError:
                break
            if not ln.strip():
                break
            lines.append(ln.strip())
        if lines:
            texts.append(" ".join(lines))
    if not (args.file or args.text or args.interactive):
        return dict(TEST_TEXTS)
    return {f"text{i}": t for i, t in enumerate(texts, start=1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Text preprocessing for IR: tokenization, normalization, stop-word "
                    "removal and Porter stemming, evaluated with Precision and Recall.",
        epilog="Without -f/-t/-i the three test texts of the assignment are processed.")
    parser.add_argument("-f", "--file", help="read texts from FILE, one per non-empty line "
                                             "('-' = standard input)")
    parser.add_argument("-t", "--text", action="append",
                        help="a text to process (can be repeated)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="type a text on the keyboard")
    parser.add_argument("-o", "--output", default="outputs.json",
                        help="where to save the output terms (default: outputs.json)")
    parser.add_argument("-g", "--gold", help="JSON file {\"text1\": [terms], ...} with a gold "
                                             "standard for your own texts")
    parser.add_argument("-b", "--brief", action="store_true",
                        help="show only the final output, not every stage")
    parser.add_argument("--ablation", action="store_true",
                        help="only run the ablation study (error analysis)")
    parser.add_argument("--no-color", action="store_true", help="plain text output")
    args = parser.parse_args(argv)
    Out.setup(args.no_color)

    if args.ablation:
        run_ablation()
        return 0

    try:
        texts = read_texts(args)
        user_gold = {}
        if args.gold:
            with open(args.gold, encoding="utf-8-sig") as fh:
                user_gold = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not texts:
        print("No text to process.", file=sys.stderr)
        return 1

    outputs, rows = {}, []
    for text_id, text in texts.items():
        tr = preprocess(text)
        outputs[text_id] = tr.terms
        if args.brief:
            print()
            line(text_id, text, BOLD + ";" + CYAN, BOLD)
            line("FINAL OUTPUT", as_json(tr.terms), BOLD + ";" + GREEN, BOLD)
        else:
            print_trace(text_id, tr)

        built_in = _BUILT_IN_BY_TEXT.get(_canonical(text))
        if text_id in user_gold:
            gold_terms, full_gold = list(user_gold[text_id]), None
        elif built_in:
            gold_terms, full_gold = GOLD_STANDARD[built_in]["terms"], GOLD_STANDARD[built_in]
        else:
            line("Evaluation", "no gold standard for this text (see --gold)", YELLOW, DIM)
            continue
        m = evaluate(tr.terms, gold_terms)
        print_metrics(m, gold_terms, text, full_gold)
        rows.append((text_id, len(tr.terms), m))

    if rows:
        print_summary(rows)
    try:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(to_json(outputs))
        print(Out.c(f"\n Output terms saved to {args.output}", DIM))
    except OSError as exc:
        print(f"could not write {args.output}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
