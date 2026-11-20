# Text Preprocessing for Information Retrieval

Course project — *Foundations of Information Retrieval and Web Search*, part 2 (Boolean retrieval).
Author: **Samane Basiriniya** (سمانه بصیری نیا)

The program turns raw text into the sequence of index terms used to build an inverted index:

```
text ──► tokenization ──► normalization ──► stop-word removal ──► stemming ──► terms
```

and measures how well the result agrees with a hand-made gold standard (Precision / Recall).

<div dir="rtl">

**خلاصه:** این برنامه مراحل پیش‌پردازش متن برای ساخت شاخص معکوس را پیاده‌سازی می‌کند:
توکن‌سازی، یکسان‌سازی، حذف کلمات ایست و ریشه‌یابی (ریشه‌یاب پورتر، بدون کتابخانه و از صفر).
خروجی هر متن با «خروجی طلایی» که دستی ساخته شده مقایسه و Precision و Recall آن محاسبه می‌شود.
برای اجرا فقط پایتون ۳.۸ یا جدیدتر لازم است.

</div>

## Requirements

Python 3.8 or newer. **No packages to install** — only the standard library is used
(the Porter stemmer is implemented in the file and the stop list is embedded).

## How to run

```bash
git clone https://github.com/<your-username>/ir-text-preprocessing.git
cd ir-text-preprocessing

python3 Samane_Basiriniya.py            # the 3 test texts, every stage, Precision & Recall
```

On Windows use `python` instead of `python3`.

| Command | What it does |
|---|---|
| `python3 Samane_Basiriniya.py` | processes the three test texts of the assignment and evaluates them |
| `python3 Samane_Basiriniya.py -t "Any text you like."` | a text from the command line (`-t` can be repeated) |
| `python3 Samane_Basiriniya.py -f sample_texts.txt` | texts from a file, one text per non-empty line |
| `echo "Some text" \| python3 Samane_Basiriniya.py -f -` | texts from standard input |
| `python3 Samane_Basiriniya.py -i` | type a text on the keyboard (finish with an empty line) |
| `python3 Samane_Basiriniya.py -g my_gold.json -f my_texts.txt` | evaluate your own texts against your own gold standard `{"text1": [...], ...}` |
| `python3 Samane_Basiriniya.py -b` | brief: final output and metrics only |
| `python3 Samane_Basiriniya.py --ablation` | error analysis: switch rules off one at a time |
| `python3 -m unittest -v` | run the 24 unit tests |

The output terms are also saved to `outputs.json`:

```json
{
  "text1": ["friend", "roman", "countrymen", "lend", "ear"],
  "text2": ["john", "new", "usa", "laptop", "state-of-the-art", "devic", "price", "high", "qualiti", "amaz"],
  "text3": ["author", "author", "project", "pend", "author", "payment", "regularli"]
}
```

## What each stage does

| Stage | Rules |
|---|---|
| 1. Input | built-in test texts, `-t`, `-f FILE`, `-f -` (stdin) or `-i` |
| 2. Tokenization | punctuation is separated from words and dropped, except dots inside abbreviations (`U.S.A.`); hyphenated words stay one token (`state-of-the-art`); possessive `'s` / `’s` is removed (`John's → John`, `students' → students`); numbers such as `3.14` and `1,000` stay whole |
| 3a. Normalization | lower-casing; abbreviations lose their dots so `U.S.A.`, `U.S.A` and `USA` all become `usa`; curly apostrophes and accents are unified (`café → cafe`) |
| 3b. Stop words | a fixed list: the 198-word English list of NLTK, embedded in the code (no download) |
| 3c. Stemming | the Porter stemmer, written from scratch — identical to Martin Porter's reference output on all 23,531 words of the official test vocabulary |

Stop words are removed **before** stemming: otherwise `this` is stemmed to `thi`, which is no
longer in the stop list (the ablation study shows it).

## Evaluation

For every text, with `O` = the set of distinct terms produced by the program and `G` = the
set of distinct terms of the gold standard:

```
Precision = |O ∩ G| / |O|        Recall = |O ∩ G| / |G|        (0 if the denominator is 0)
```

The program also prints the extra terms (`O − G`) and the missed terms (`G − O`), and checks
each stage separately by feeding it the gold output of the previous stage, so that an error
is charged to the stage that makes it.

| text | Precision | Recall |
|---|---|---|
| text1 | 1.000 | 1.000 |
| text2 | 1.000 | 1.000 |
| text3 | 1.000 | 1.000 |

`--ablation` shows how these numbers fall when one rule is switched off — for example
without stemming the average precision drops to 0.41, and without stop-word removal it
drops to 0.64 while recall stays at 1.00.

## Files

```
Samane_Basiriniya.py        the program
tests/                      unit tests (python3 -m unittest -v)
sample_texts.txt            the three test texts, one per line
outputs.json                output of the last run
docs/report_fa.pdf          project report (in Persian): design decisions, hand-made gold standard, error analysis
```
