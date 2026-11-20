"""Unit tests - standard library only.   Run:  python3 -m unittest -v"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Samane_Basiriniya import (  # noqa: E402
    GOLD_STANDARD, PORTER, STOP_WORDS, SUBMITTED, TEST_TEXTS, evaluate, normalize, preprocess,
    remove_stop_words, stage_check, tokenize)


class TokenizationTests(unittest.TestCase):
    def test_punctuation_is_separated_and_dropped(self):
        self.assertEqual(tokenize("Friends, Romans, countrymen, lend me your ears;"),
                         ["Friends", "Romans", "countrymen", "lend", "me", "your", "ears"])
        self.assertEqual(tokenize("Really?! (yes) - no..."), ["Really", "yes", "no"])

    def test_hyphenated_word_is_one_token(self):
        self.assertEqual(tokenize("a state-of-the-art device"), ["a", "state-of-the-art", "device"])

    def test_possessive_is_removed_for_both_apostrophes(self):
        self.assertEqual(tokenize("John's"), ["John"])
        self.assertEqual(tokenize("John’s"), ["John"])
        self.assertEqual(tokenize("the students' books"), ["the", "students", "books"])

    def test_abbreviation_keeps_its_dots(self):
        self.assertEqual(tokenize("new U.S.A. laptop"), ["new", "U.S.A.", "laptop"])
        self.assertEqual(tokenize("I live in the U.S.A."), ["I", "live", "in", "the", "U.S.A."])
        self.assertEqual(tokenize("U.S.A.'s policy"), ["U.S.A.", "policy"])
        self.assertEqual(tokenize("end.Next"), ["end", "Next"])        # not an abbreviation

    def test_numbers_stay_whole(self):
        self.assertEqual(tokenize("pi is 3.14, not 1,000"), ["pi", "is", "3.14", "not", "1,000"])


class NormalizationTests(unittest.TestCase):
    def test_lower_case(self):
        self.assertEqual(normalize("Friends"), "friends")

    def test_usa_spellings_meet(self):
        self.assertEqual(normalize("U.S.A."), "usa")
        self.assertEqual(normalize("USA"), "usa")
        self.assertEqual(normalize("U.S.A"), "usa")

    def test_apostrophes_and_accents(self):
        self.assertEqual(normalize("Don’t"), "don't")
        self.assertEqual(normalize("Café"), "cafe")


class StopWordTests(unittest.TestCase):
    def test_fixed_list(self):
        self.assertEqual(len(STOP_WORDS), 198)
        for w in ("the", "of", "to", "me", "your", "is", "a", "but", "this", "he", "for"):
            self.assertIn(w, STOP_WORDS)

    def test_removal(self):
        self.assertEqual(remove_stop_words(["the", "price", "is", "high"]), ["price", "high"])

    def test_why_stop_words_go_before_stemming(self):
        self.assertEqual(PORTER.stem("this"), "thi")
        self.assertNotIn("thi", STOP_WORDS)


class PorterTests(unittest.TestCase):
    """Examples from Porter (1980), checked step by step."""

    def check(self, step, pairs):
        for word, expected in pairs:
            with self.subTest(word=word):
                self.assertEqual(step(word), expected)

    def test_step1a(self):
        self.check(PORTER._step1a, [("caresses", "caress"), ("ponies", "poni"), ("ties", "ti"),
                                    ("caress", "caress"), ("cats", "cat")])

    def test_step1b(self):
        self.check(PORTER._step1b, [
            ("feed", "feed"), ("agreed", "agree"), ("plastered", "plaster"), ("bled", "bled"),
            ("motoring", "motor"), ("sing", "sing"), ("conflated", "conflate"),
            ("troubled", "trouble"), ("sized", "size"), ("hopping", "hop"), ("tanned", "tan"),
            ("falling", "fall"), ("hissing", "hiss"), ("fizzed", "fizz"), ("failing", "fail"),
            ("filing", "file")])

    def test_step1c(self):
        self.check(PORTER._step1c, [("happy", "happi"), ("sky", "sky")])

    def test_step2(self):
        self.check(PORTER._step2, [
            ("relational", "relate"), ("conditional", "condition"), ("rational", "rational"),
            ("valenci", "valence"), ("hesitanci", "hesitance"), ("digitizer", "digitize"),
            ("conformabli", "conformable"), ("radicalli", "radical"),
            ("differentli", "different"), ("vileli", "vile"), ("analogousli", "analogous"),
            ("vietnamization", "vietnamize"), ("predication", "predicate"),
            ("operator", "operate"), ("feudalism", "feudal"), ("decisiveness", "decisive"),
            ("hopefulness", "hopeful"), ("callousness", "callous"), ("formaliti", "formal"),
            ("sensitiviti", "sensitive"), ("sensibiliti", "sensible")])

    def test_step3(self):
        self.check(PORTER._step3, [
            ("triplicate", "triplic"), ("formative", "form"), ("formalize", "formal"),
            ("electriciti", "electric"), ("electrical", "electric"), ("hopeful", "hope"),
            ("goodness", "good")])

    def test_step4(self):
        self.check(PORTER._step4, [
            ("revival", "reviv"), ("allowance", "allow"), ("inference", "infer"),
            ("airliner", "airlin"), ("gyroscopic", "gyroscop"), ("adjustable", "adjust"),
            ("defensible", "defens"), ("irritant", "irrit"), ("replacement", "replac"),
            ("adjustment", "adjust"), ("dependent", "depend"), ("adoption", "adopt"),
            ("homologou", "homolog"), ("communism", "commun"), ("activate", "activ"),
            ("angulariti", "angular"), ("homologous", "homolog"), ("effective", "effect"),
            ("bowdlerize", "bowdler")])

    def test_step5(self):
        self.check(PORTER._step5, [("probate", "probat"), ("rate", "rate"), ("cease", "ceas"),
                                   ("controll", "control"), ("roll", "roll")])

    def test_whole_words(self):
        self.check(PORTER.stem, [
            ("generalizations", "gener"), ("oscillators", "oscil"), ("authorization", "author"),
            ("authorized", "author"), ("authorizes", "author"), ("quality", "qualiti"),
            ("is", "is"), ("as", "as")])

    def test_reference_vocabulary_if_available(self):
        """23,531 words of Martin Porter's reference test set (nltk_data 'porter_test')."""
        folder = os.path.expanduser("~/nltk_data/stemmers/porter_test")
        voc = os.path.join(folder, "porter_vocabulary.txt")
        ref = os.path.join(folder, "porter_martin_output.txt")
        if not (os.path.exists(voc) and os.path.exists(ref)):
            self.skipTest("run  python -c \"import nltk; nltk.download('porter_test')\"  first")
        with open(voc) as v, open(ref) as r:
            pairs = list(zip(v.read().split(), r.read().split()))
        wrong = [(w, e, PORTER.stem(w)) for w, e in pairs if PORTER.stem(w) != e]
        self.assertEqual(wrong, [])


class EvaluationTests(unittest.TestCase):
    def test_example_from_the_assignment(self):
        m = evaluate(["john", "new", "laptop", "usa", "project"], ["john", "new", "laptop", "high"])
        self.assertAlmostEqual(m.precision, 0.60)
        self.assertAlmostEqual(m.recall, 0.75)
        self.assertEqual(m.extra, ["usa", "project"])
        self.assertEqual(m.missed, ["high"])

    def test_repetitions_do_not_count(self):
        m = evaluate(["author", "author", "pend"], ["author", "pend"])
        self.assertEqual((m.precision, m.recall), (1.0, 1.0))
        self.assertFalse(m.same_sequence)

    def test_empty_denominator_gives_zero(self):
        self.assertEqual(evaluate([], ["a"]).precision, 0.0)
        self.assertEqual(evaluate(["a"], []).recall, 0.0)


class GoldStandardTests(unittest.TestCase):
    def test_three_texts(self):
        for text_id, text in TEST_TEXTS.items():
            with self.subTest(text=text_id):
                self.assertEqual(preprocess(text).terms, GOLD_STANDARD[text_id]["terms"])
                for stage, m in stage_check(text, GOLD_STANDARD[text_id], SUBMITTED).items():
                    self.assertTrue(m.same_sequence, stage)


if __name__ == "__main__":
    unittest.main()
