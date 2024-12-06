from collections import Counter

from doctr.datasets import VOCABS


def test_vocabs_duplicates():
    for key, vocab in VOCABS.items():
        assert isinstance(vocab, str)

        duplicates = [char for char, count in Counter(vocab).items() if count > 1]
        assert not duplicates, f"Duplicate characters in {key} vocab: {duplicates}"

        # Check for whitespace characters
        assert not any(char.isspace() for char in vocab), f"Whitespace in {key} vocab"
        # Check for line breaks
        assert "\n" not in vocab, f"Line breaks in {key} vocab"
