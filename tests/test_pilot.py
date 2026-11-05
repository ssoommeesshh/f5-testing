import pytest
from pilot.text import normalize, error_rates
from pilot.timing import Word, gaps, validate_words

def test_normalization_and_wer():
    assert normalize("Hello, WORLD!") == "hello world"
    assert error_rates("Hello world", "hello, world!")["wer"] == 0

def test_gap_accounting():
    r=gaps([Word('one',0,.1), Word('two',.35,.5), Word('three',.55,.7)], .2)
    assert len(r)==2 and r[0]['candidate'] and not r[1]['candidate']

def test_failed_alignment_is_rejected():
    with pytest.raises(ValueError): validate_words([Word('a',.2,.1)])
    with pytest.raises(ValueError): validate_words([Word('a',0,.3),Word('b',.2,.4)])
