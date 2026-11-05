import pytest
from pilot.text import normalize, error_rates
from pilot.timing import Word, gaps, validate_words
from pilot.media import read_vtt, read_stm, select, validate_plan

def test_normalization_and_wer():
    assert normalize("Hello, WORLD!") == "hello world"
    assert error_rates("Hello world", "hello, world!")["wer"] == 0

def test_gap_accounting():
    r=gaps([Word('one',0,.1), Word('two',.35,.5), Word('three',.55,.7)], .2)
    assert len(r)==2 and r[0]['candidate'] and not r[1]['candidate']

def test_failed_alignment_is_rejected():
    with pytest.raises(ValueError): validate_words([Word('a',.2,.1)])
    with pytest.raises(ValueError): validate_words([Word('a',0,.3),Word('b',.2,.4)])

def test_caption_and_stm_extraction(tmp_path):
    vtt=tmp_path/'x.vtt'; vtt.write_text('WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello <b>world</b>\n')
    stm=tmp_path/'x.stm'; stm.write_text('talk 1 speaker 1.0 2.0 <o,f0,male> hello world\n')
    assert select(read_vtt(vtt), 1, 2) == 'Hello world'
    assert select(read_stm(stm), 1, 2) == 'hello world'

def test_source_plan_checks_overlap_and_reference_length():
    validate_plan((0,8), {'short':(20,25)})
    with pytest.raises(ValueError): validate_plan((0,4), {'short':(20,25)})
    with pytest.raises(ValueError): validate_plan((0,8), {'short':(4,10)})
