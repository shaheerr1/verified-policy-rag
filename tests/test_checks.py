import pytest

from app.checks import nli_batch, norm, verdict, windows


@pytest.mark.parametrize("premise,hypothesis,want", [
    ("The cat sat on the mat.", "A cat is on a mat.", "entailment"),
    ("The cat sat on the mat.", "There is no cat on the mat.", "contradiction"),
    ("The cat sat on the mat.", "The cat is hungry.", "neutral"),
    ("We will pay for contents in halls of residence while attending university.",
     "Belongings in university halls are covered.", "entailment"),
])
def test_nli_sanity(premise, hypothesis, want):
    s = nli_batch([premise], hypothesis)[0]
    assert max(s, key=s.get) == want


def test_norm():
    assert norm("dog's") == "dog"
    assert norm("windows") == "window"
    assert norm("glass") == "glass"


def test_windows_keep_list_lead_in():
    text = ("We will cover loss or damage caused by:\n- a. escape of water from any:\n"
            "- i. heating system;\n- ii. fridge or freezer.")
    assert any("We will cover" in w and "fridge or freezer" in w for w in windows(text))


def test_wrong_first_sentence_withholds_whole_answer():
    checked = [{"text": "Yes, covered.", "citations": ["c1"], "status": "off_topic"},
               {"text": "A true condition.", "citations": ["c2"], "status": "supported"}]
    v = verdict(checked)
    assert v["label"] == "HALLUCINATION" and v["deliver"] == []


def test_miscitation_is_repaired_not_blocked():
    checked = [{"text": "A true fact.", "citations": ["c1"], "status": "miscited", "best": "c2"}]
    v = verdict(checked)
    assert v["label"] == "SUCCESS" and v["deliver"][0]["citations"] == ["c2"]


from app.checks import numbers


def test_numbers_normalise():
    assert numbers("up to £1,000.") == {1000.0}
    assert numbers("within three months") == numbers("within 3 months")


def test_wrong_figure_is_caught():
    assert numbers("covered up to £5,000") - numbers("We will pay up to £50,000") == {5000.0}


def test_blocked_condition_withholds_whole_answer():
    checked = [{"text": "Yes, we will pay.", "citations": ["c1"], "status": "supported"},
               {"text": "Not if it was a gift.", "citations": ["c2"], "status": "unsupported"}]
    assert verdict(checked)["deliver"] == []
