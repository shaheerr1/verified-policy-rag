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


from app.checks import matching_exceptions


def test_exception_matching_customer_situation_is_found():
    clause = ("You must tell us about the work at least 30 days before it starts.\n"
              "(You do not need to tell us if the work is to replace or refit existing kitchen or bathroom "
              "fittings and does not involve making any structural changes to walls, ceilings or floors.)")
    q = "I'm ripping out the old kitchen units and fitting new ones without knocking down any walls. Do I have to tell you?"
    assert matching_exceptions(q, [clause])


def test_unrelated_exception_is_ignored():
    clause = "We don't cover theft unless force or violence was used to enter the building."
    assert not matching_exceptions("Is my dog's vet bill covered?", [clause])


from app.profile import describe


def test_profile_only_states_missing_covers_when_complete():
    partial = describe("Uinsure", {"covers": ["Family Legal Expenses"]})
    full = describe("Policy Expert", {"tier": "Basic", "covers": ["Buildings", "Contents"], "complete": True})
    assert "not shown" not in partial
    assert "Policy Expert Basic" in full and "Option A" in full and "not shown" in full


from app.checks import contains


def test_quote_must_come_from_its_clause():
    clause = "Section 2 Contents\nWe will pay for loss or damage to contents in halls of residence."
    assert contains(clause, "We will pay for loss or damage to contents in halls of residence")
    assert contains(clause, "we will pay for loss or damage to  contents in halls of residence.")
    assert not contains(clause, "We will pay for loss of your car anywhere in the world")


def test_refusal_never_shows_model_text():
    checked = [{"text": "No, your son's things are not covered.", "citations": [], "status": "refusal"}]
    v = verdict(checked)
    assert v["label"] == "REFUSAL"
    assert "not covered" not in v["deliver"][0]["text"]


from app.checks import key_terms, strip_ids


def test_clause_ids_are_removed_from_answers():
    assert strip_ids("Under c0008, you have up to £2,500 towards it.") == "you have up to £2,500 towards it."
    assert strip_ids("The excess is £250 (c0412).") == "The excess is £250 ."


def test_glossary_terms_always_count_as_key_terms():
    assert "vehicle" in key_terms("Someone smashed my car window and stole my backpack. Can I claim?")


def test_delivered_answer_drops_repeats_and_fragments():
    checked = [{"text": "You have these benefits:", "citations": ["c1"], "status": "supported"},
               {"text": "We pay up to £2,500.", "citations": ["c1"], "status": "supported"},
               {"text": "We pay up to £2,500.", "citations": ["c1"], "status": "supported"}]
    assert [c["text"] for c in verdict(checked)["deliver"]] == ["We pay up to £2,500."]
