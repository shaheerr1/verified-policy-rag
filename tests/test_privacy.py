from app.privacy import leaks, mask

PII = ("Hi, I'm Sarah Khan, I live at 14 Acacia Avenue, SW1A 1AA. "
       "Email sarah.khan@example.com or call 07700 900123. My son's laptop was stolen.")


def test_mask_removes_personal_data():
    out = mask(PII)
    for item in ["Sarah Khan", "SW1A 1AA", "sarah.khan@example.com", "07700 900123"]:
        assert item not in out
    assert "laptop was stolen" in out


def test_policy_details_are_not_leaks():
    source = ["Call our complaints team on 0330 024 9930. Covéa Insurance plc."]
    assert leaks(["You can call Covea on 0330 024 9930."], source) == []


def test_customer_data_in_answer_is_a_leak():
    assert "PERSON" in leaks(["Sarah Khan, your claim is covered."], ["We will pay for theft."])


def test_insurer_name_is_never_masked():
    assert "Covéa" in mask("Will Covéa reimburse me?")
