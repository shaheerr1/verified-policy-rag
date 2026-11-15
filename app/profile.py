"""The customer's own policy details (tier and covers held), supplied as a citable clause."""

POLICIES = {
    "Policy Expert": {"tiers": ["Basic", "Bronze", "Silver", "Gold"],
                      "covers": ["Buildings", "Contents", "Option A - personal possessions away from home",
                                 "Option B - pedal cycles"]},
    "Uinsure": {"tiers": [],
                "covers": ["Buildings", "Contents", "Accidental damage", "Family Legal Expenses", "Home Emergency"]},
    "Covea": {"tiers": [],
              "covers": ["Contents, art and jewellery", "Buildings", "Home Emergency", "Legal Protection",
                         "Cyber Assistance"]},
}


def describe(insurer, profile):
    tier = profile.get("tier")
    held = profile.get("covers", [])
    parts = [f"The customer holds {insurer} {tier}." if tier else f"The customer holds {insurer}."]
    if held:
        parts.append("Covers shown in their schedule: " + ", ".join(held) + ".")
    if held and profile.get("complete"):
        missing = [c for c in POLICIES[insurer]["covers"] if c not in held]
        if missing:
            parts.append("Covers not shown in their schedule: " + ", ".join(missing) + ".")
    return " ".join(parts)


def profile_hit(insurer, profile):
    return {"id": "profile", "text": describe(insurer, profile), "insurer": insurer,
            "section": "Customer policy details", "printed_pages": ""}
