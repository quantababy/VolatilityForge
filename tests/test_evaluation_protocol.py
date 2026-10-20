from pathlib import Path


def test_dm_inference_method_and_limitations_are_documented() -> None:
    protocol = Path("docs/evaluation_protocol.md").read_text(encoding="utf-8")

    required_phrases = (
        "Bartlett-weighted Newey-West HAC estimator",
        "Harvey-Leybourne-Newbold (HLN) finite-sample adjustment",
        "Student-t reference distribution with n - 1 degrees",
        "numerical agreement",
        "HAC bandwidth",
        "overlapping five-day forecasts",
    )

    for phrase in required_phrases:
        assert phrase in protocol
