"""Behavioural tests of the model (CheckList-style: MFT, invariance, directional).

Marked ``model``: they download the real weights (~500 MB, cached afterwards).
Run only the fast suite with ``pytest -m "not model"``.
"""

import pytest

from sentiment_service.model import LABELS, preprocess


def test_preprocess_masks_users_and_links():
    text = "@acme_support thanks!   see https://t.co/abc and www.acme.com"
    assert preprocess(text) == "@user thanks! see http and http"


def test_preprocess_keeps_emails_and_plain_text():
    assert preprocess("write to info@acme.com") == "write to info@acme.com"
    assert preprocess("nothing to change") == "nothing to change"


@pytest.mark.model
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I absolutely love this phone, best purchase ever!", "positive"),
        ("This is the worst customer service I have ever experienced.", "negative"),
        ("The store opens at 9am on Monday.", "neutral"),
        ("Thanks @support, issue solved in 5 minutes 🙌", "positive"),
        ("My order arrived broken and nobody answers. Terrible.", "negative"),
    ],
)
def test_minimum_functionality(real_model, text, expected):
    assert real_model.predict([text])[0].label == expected


@pytest.mark.model
def test_output_is_a_probability_distribution(real_model):
    pred = real_model.predict(["just a normal day"])[0]
    assert set(pred.scores) == set(LABELS)
    assert sum(pred.scores.values()) == pytest.approx(1.0, abs=1e-4)
    assert pred.score == max(pred.scores.values())


@pytest.mark.model
def test_invariant_to_user_handles_and_urls(real_model):
    a, b = real_model.predict(
        [
            "@mario thanks for the help, great support! https://t.co/xyz",
            "@luigi_99 thanks for the help, great support! https://bit.ly/123",
        ]
    )
    assert a.label == b.label
    assert a.score == pytest.approx(b.score, abs=1e-4)


@pytest.mark.model
def test_negation_moves_prediction_away_from_positive(real_model):
    pos, neg = real_model.predict(["The new update is good.", "The new update is not good."])
    assert neg.scores["positive"] < pos.scores["positive"]


@pytest.mark.model
def test_batch_equals_single(real_model):
    texts = ["love it", "hate it", "it is tuesday"]
    batched = [p.label for p in real_model.predict(texts)]
    single = [real_model.predict([t])[0].label for t in texts]
    assert batched == single
