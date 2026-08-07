from scripts.validate_model_profile import (
    build_token_prompt,
    capacity_has_margin,
)


def test_build_token_prompt_exact_length():
    assert build_token_prompt([7, 8], 5) == [7, 8, 7, 8, 7]


def test_capacity_has_margin_requires_peak_and_fit_reserve():
    assert capacity_has_margin(30000, 32768, 1536) is True
    assert capacity_has_margin(32000, 32768, 1536) is False
    assert capacity_has_margin(0, 32768, 1536) is False
