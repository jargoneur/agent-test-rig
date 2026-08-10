from scripts.validate_model_profile import (
    assess_prefill_completion,
    build_token_prompt,
    capacity_has_margin,
)


def test_build_token_prompt_exact_length():
    assert build_token_prompt([7, 8], 5) == [7, 8, 7, 8, 7]


def test_capacity_has_margin_requires_peak_and_fit_reserve():
    assert capacity_has_margin(30000, 32768, 1536) is True
    assert capacity_has_margin(32000, 32768, 1536) is False
    assert capacity_has_margin(0, 32768, 1536) is False


def test_full_prompt_at_reserved_generation_boundary_is_not_prompt_loss():
    result = assess_prefill_completion(
        {"tokens_evaluated": 262143, "truncated": True},
        262143,
        262144,
    )

    assert result["prompt_prefill_complete"] is True
    assert result["prompt_tokens_truncated"] is False
    assert result["capacity_boundary_stop"] is True


def test_missing_prompt_tokens_remain_a_validation_failure():
    result = assess_prefill_completion(
        {"tokens_evaluated": 262000, "truncated": True},
        262143,
        262144,
    )

    assert result["prompt_prefill_complete"] is False
    assert result["prompt_tokens_truncated"] is True
    assert result["capacity_boundary_stop"] is False
