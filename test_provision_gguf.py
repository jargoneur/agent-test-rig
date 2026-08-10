import json
import shutil
from pathlib import Path

import pytest

from scripts.provision_gguf import (
    check_storage,
    prepare_converter_snapshot,
)


def test_check_storage_records_location_and_rule(tmp_path: Path):
    report = check_storage(tmp_path, 0, 0, "test_rule")

    assert report["path"] == str(tmp_path)
    assert report["estimate_rule"] == "test_rule"


def test_check_storage_rejects_requirement_above_free_space(tmp_path: Path):
    free = shutil.disk_usage(tmp_path).free

    with pytest.raises(RuntimeError, match="Insufficient free storage"):
        check_storage(tmp_path, free + 1, 0, "test_rule")


def test_prepare_converter_snapshot_normalizes_list_without_mutating_source(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    config_path = source / "tokenizer_config.json"
    config_path.write_text(
        json.dumps(
            {
                "tokenizer_class": "GemmaTokenizer",
                "extra_special_tokens": ["<|video|>"],
            }
        ),
        encoding="utf-8",
    )
    tokenizer_path = source / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(
            {
                "added_tokens": [
                    {
                        "id": 258884,
                        "content": "<|video|>",
                        "special": True,
                    }
                ],
                "model": {"vocab": {"<|video|>": 258884}},
            }
        ),
        encoding="utf-8",
    )
    original_config = config_path.read_bytes()
    original_tokenizer = tokenizer_path.read_bytes()

    converter, record = prepare_converter_snapshot(
        source,
        tmp_path / "converter",
    )

    assert converter != source
    assert record is not None
    assert record["tokens"] == [{"token": "<|video|>", "id": 258884}]
    assert config_path.read_bytes() == original_config
    assert tokenizer_path.read_bytes() == original_tokenizer
    normalized = json.loads((converter / "tokenizer_config.json").read_text())
    assert "extra_special_tokens" not in normalized
    assert normalized["additional_special_tokens"] == ["<|video|>"]
    assert (converter / "tokenizer.json").read_bytes() == original_tokenizer


def test_prepare_converter_snapshot_rejects_unverified_special_token(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": ["<missing>"]}),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text(
        json.dumps({"added_tokens": [], "model": {"vocab": {}}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not marked special"):
        prepare_converter_snapshot(source, tmp_path / "converter")
    assert not (tmp_path / "converter").exists()
