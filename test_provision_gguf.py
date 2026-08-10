import shutil
from pathlib import Path

import pytest

from scripts.provision_gguf import check_storage


def test_check_storage_records_location_and_rule(tmp_path: Path):
    report = check_storage(tmp_path, 0, 0, "test_rule")

    assert report["path"] == str(tmp_path)
    assert report["estimate_rule"] == "test_rule"


def test_check_storage_rejects_requirement_above_free_space(tmp_path: Path):
    free = shutil.disk_usage(tmp_path).free

    with pytest.raises(RuntimeError, match="Insufficient free storage"):
        check_storage(tmp_path, free + 1, 0, "test_rule")
