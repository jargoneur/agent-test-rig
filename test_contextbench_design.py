import json
from collections import Counter

from harness.contextbench_design import (
    declared_task_type,
    patch_statistics,
    select_representative,
)


def make_record(index):
    file_count = index % 5 + 1
    chunks = []
    for file_index in range(file_count):
        path = "src/group%d/file%d.py" % (file_index % 3, file_index)
        chunks.append(
            "diff --git a/{0} b/{0}\n"
            "--- a/{0}\n"
            "+++ b/{0}\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new-{1}\n".format(path, index)
        )
    return {
        "instance_id": "Bench__python__maintenance__bugfix__%04d" % index,
        "original_inst_id": "org__repo-%d" % index,
        "repo": "org/repo-%02d" % (index % 15),
        "source": ("Verified", "Pro", "Poly", "Multi")[index % 4],
        "language": (
            "python",
            "javascript",
            "typescript",
            "go",
            "c",
            "rust",
            "cpp",
            "java",
        )[index % 8],
        "problem_statement": "Fix behavior " + ("x" * (index * 3)),
        "patch": "".join(chunks),
        "test_patch": chunks[0] * (index % 4),
        "gold_context": json.dumps(
            [
                {
                    "file": "src/group%d/file%d.py" % (value % 3, value),
                    "start_line": 1,
                    "end_line": index + 2,
                    "content": "x" * (index + 1),
                }
                for value in range(file_count)
            ]
        ),
    }


def test_patch_statistics_and_declared_taxonomy():
    record = make_record(4)
    stats = patch_statistics(record["patch"])

    assert stats["file_count"] == 5
    assert stats["hunks"] == 5
    assert stats["changed_lines"] == 10
    assert declared_task_type(record["instance_id"]) == "maintenance_bugfix"


def test_representative_selection_is_deterministic_and_complexity_balanced():
    records = [make_record(index) for index in range(90)]
    first = select_representative(records, limit=30, seed=17)
    second = select_representative(records, limit=30, seed=17)

    assert [row["instance_id"] for row in first] == [
        row["instance_id"] for row in second
    ]
    assert len(first) == len({row["instance_id"] for row in first}) == 30
    assert Counter(
        row["task_design"]["complexity"]["bin"] for row in first
    ) == {"easy": 10, "intermediate": 10, "hard": 10}
    assert {row["source"] for row in first} == {
        "Verified",
        "Pro",
        "Poly",
        "Multi",
    }
    assert len({row["language"] for row in first}) == 8
    assert all(
        row["task_design"]["source_declared_task_type"]
        == "maintenance_bugfix"
        for row in first
    )
