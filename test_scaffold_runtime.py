from scaffolds.upstream_base import AiderIOShim, UpstreamScaffold


class DummyPolicy:
    def __init__(self):
        self.labels = []

    def checkpoint(self, label):
        self.labels.append(label)


class DummyModel:
    model_name = "dummy"

    def __init__(self):
        self.options = {"temperature": 0.7}
        self.last_generation_metadata = {}
        self.seeds = []

    def generate(self, prompt, seed=None, timeout=None):
        self.seeds.append(seed)
        self.last_generation_metadata = {"seed": seed}
        return "response"

    def generate_messages(self, messages, seed=None, timeout=None):
        self.seeds.append(seed)
        self.last_generation_metadata = {"seed": seed}
        return "summary"


def test_auxiliary_calls_are_seeded_logged_and_checkpointed():
    policy = DummyPolicy()
    model = DummyModel()
    scaffold = UpstreamScaffold(
        model=model,
        task={"id": "task"},
        options={
            "_run_seed": 12345,
            "_resource_policy": policy,
        },
    )

    assert scaffold.auxiliary_generate(
        "prompt",
        purpose="localization",
        options={"temperature": 0.0},
    ) == "response"
    assert scaffold.auxiliary_generate_messages(
        [{"role": "user", "content": "history"}],
        purpose="summary",
    ) == "summary"

    assert len(model.seeds) == 2
    assert all(isinstance(seed, int) and seed > 0 for seed in model.seeds)
    assert model.seeds[0] != model.seeds[1]
    assert [call["seed"] for call in scaffold.auxiliary_calls] == model.seeds
    assert model.options == {"temperature": 0.7}
    assert policy.labels == [
        "auxiliary:localization:1:before_model_call",
        "auxiliary:localization:1:after_model_call",
        "auxiliary:summary:2:before_model_call",
        "auxiliary:summary:2:after_model_call",
    ]


def test_aider_io_shim_reads_text_and_handles_missing_files(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("print('ok')\n", encoding="utf-8")

    io = AiderIOShim()
    assert io.read_text(source) == "print('ok')\n"
    assert io.read_text(tmp_path / "missing.py", silent=True) is None
    assert io.messages == []

    assert io.read_text(tmp_path / "missing.py") is None
    assert io.messages == [f"{tmp_path / 'missing.py'}: file not found error"]
