from types import SimpleNamespace

from scaffolds.agentless_localization import Scaffold as AgentlessScaffold
from scaffolds.no_scaffold import NoScaffold
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


class DummyTools:
    def list_files(self, limit=400):
        return ["would-not-be-injected.py"]


def test_no_scaffold_has_no_precomputed_context_and_keeps_full_history():
    scaffold = NoScaffold(
        model=DummyModel(),
        task={"id": "control", "tests_enabled": False},
        options={},
    )
    tools = DummyTools()
    state = scaffold.init_state("task", tools)
    history = [
        {
            "step": index,
            "action": {"action": "finish"},
            "observation": {"type": "finished"},
        }
        for index in range(1, 9)
    ]

    prompt = scaffold.build_context("task", tools, history, state)

    assert state["context_files"] == []
    assert state["context_spans"] == {}
    assert "No precomputed repository context" in prompt
    assert "would-not-be-injected.py" not in prompt
    assert '"step": 1' in prompt
    assert '"step": 8' in prompt


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


def test_agentless_patches_model_factory_at_actual_import_source_only():
    original_factory = object()
    scaffold = AgentlessScaffold.__new__(AgentlessScaffold)
    scaffold.model_module = SimpleNamespace(make_model=original_factory)
    scaffold.fl_module = SimpleNamespace()
    scaffold.options = {}
    scaffold.model = DummyModel()
    scaffold.auxiliary_calls = []

    with scaffold._patched_model_factory():
        assert callable(scaffold.model_module.make_model)
        assert not hasattr(scaffold.fl_module, "make_model")
        decoder = scaffold.model_module.make_model(
            model="evaluated-model",
            backend="openai",
            logger=None,
            batch_size=1,
            max_tokens=321,
            temperature=0.25,
        )
        assert decoder.scaffold is scaffold
        assert decoder.name == "evaluated-model"
        assert decoder.max_new_tokens == 321
        assert decoder.temperature == 0.25

    assert scaffold.model_module.make_model is original_factory
    assert not hasattr(scaffold.fl_module, "make_model")
