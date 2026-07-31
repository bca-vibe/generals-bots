import sys
from types import SimpleNamespace

from generals.training.config import TrainingConfig
from generals.training.tracking import WandbTracker


class FakeRun:
    def __init__(self):
        self.url = "https://wandb.example/test-run"
        self.defined_metrics = []
        self.logged = []
        self.finished = False

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))

    def log(self, metrics):
        self.logged.append(metrics)

    def finish(self):
        self.finished = True


def test_tracking_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    tracker = WandbTracker.start(
        TrainingConfig(), start_iteration=0, resume=None, device_count=8
    )
    assert not tracker.active


def test_tracking_initializes_grouped_resume_run_and_namespaces_metrics(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    fake_run = FakeRun()
    init_calls = []

    def init(**kwargs):
        init_calls.append(kwargs)
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    config = TrainingConfig(
        run_name="conv_branch",
        wandb_project="generals-bots",
        wandb_entity="research",
        wandb_group="architecture-ab",
        wandb_tags=("conv", "locked-maps"),
    )
    tracker = WandbTracker.start(
        config,
        start_iteration=540,
        resume="checkpoints/conv_branch/checkpoint_000540.eqx",
        device_count=8,
    )

    assert tracker.active
    assert init_calls[0]["project"] == "generals-bots"
    assert init_calls[0]["entity"] == "research"
    assert init_calls[0]["name"] == "conv_branch-resume-000540"
    assert init_calls[0]["group"] == "architecture-ab"
    assert init_calls[0]["tags"] == ["conv", "locked-maps"]
    assert init_calls[0]["config"]["start_iteration"] == 540
    assert init_calls[0]["config"]["resume_checkpoint"] == "checkpoint_000540.eqx"
    assert init_calls[0]["config"]["device_count"] == 8
    assert (("*",), {"step_metric": "iteration"}) in fake_run.defined_metrics

    tracker.log_training(
        {
            "iteration": 541,
            "stage": 2,
            "loss": 1.25,
            "samples_per_second": 390_000.0,
        }
    )
    tracker.log_evaluation(
        {"iteration": 550, "evaluation/score": 0.625, "evaluation/wins": 128.0}
    )

    assert fake_run.logged[0] == {
        "iteration": 541,
        "curriculum/stage": 2,
        "training/loss": 1.25,
        "performance/samples_per_second": 390_000.0,
    }
    assert fake_run.logged[1]["evaluation/score"] == 0.625

    tracker.finish()
    assert fake_run.finished
    assert not tracker.active


def test_tracking_failure_does_not_abort_training(monkeypatch, capsys):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    def fail_init(**kwargs):
        raise ConnectionError("network unavailable")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fail_init))
    tracker = WandbTracker.start(
        TrainingConfig(wandb_project="generals-bots"),
        start_iteration=0,
        resume=None,
        device_count=1,
    )

    assert not tracker.active
    assert "continuing locally" in capsys.readouterr().err
