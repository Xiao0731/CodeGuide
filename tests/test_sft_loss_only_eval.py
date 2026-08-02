from contextlib import nullcontext
import sys

from src.training.train_sft import LossOnlyPredictionMixin


class _FakeLoss:
    def detach(self):
        return self

    def mean(self):
        return 3.0


class _FakeTorch:
    @staticmethod
    def no_grad():
        return nullcontext()


class _FakeTrainer(LossOnlyPredictionMixin):
    def __init__(self):
        self.return_outputs = None
        self.prepared_inputs = None

    def _prepare_inputs(self, inputs):
        self.prepared_inputs = {"prepared": inputs["value"]}
        return self.prepared_inputs

    def compute_loss_context_manager(self):
        return nullcontext()

    def compute_loss(self, model, inputs, return_outputs=False):
        del model
        self.return_outputs = return_outputs
        assert inputs == self.prepared_inputs
        return _FakeLoss()


def test_loss_only_prediction_never_requests_model_outputs(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    trainer = _FakeTrainer()

    loss, logits, labels = trainer.prediction_step(
        object(), {"value": 7}, prediction_loss_only=False, ignore_keys=["logits"]
    )

    assert trainer.return_outputs is False
    assert loss == 3.0
    assert logits is None
    assert labels is None
