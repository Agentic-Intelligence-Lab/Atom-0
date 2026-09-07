from openpi.training import checkpoints


class _RecordingManager:
    def __init__(self):
        self.step = None
        self.items = None

    def save(self, step, items):
        self.step = step
        self.items = items


def test_save_state_params_only_omits_train_state(monkeypatch) -> None:
    manager = _RecordingManager()
    monkeypatch.setattr(checkpoints, "_split_params", lambda state: ("optimizer-state", "model-params"))

    checkpoints.save_state(manager, object(), object(), 10, params_only=True)

    assert manager.step == 10
    assert set(manager.items) == {"assets", "params"}
    assert manager.items["params"] == {"params": "model-params"}


def test_save_state_defaults_to_full_train_state(monkeypatch) -> None:
    manager = _RecordingManager()
    monkeypatch.setattr(checkpoints, "_split_params", lambda state: ("optimizer-state", "model-params"))

    checkpoints.save_state(manager, object(), object(), 20)

    assert manager.step == 20
    assert set(manager.items) == {"assets", "params", "train_state"}
    assert manager.items["train_state"] == "optimizer-state"
