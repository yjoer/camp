import json
from tempfile import TemporaryDirectory

import fsspec
import torch
from safetensors.torch import load
from safetensors.torch import save
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


def save_optimizer(path: str, optimizer: Optimizer) -> None:
    state_dict = optimizer.state_dict()
    tensors = {}
    metadata = {}

    for k1, v1 in state_dict["state"].items():
        metadata[k1] = {}
        for k2, v2 in v1.items():
            if torch.is_tensor(v2):
                tensors[f"{k1}.{k2}"] = v2
            else:
                metadata[k1][k2] = v2

    with fsspec.open(f"{path}/optimizer.safetensors", "wb") as f:
        f.write(save(tensors))

    with fsspec.open(f"{path}/optimizer.json", "w") as f:
        payload = {"param_groups": state_dict["param_groups"], "metadata": metadata}
        f.write(json.dumps(payload))


def load_optimizer(path: str, optimizer: Optimizer) -> None:
    with fsspec.open(f"{path}/optimizer.safetensors", "rb") as f:
        tensors = load(f.read())

    with fsspec.open(f"{path}/optimizer.json", "r") as f:
        payload = json.loads(f.read())

    state_dict = {"state": {}, "param_groups": payload["param_groups"]}

    for k, v in tensors.items():
        k1, k2 = k.split(".")
        state_dict["state"].setdefault(int(k1), {})[k2] = v

    for k1, v1 in payload["metadata"].items():
        for k2, v2 in v1.items():
            state_dict["state"][int(k1)][k2] = v2

    optimizer.load_state_dict(state_dict)


def save_initial_weights(
    path: str,
    model: nn.Module,
    storage_options: dict | None = None,
) -> None:
    if storage_options is None:
        storage_options = {}

    filepath = f"{path}/initial_weights.safetensors"

    with fsspec.open(filepath, "wb", **storage_options) as f:
        f.write(save(model.state_dict()))


def load_initial_weights(
    path: str,
    model: nn.Module,
    storage_options: dict | None = None,
) -> None:
    if storage_options is None:
        storage_options = {}

    filepath = f"{path}/initial_weights.safetensors"

    with fsspec.open(filepath, "rb", **storage_options) as f:
        model.load_state_dict(load(f.read()))


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    scaler: torch.GradScaler | None = None,
    storage_options: dict | None = None,
) -> None:
    if storage_options is None:
        storage_options = {}

    cp_path = f"{path}/checkpoint-{epoch}"

    with fsspec.open(f"{cp_path}/model.safetensors", "wb", **storage_options) as f:
        f.write(save(model.state_dict()))

    with fsspec.open(f"{cp_path}/optimizer.pt", "wb", **storage_options) as f:
        torch.save(optimizer.state_dict(), f)

    if scheduler is not None:
        with fsspec.open(f"{cp_path}/scheduler.pt", "wb", **storage_options) as f:
            torch.save(scheduler.state_dict(), f)

    if scaler is not None:
        with fsspec.open(f"{cp_path}/scaler.pt", "wb", **storage_options) as f:
            torch.save(scaler.state_dict(), f)


def load_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.GradScaler,
    storage_options: dict | None = None,
) -> None:
    if storage_options is None:
        storage_options = {}

    cp_path = f"{path}/checkpoint-{epoch}"

    with fsspec.open(f"{cp_path}/model.safetensors", "rb", **storage_options) as f:
        model.load_state_dict(load(f.read()))

    with fsspec.open(f"{cp_path}/optimizer.pt", "rb", **storage_options) as f:
        optimizer.load_state_dict(torch.load(f, weights_only=True))

    with fsspec.open(f"{cp_path}/scheduler.pt", "rb", **storage_options) as f:
        scheduler.load_state_dict(torch.load(f, weights_only=True))

    with fsspec.open(f"{cp_path}/scaler.pt", "rb", **storage_options) as f:
        scaler.load_state_dict(torch.load(f, weights_only=True))


def load_model(
    path: str,
    epoch: int,
    model: nn.Module,
    storage_options: dict | None = None,
) -> None:
    if storage_options is None:
        storage_options = {}

    cp_path = f"{path}/checkpoint-{epoch}"

    with fsspec.open(f"{cp_path}/model.safetensors", "rb", **storage_options) as f:
        model.load_state_dict(load(f.read()))


class TestTorchUtils:
    @staticmethod
    def test_save_load_initial_weights() -> None:
        model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        model_t = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))

        assert str(model_t.state_dict()) != str(model.state_dict())

        with TemporaryDirectory() as temp_dir:
            save_initial_weights(temp_dir, model)
            load_initial_weights(temp_dir, model_t)

        assert str(model_t.state_dict()) == str(model.state_dict())

    @staticmethod
    def test_save_load_checkpoint() -> None:
        epoch = 1
        model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        scaler = torch.GradScaler("cpu", growth_factor=2.5)

        model_t = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        optimizer_t = optim.Adam(model.parameters())
        scheduler_t = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler_t = torch.GradScaler("cpu", growth_factor=1.5)

        # Ensure that they are different before loading the checkpoint.
        assert str(model_t.state_dict()) != str(model.state_dict())
        assert str(optimizer_t.state_dict()) != str(optimizer.state_dict())
        assert str(scheduler_t.state_dict()) != str(scheduler.state_dict())
        assert str(scaler_t.state_dict()) != str(scaler.state_dict())

        with TemporaryDirectory() as temp_dir:
            save_checkpoint(temp_dir, epoch, model, optimizer, scheduler, scaler)

            fs = fsspec.filesystem("file")
            assert fs.exists(f"{temp_dir}/checkpoint-{epoch}")

            load_checkpoint(
                temp_dir,
                epoch,
                model_t,
                optimizer_t,
                scheduler_t,
                scaler_t,
            )

        assert str(model_t.state_dict()) == str(model.state_dict())
        assert str(optimizer_t.state_dict()) == str(optimizer.state_dict())
        assert str(scheduler_t.state_dict()) == str(scheduler.state_dict())
        assert str(scaler_t.state_dict()) == str(scaler.state_dict())
