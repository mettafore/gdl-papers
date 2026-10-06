"""Loss, metric, and benchmark contracts for the MACE reproduction."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

import evaluate
import model
import train

import data


def test_reference_table_matches_approved_target() -> None:
    assert evaluate.REFERENCE_RMSE_MEV == {
        "300K": {"energy": 3.0, "forces": 8.8},
        "600K": {"energy": 9.7, "forces": 21.8},
        "1200K": {"energy": 29.8, "forces": 62.0},
        "dihedral": {"energy": 7.8, "forces": 16.5},
    }


def test_energy_force_loss_uses_published_weights() -> None:
    predicted_energy = torch.tensor([1.0, 3.0])
    target_energy = torch.tensor([1.0, 1.0])
    predicted_forces = torch.tensor([[1.0, 0.0, 0.0]])
    target_forces = torch.zeros((1, 3))
    loss = train.energy_force_loss(
        predicted_energy,
        predicted_forces,
        target_energy,
        target_forces,
        energy_weight=1.0,
        force_weight=1000.0,
    )
    expected_energy_mse = 2.0
    expected_force_mse = 1.0 / 3.0
    assert loss.item() == pytest.approx(
        expected_energy_mse + 1000.0 * expected_force_mse
    )


def test_data_statistics_use_residual_energy_per_atom_and_force_components() -> None:
    first = data.AtomicGraph(
        atomic_numbers=torch.tensor([1, 6]),
        node_attrs=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        positions=torch.zeros((2, 3), dtype=torch.float64),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        energy=torch.tensor(34.0, dtype=torch.float64),
        forces=torch.ones((2, 3), dtype=torch.float64),
    )
    second = data.AtomicGraph(
        atomic_numbers=torch.tensor([8]),
        node_attrs=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64),
        positions=torch.zeros((1, 3), dtype=torch.float64),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        energy=torch.tensor(43.0, dtype=torch.float64),
        forces=torch.full((1, 3), 3.0, dtype=torch.float64),
    )

    mean, rms = train.compute_mean_rms_energy_forces(
        [first, second],
        atomic_energies=torch.tensor([10.0, 20.0, 30.0, 40.0]),
    )

    assert mean == pytest.approx(2.5)
    assert rms == pytest.approx((11.0 / 3.0) ** 0.5)


def test_average_num_neighbors_uses_receiver_counts() -> None:
    graphs = [
        data.AtomicGraph(
            positions=torch.zeros((3, 3)),
            edge_index=torch.tensor([[0, 1, 2], [1, 1, 0]], dtype=torch.long),
        ),
        data.AtomicGraph(
            positions=torch.zeros((3, 3)),
            edge_index=torch.tensor([[0, 1], [1, 1]], dtype=torch.long),
        ),
    ]

    assert train.compute_average_num_neighbors(graphs) == pytest.approx(5.0 / 3.0)


def test_resolve_device_auto_prefers_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert train.resolve_device("auto") == torch.device("cuda")


def test_resolve_device_auto_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert train.resolve_device("auto") == torch.device("cpu")


def test_resolve_device_rejects_unsupported_or_unavailable_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA is not available"):
        train.resolve_device("cuda")
    with pytest.raises(ValueError, match="MPS is not supported"):
        train.resolve_device("mps")


def test_train_config_rejects_nonpositive_train_batch_limit() -> None:
    with pytest.raises(ValueError, match="max_train_batches must be positive"):
        train.TrainConfig(max_train_batches=0)


def test_rmse_uses_components_not_vector_norms() -> None:
    prediction = torch.tensor([[3.0, 0.0, 0.0]])
    target = torch.zeros((1, 3))
    assert evaluate.root_mean_square_error(prediction, target) == pytest.approx(3**0.5)


def test_evaluate_graphs_batches_records_and_reports_rmse_in_mev() -> None:
    class ZeroPredictionModel(torch.nn.Module):
        def forward(
            self,
            node_attrs: torch.Tensor,
            positions: torch.Tensor,
            edge_index: torch.Tensor,
            graph_index: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del node_attrs, edge_index
            return positions.new_zeros(int(graph_index.max()) + 1), torch.zeros_like(
                positions
            )

    graphs = [
        data.AtomicGraph(
            atomic_numbers=torch.tensor([1]),
            node_attrs=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
            positions=torch.zeros((1, 3), dtype=torch.float64),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            energy=torch.tensor(2.0, dtype=torch.float64),
            forces=torch.zeros((1, 3), dtype=torch.float64),
        ),
        data.AtomicGraph(
            atomic_numbers=torch.tensor([6]),
            node_attrs=torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64),
            positions=torch.ones((1, 3), dtype=torch.float64),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            energy=torch.tensor(4.0, dtype=torch.float64),
            forces=torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64),
        ),
    ]

    rmse_mev = evaluate._evaluate_graphs(ZeroPredictionModel(), graphs, batch_size=1)

    assert rmse_mev == pytest.approx(
        {"energy": 10**0.5 * 1000.0, "forces": 1.5**0.5 * 1000.0}
    )


def test_ema_model_tracks_live_parameters_with_configured_decay() -> None:
    live_model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        live_model.weight.fill_(2.0)

    ema_model = train.make_ema_model(live_model, decay=0.99)
    ema_model.update_parameters(live_model)
    with torch.no_grad():
        live_model.weight.fill_(3.0)
    ema_model.update_parameters(live_model)

    assert ema_model.module.weight.item() == pytest.approx(2.01)


def test_ema_counter_is_created_on_the_model_device() -> None:
    model_on_meta = torch.nn.Linear(1, 1).to("meta")

    ema_model = train.make_ema_model(model_on_meta, decay=0.99)

    assert ema_model.n_averaged.device == torch.device("meta")


def test_validation_loss_uses_the_model_passed_to_it() -> None:
    class ConstantEnergyModel(torch.nn.Module):
        def __init__(self, energy: float) -> None:
            super().__init__()
            self.energy = torch.nn.Parameter(torch.tensor(energy, dtype=torch.float64))

        def forward(
            self,
            node_attrs: torch.Tensor,
            positions: torch.Tensor,
            edge_index: torch.Tensor,
            graph_index: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del node_attrs, edge_index
            num_graphs = int(graph_index.max().item()) + 1
            return self.energy.expand(num_graphs), torch.zeros_like(positions)

    graph = data.AtomicGraph(
        node_attrs=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        positions=torch.zeros((1, 3), dtype=torch.float64),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        energy=torch.tensor(0.0, dtype=torch.float64),
        forces=torch.zeros((1, 3), dtype=torch.float64),
    )
    loader = data.get_data_loader([graph], batch_size=1, shuffle=False)
    ema_model = ConstantEnergyModel(energy=2.0)

    validation_loss = train.compute_validation_loss(
        ema_model,
        loader,
        energy_weight=1.0,
        force_weight=0.0,
    )

    assert validation_loss == pytest.approx(4.0)


def test_short_train_run_can_reload_checkpoint_and_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = []
    for index in range(11):
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0 + 0.02 * index, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        records.append(
            data.AtomicConfiguration(
                atomic_numbers=torch.tensor([1, 6, 8], dtype=torch.long),
                positions=positions,
                energy=torch.tensor(-20.0 + 0.1 * index, dtype=torch.float64),
                forces=torch.full((3, 3), 0.2, dtype=torch.float64),
            )
        )

    monkeypatch.setattr(data, "load_extxyz", lambda _: records)
    monkeypatch.setattr(
        data,
        "split_train_valid",
        lambda configurations, *, seed: (configurations[:10], configurations[10:]),
    )
    monkeypatch.setattr(
        data,
        "load_isolated_atom_energies",
        lambda _: torch.zeros(4, dtype=torch.float64),
    )
    monkeypatch.setattr(data, "validate_data_dir", lambda _: None)
    monkeypatch.setattr(data, "TEST_FILES", {"300K": ("synthetic.xyz",)})

    optimizer_steps = 0
    adam_step = torch.optim.Adam.step

    def count_adam_step(optimizer, *args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return adam_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", count_adam_step)

    run_id, run_path = train.train(
        data_dir=tmp_path,
        paper_dir=tmp_path,
        device="cpu",
        train_config=train.TrainConfig(
            max_epochs=1,
            batch_size=5,
            max_train_batches=1,
        ),
        model_config=model.MACEConfig(
            channels=2,
            radial_mlp_widths=(4, 4, 4, 8),
            readout_hidden_width=2,
        ),
    )
    metrics = evaluate.evaluate(
        run_id, data_dir=tmp_path, paper_dir=tmp_path, device="cpu"
    )

    assert (run_path / "config.json").is_file()
    assert (run_path / "metrics.jsonl").is_file()
    assert (run_path / "checkpoint.pt").is_file()
    assert '"device": "cpu"' in (run_path / "config.json").read_text()
    assert optimizer_steps == 1
    assert set(metrics) == {"300K"}
    assert all(
        torch.isfinite(torch.tensor(value)) for value in metrics["300K"].values()
    )
