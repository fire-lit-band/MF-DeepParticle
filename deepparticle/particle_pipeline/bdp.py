"""BDP equations (5)--(8), arXiv:2504.11851v1, for KS and KPP data.

Two independent three-hidden-layer tanh MLPs learn forward/reverse maps.
The squared OT objectives use separate assignments and the x -> g(f(x))
consistency penalty. KPP uses explicit torus wrapping and periodic costs;
KS uses its physical initial cloud and Euclidean costs without wrapping.
This is a fresh implementation; unavailable authors' checkpoints are not implied.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import torch
from torch import nn

from .data import load_dataset, write_json
from .engine import checkpoint, csv_write, set_seed, sync
from .geometry import difference, exact_assignment
from .physics import TWO_PI
from .protocol import file_hash


class TransportMLP(nn.Module):
    def __init__(self, dimension, width=40):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension + 1, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, dimension))

    def forward(self, points, condition):
        return self.net(torch.cat((points, condition), -1))


class BidirectionalMap(nn.Module):
    def __init__(self, dimension, width=40, periodic=True):
        super().__init__()
        self.periodic = periodic
        self.forward_map = TransportMLP(dimension, width)
        self.reverse_map = TransportMLP(dimension, width)

    def forward(self, points, condition):
        result = self.forward_map(points, condition)
        return torch.remainder(result, TWO_PI) if self.periodic else result

    def reverse(self, points, condition):
        result = self.reverse_map(points, condition)
        return torch.remainder(result, TWO_PI) if self.periodic else result


def bdp_loss(model, source, target, condition, forward_matched, reverse_matched, cycle_weight):
    forward = model(source, condition)
    reverse = model.reverse(target, condition)
    cycle = model.reverse(forward, condition)
    f = difference(forward, forward_matched, model.periodic).square().sum(-1).mean()
    r = difference(reverse, reverse_matched, model.periodic).square().sum(-1).mean()
    c = difference(cycle, source, model.periodic).square().sum(-1).mean()
    return f + r + cycle_weight * c, f, r, c


def train_bdp(dataset, out, *, seeds=(88, 89, 90), steps=1000, batch_size=2000,
              width=40, lr=.001, cycle_weight=.001, batch_refresh=500, ot_refresh=50, device="cuda"):
    if min(steps, batch_size, width, batch_refresh, ot_refresh) < 1 or lr <= 0 or cycle_weight < 0:
        raise ValueError("Invalid BDP training configuration")
    protocol, manifest, splits = load_dataset(dataset)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("BDP requires nonempty, distinct training seeds")
    periodic = protocol.periodic
    source_description = ("uniform torus; physical initial arrays, unpaired OT" if periodic else
                          "physical initial arrays on R^d; unpaired Euclidean OT")
    out = Path(out)
    config = {"dataset": str(Path(dataset).resolve()), "dataset_signature": manifest["signature"],
              "protocol": protocol.metadata(), "purpose": manifest["purpose"], "steps": steps,
              "model_seeds": list(seeds), "batch_size": batch_size, "width": width, "lr": lr,
              "cycle_weight": cycle_weight, "batch_refresh": batch_refresh, "ot_refresh": ot_refresh,
              "device": device, "source": source_description, "periodic": periodic,
              "loss": "mean forward W2 squared + mean reverse W2 squared + cycle_weight * mean squared cycle distance",
              "architecture": "two independent 3-hidden-layer tanh MLPs with linear outputs; width 40 from BDP paper",
              "primary_source": "https://arxiv.org/html/2504.11851v1#S2.SS2",
              "source_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in ("bdp.py", "geometry.py")},
              "reproduction_scope": "paper-defined bidirectional objective and MLPs; explicit benchmark geometry/source and fixed update budget; no original-author checkpoints implied",
              "extrapolation_used_for_selection": False}
    config_path = out / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if {k: v for k, v in saved.items() if k != "status"} != config:
            raise ValueError("Existing BDP run has different settings")
    out.mkdir(parents=True, exist_ok=True)
    write_json(config_path, {**config, "status": "training"})
    package = Path(__file__).parent
    sources = [package / "bdp.py", package / "geometry.py", package / "protocol.py", package / "engine.py"]
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
        for source in sources:
            snapshot.write(source, source.name)
    write_json(out / "source_manifest.json", {p.name: file_hash(p) for p in sources})
    write_json(out / "runtime.json", {"python": sys.version.split()[0], "torch": str(torch.__version__),
               "numpy": np.__version__, "device": device,
               "gpu": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None})
    model_manifest_path = out / "checkpoint_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8")) if model_manifest_path.exists() else {}
    timings_path = out / "timings.json"
    timings = json.loads(timings_path.read_text(encoding="utf-8")) if timings_path.exists() else []
    for seed in seeds:
        if str(seed) in model_manifest:
            saved_checkpoint = out / model_manifest[str(seed)]["file"]
            if file_hash(saved_checkpoint) != model_manifest[str(seed)]["sha256"]:
                raise ValueError("Saved BDP checkpoint hash differs")
            print(f"{protocol.name} BDP seed={seed}: verified completed training", flush=True)
            continue
        set_seed(seed)
        folder = out / f"seed{seed}"
        folder.mkdir(exist_ok=True)
        model = BidirectionalMap(protocol.dimension, width, periodic).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        pools = [(torch.from_numpy(case["initial"]).to(device), torch.from_numpy(case["target"]).to(device))
                 for case in splits["train"]]
        batches, rows, assignments, solve_seconds = None, [], 0, 0.
        sync(device)
        start = time.perf_counter()
        for step in range(1, steps + 1):
            changed = batches is None or (step - 1) % batch_refresh == 0
            if changed:
                size = min(batch_size, *(len(q) for pair in pools for q in pair))
                si = torch.randperm(len(pools[0][0]))[:size].to(device)
                ti = torch.randperm(len(pools[0][1]))[:size].to(device)
                source = torch.stack([pair[0][si] for pair in pools])
                target = torch.stack([pair[1][ti] for pair in pools])
                condition = torch.tensor([protocol.condition(case["level"]) for case in splits["train"]],
                                         device=device).view(-1, 1, 1).expand(-1, size, 1)
                batches = source, target, condition
            if changed or (step - 1) % ot_refresh == 0:
                with torch.no_grad():
                    forward = model(source, condition).cpu().numpy()
                    reverse = model.reverse(target, condition).cpu().numpy()
                source_np, target_np = source.cpu().numpy(), target.cpu().numpy()
                ot_inputs = [(a, b, periodic) for a, b in zip(forward, target_np)] + [(a, b, periodic) for a, b in zip(reverse, source_np)]
                solve_start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=2) as executor:
                    permutations = list(executor.map(lambda inputs: exact_assignment(*inputs), ot_inputs))
                solve_seconds += time.perf_counter() - solve_start
                assignments += len(permutations)
                f_indices = torch.tensor(np.stack(permutations[:len(pools)]), device=device)
                r_indices = torch.tensor(np.stack(permutations[len(pools):]), device=device)
                forward_matched = target.gather(1, f_indices[:, :, None].expand(-1, -1, protocol.dimension))
                reverse_matched = source.gather(1, r_indices[:, :, None].expand(-1, -1, protocol.dimension))
            loss, forward_loss, reverse_loss, cycle_loss = bdp_loss(
                model, source, target, condition, forward_matched, reverse_matched, cycle_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BDP loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows.append({"step": step, "loss": loss.item(), "forward_w2_squared": forward_loss.item(),
                         "reverse_w2_squared": reverse_loss.item(), "cycle_squared": cycle_loss.item()})
            if step == 1 or step % 250 == 0 or step == steps:
                csv_write(folder / "loss.csv", rows)
                print(f"{protocol.name} BDP weight={cycle_weight:g} seed={seed} {step}/{steps}, loss={loss.item():.6g}", flush=True)
        sync(device)
        checkpoint(folder / "bdp.pt", model, optimizer, step=steps, protocol=protocol,
                   dataset_signature=manifest["signature"],
                   architecture={"kind": "bdp", "width": width, "dimension": protocol.dimension, "periodic": periodic,
                                 "cycle_weight": cycle_weight, "source": source_description})
        model_manifest[str(seed)] = {"file": (folder / "bdp.pt").relative_to(out).as_posix(),
                                    "sha256": file_hash(folder / "bdp.pt"), "step": steps}
        timings.append({"seed": seed, "seconds": time.perf_counter() - start, "ot_seconds": solve_seconds,
                        "ot_assignments": assignments, "parameter_count": sum(p.numel() for p in model.parameters())})
        write_json(model_manifest_path, model_manifest)
        write_json(timings_path, timings)
    write_json(config_path, {**config, "status": "complete"})
    return out


def read_bdp(run, protocol, signature, device):
    run = Path(run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if config["status"] != "complete" or config["dataset_signature"] != signature:
        raise ValueError("BDP run is incomplete or belongs to different data")
    models = {}
    for seed, item in manifest.items():
        path = (run / item["file"]).resolve()
        if not path.is_relative_to(run.resolve()) or file_hash(path) != item["sha256"]:
            raise ValueError("BDP checkpoint escaped run directory or its checksum differs")
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if payload["dataset_signature"] != signature or payload["protocol"] != protocol.metadata() or payload["step"] != config["steps"]:
            raise ValueError("BDP checkpoint identity or training budget differs")
        periodic = config.get("periodic", protocol.periodic)
        if periodic != protocol.periodic or payload["architecture"]["periodic"] != periodic:
            raise ValueError("BDP checkpoint geometry differs from the physical protocol")
        model = BidirectionalMap(protocol.dimension, config["width"], periodic).to(device)
        model.load_state_dict(payload["model"], strict=True)
        models[int(seed)] = model.eval().requires_grad_(False)
    return config, models


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cycle-weight", type=float, default=.001)
    parser.add_argument("--seeds", type=int, nargs="+", default=[88, 89, 90])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--ot-refresh", type=int, default=50)
    args = parser.parse_args()
    train_bdp(args.data, args.out, seeds=args.seeds, steps=args.steps, batch_size=args.batch_size,
              cycle_weight=args.cycle_weight, ot_refresh=args.ot_refresh)


if __name__ == "__main__":
    main()
