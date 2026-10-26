"""Train the existing MF/DP methods on explicitly versioned uploaded KPP data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ORIGINAL = ROOT / "output/kpp_original_sampling_20261014"
ARCHIVE = ROOT / "deepparticle/data/particle_sources/kpp_archive_20261012/source_snapshot"
sys.path.insert(0, str(ROOT / "deepparticle"))
from particle_pipeline import engine
from particle_pipeline.data import prior, read_npz, write_json, write_npz
from particle_pipeline.protocol import Protocol, file_hash, fingerprint
from paper_reproduction.portability import protocols_match, resolve_recorded_path, verified_source_identity

TRAIN_LEVELS = [2 + .25 * i for i in range(8)]
EVAL_LEVELS = [2 + .25 * i for i in range(11)]
MODEL_SEEDS = [88, 89, 90]


def now():
    return datetime.now(timezone.utc).isoformat()


class UploadedProtocol(Protocol):
    @property
    def model_id(self):
        sign = "plus" if self.dimension == 2 else "minus"
        return f"{self.name}_uploaded_plus_drift_{sign}_potential_kappa_offset_resample256_v1"

    def metadata(self):
        meta = super().metadata()
        source = ARCHIVE / ("kpp_common.py" if self.dimension == 2 else "kpp3d_kolmogorov.py")
        return {**meta, "schema": "uploaded_kpp_training_v1", "model_id": self.model_id,
                "potential": "kappa*lambda^2 + lambda*v1" if self.dimension == 2 else "kappa*lambda^2 - lambda*v1",
                "eigenvalue": "kappa + sum log(mean weights)/elapsed time",
                "reaction_in_weights": 0., "time_direction": "forward j*dt",
                **verified_source_identity(source, root=ROOT),
                "resampler": "multinomial inverse CDF, once per forcing period"}

    def evidence(self):
        return {"basis": "Actual executed user-uploaded source, not canonical mainV2 physics",
                "source": self.metadata()["source_file"], "source_sha256": self.metadata()["source_sha256"],
                "preserved": ["2D time-dependent flow", "3D positive drift and negative potential",
                              "no reaction term in weights", "kappa offset", "resample every 256 substeps"],
                "previous_material": str(ORIGINAL / "gpu_manifest.json")}


def protocol(model):
    return UploadedProtocol(model, int(model[-2]), "kpp", 2. ** -8,
                            reaction_rate=0., resample_every=256,
                            generations=4096, n_train=20000, n_eval=20000)


def load_accelerator():
    path = ORIGINAL / "original_gpu_accelerator.py"
    spec = importlib.util.spec_from_file_location("uploaded_gpu_for_training", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_old_files():
    manifest = json.loads((ORIGINAL / "gpu_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or len(manifest["outputs"]) != 8:
        raise ValueError("Representative original reference is incomplete")
    for item in manifest["outputs"]:
        if file_hash(Path(item["file"])) != item["sha256"]:
            raise ValueError("Representative source reference changed")


def find_raw(model, level, seed):
    suffix = str(float(level)).replace(".", "p")
    name = f"original_gpu_{model}_s{suffix}_seed{seed}.npz"
    roots = (ORIGINAL, OUT / "additional_raw")
    matches = [root / name for root in roots if (root / name).exists()]
    if len(matches) != 1:
        raise ValueError(f"Raw cloud missing or duplicated: {name}")
    return matches[0]


def export_dataset(model):
    p = protocol(model)
    folder = OUT / "datasets" / model
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"protocol": p.metadata(), "train_levels": TRAIN_LEVELS, "eval_levels": EVAL_LEVELS,
                "train_physical_seed": 666, "eval_physical_seed": 999,
                "prior": "Normal(0,pi), distinct training/evaluation pools; same pool across kappa",
                "source": "Validated GPU port of the unmodified uploaded source, independent Philox streams by seed"}
    manifest = {"identity": identity, "protocol": p.metadata(), "purpose": "experiment",
                "evidence": p.evidence(), "status": "preparing", "train": [], "eval": []}
    for split, levels, seed in (("train", TRAIN_LEVELS, 666), ("eval", EVAL_LEVELS, 999)):
        for i, level in enumerate(levels):
            source = find_raw(model, level, seed)
            arrays, raw_meta = read_npz(source)
            if any(raw_meta[k] != value for k, value in {
                    "model": model, "level": level, "seed": seed,
                    "generations": 4096, "particles": 20000}.items()):
                raise ValueError("Raw cloud has different generation settings")
            if raw_meta["source_hash"] != p.metadata()["source_sha256"]:
                raise ValueError("Raw cloud source differs")
            if raw_meta["protocol"]["resample_every"] != 256:
                raise ValueError("Raw cloud has different resampling")
            if p.dimension == 3 and raw_meta["protocol"]["potential"] != "kappa*lambda^2 - lambda*v1":
                raise ValueError("3D reference potential sign differs")
            dest = folder / "clouds" / f"{split}_{i:02d}.npz"
            meta = {"protocol": p.metadata(), "source_file": str(source), "source_sha256": file_hash(source),
                    "source_meta": raw_meta, "split": split, "level": level, "seed": seed,
                    "trace": raw_meta["trace"], "c_hat": raw_meta["c_hat"], "mu_hat": raw_meta["mu_hat"],
                    "endpoint_pairing": "unpaired FK marginals; MF pairs independent Gaussian source and target samples"}
            write_npz(dest, meta=meta, initial=arrays["initial"], target=arrays["target"],
                      prior=prior(p, 20000, seed + 400003))
            manifest[split].append({"level": level, "parameter": p.parameter(level),
                                    "file": dest.relative_to(folder).as_posix(), "sha256": file_hash(dest)})
    manifest["signature"] = fingerprint({"identity": identity, "train": manifest["train"], "eval": manifest["eval"]})
    manifest["status"] = "complete"
    write_json(folder / "manifest.json", manifest)
    load_dataset(folder)
    return manifest["signature"]


def load_dataset(path):
    folder = Path(path)
    if folder.name == "manifest.json":
        folder = folder.parent
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Original-protocol data incomplete")
    expected = fingerprint({k: manifest[k] for k in ("identity", "train", "eval")})
    if manifest["signature"] != expected:
        raise ValueError("Dataset identity changed")
    p = protocol(manifest["protocol"]["name"])
    if not protocols_match(manifest["protocol"], p.metadata(), root=ROOT):
        raise ValueError("Uploaded physical protocol changed")
    splits = {}
    for split in ("train", "eval"):
        cases = []
        for item in manifest[split]:
            # New data runs may live outside the checkout; each cloud remains
            # strictly confined to its own dataset directory.
            target_path = (folder / item["file"].replace("\\", "/")).resolve(strict=True)
            if not target_path.is_relative_to(folder.resolve()) or file_hash(target_path) != item["sha256"]:
                raise ValueError("Dataset file changed or escaped its folder")
            arrays, meta = read_npz(target_path)
            if not protocols_match(meta["protocol"], p.metadata(), root=ROOT) or arrays["target"].shape != (20000, p.dimension):
                raise ValueError("Dataset file protocol or shape differs")
            cases.append({**item, **arrays, "meta": meta})
        splits[split] = cases
    if [c["level"] for c in splits["train"]] != TRAIN_LEVELS or [c["level"] for c in splits["eval"]] != EVAL_LEVELS:
        raise ValueError("Training/evaluation grids differ")
    if {c["meta"]["seed"] for c in splits["train"]} != {666} or {c["meta"]["seed"] for c in splits["eval"]} != {999}:
        raise ValueError("Training/evaluation physical seeds not separated")
    return p, manifest, splits


def prepare():
    torch.set_num_threads(1)
    verify_old_files()
    folder = OUT / "additional_raw"
    folder.mkdir(parents=True, exist_ok=True)
    gpu = load_accelerator()
    gpu.OUT = folder
    gpu.LEVELS = tuple(s for s in EVAL_LEVELS if s not in (3.75, 4.5))
    identity = {"levels": gpu.LEVELS, "seeds": list(gpu.SEEDS), "particles": 20000, "generations": 4096,
                "previous_accelerator_sha256": file_hash(ORIGINAL / "original_gpu_accelerator.py"),
                "wrapper_sha256": file_hash(__file__)}
    write_json(folder / "generation_spec.json", identity)
    outputs = []
    for dimension in (2, 3):
        validation = gpu.validate_dimension(dimension)
        write_json(folder / f"validation_{dimension}d.json", validation)
        print(f"Original KPP{dimension}D additional-level source validation passed", flush=True)
        outputs.extend(gpu.run_dimension(dimension, 20000, 4096, validation))
        signature = export_dataset(f"kpp{dimension}d")
        write_json(OUT / "preparation_progress.json", {"stage": "prepare", "dimension_completed": dimension,
                   "dataset_signature": signature, "additional_cases_completed": len(outputs), "time": now()})
    write_json(folder / "manifest.json", {"status": "complete", "generation_spec": identity, "outputs": outputs})
    print("Both full eight-condition training sets and independent eleven-condition evaluation sets complete", flush=True)


def training_args():
    return SimpleNamespace(mf_steps=500, dp_steps=500, mf_lr=.001, dp_lr=.0001,
                           width=64, mf_width=64, mf_hidden_layers=3, dp_blocks=4,
                           alpha=.1, hybrid_alpha=.5, residual_weight=0.,
                           profile="original_tuned", batch_size=2000, batch_refresh=500,
                           ot_refresh=500, seeds=MODEL_SEEDS, device="cuda")


def evaluation_args():
    return SimpleNamespace(device="cuda", bins=72, metric_particles=2000, metric_seed=999,
                           metric_workers=2, fk=False, fk_generations=128, fk_repeats=3,
                           fk_seed=1234, fk_levels=[3.75, 4.5], fk_tolerance=.01,
                           fk_workers=1, fk_backend="cuda")


def train(model):
    engine.load_dataset = load_dataset
    dataset, run = OUT / "datasets" / model, OUT / model / "run"
    args = training_args()
    write_json(OUT / f"{model}_training_spec.json", {"started": now(), "training": vars(args),
               "budgets": {"mf": 1000, "dp": 1000, "mf_dp": "500 MF + 500 DP"},
               "training_source_sha256": file_hash(ROOT / "deepparticle/particle_pipeline/engine.py"),
               "wrapper_sha256": file_hash(__file__), "physics": protocol(model).metadata()})
    engine.train(dataset, run, args)
    shutil.copyfile(__file__, run / "original_protocol_wrapper.py")
    print(f"{model}: three methods and initializer ablation completed for seeds {MODEL_SEEDS}", flush=True)


def evaluate(model):
    engine.load_dataset = load_dataset
    result = engine.evaluate(OUT / model / "run", evaluation_args())
    print(str(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--model", choices=("kpp2d", "kpp3d"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.model is None:
        parser.error("--model required")
    elif args.command == "train":
        train(args.model)
    else:
        evaluate(args.model)


if __name__ == "__main__":
    main()
