"""Immutable caches, common particle schema and provenance-preserving imports."""
from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

from . import physics
from .protocol import Protocol, fingerprint, file_hash


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz(path, *, meta, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, meta=json.dumps(meta, ensure_ascii=False, allow_nan=False), **arrays)
    temporary.replace(path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["meta"].item())
        arrays = {key: data[key].copy() for key in data.files if key != "meta"}
    return arrays, meta


def generator_hash():
    return fingerprint({name: file_hash(Path(__file__).with_name(name)) for name in ("physics.py", "protocol.py", "data.py")})


def prior(protocol, n, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, np.pi, (n, protocol.dimension)) if protocol.periodic
            else physics.unit_ball(n, protocol.dimension, rng)).astype(np.float32)


def _prepare_case(protocol, out, identity, split, index, level, count, seed, shared, device):
    case_seed = seed + 1009 * index
    case_identity = {**identity, "split": split, "level": float(level), "particles": count, "seed": case_seed}
    key = fingerprint(case_identity)
    path = out / "clouds" / f"{split}_{index:02d}_{key[:20]}.npz"
    if path.exists():
        arrays, meta = read_npz(path)
        if meta.get("cache_identity") != case_identity:
            raise ValueError("Cache identity mismatch")
        physics.check_cloud(arrays["target"], protocol.dimension, protocol.periodic)
        print(f"{protocol.name} {split} level={level:g}: verified cached particles", flush=True)
    else:
        print(f"{protocol.name} {split} level={level:g}: generating {count} particles", flush=True)
        if protocol.periodic:
            def progress(g, total):
                write_json(out / "progress" / f"{split}_{index:02d}.json",
                           {"protocol": protocol.name, "split": split, "level": level,
                            "generation": g, "total_generations": total})
                print(f"  {protocol.name} {split} s={level:g}: FK {g}/{total}", flush=True)
            result = physics.kpp_fk(level, protocol, seed=case_seed, particles=count, progress=progress)
        else:
            result = physics.ks_simulate(shared, level, protocol, seed=case_seed, device=device)
        initial, target = result.pop("initial"), result.pop("target")
        meta = {"cache_identity": case_identity, "protocol": protocol.metadata(),
                "endpoint_pairing": "unpaired FK marginals" if protocol.periodic else "shared physical initial particles",
                **result}
        source = prior(protocol, count, seed + 400003) if protocol.periodic else shared
        write_npz(path, meta=meta, initial=initial, target=target, prior=source)
    return split, index, {"level": float(level), "parameter": protocol.parameter(level),
                          "file": path.relative_to(out).as_posix(), "sha256": file_hash(path)}


def prepare(protocol, out, *, train_seed=666, eval_seed=999,
            train_levels=None, eval_levels=None, device="cpu", purpose="experiment", workers=1):
    out = Path(out)
    train_levels = protocol.train_levels if train_levels is None else list(train_levels)
    eval_levels = protocol.eval_levels if eval_levels is None else list(eval_levels)
    if not train_levels or not eval_levels or train_seed == eval_seed:
        raise ValueError("Use nonempty grids and different training/evaluation physical seeds")
    if workers < 1 or (workers > 1 and not protocol.periodic):
        raise ValueError("Positive worker count required; parallel preparation is for independent CPU KPP cases only")
    for value in train_levels + eval_levels:
        protocol.parameter(value)
    import torch
    backend = {"numpy": np.__version__, "torch": str(torch.__version__),
               "simulation": "numpy_cpu" if protocol.periodic else f"torch_float64_{device}"}
    identity = {"protocol": protocol.metadata(), "train_seed": train_seed, "eval_seed": eval_seed,
                "train_levels": train_levels, "eval_levels": eval_levels,
                "generator_hash": generator_hash(), "backend": backend, "purpose": purpose}
    signature = fingerprint(identity)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["signature"] != signature:
            raise ValueError("Output already belongs to different physical parameters or generator; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"signature": signature, **identity, "evidence": protocol.evidence(),
                "train": [], "eval": [], "status": "preparing"}
    write_json(manifest_path, manifest)
    jobs = []
    for split, levels, count, seed in (("train", train_levels, protocol.n_train, train_seed),
                                       ("eval", eval_levels, protocol.n_eval, eval_seed)):
        shared = None if protocol.periodic else prior(protocol, count, seed)
        for index, level in enumerate(levels):
            jobs.append((protocol, out, identity, split, index, level, count, seed, shared, device))
    completed = {"train": {}, "eval": {}}
    def record(result):
        split, index, item = result
        completed[split][index] = item
        manifest[split] = [entry for _, entry in sorted(completed[split].items())]
        write_json(manifest_path, manifest)
        write_json(out / "progress.json", {"protocol": protocol.name,
                   "completed_cases": sum(len(values) for values in completed.values()),
                   "total_cases": len(jobs), "workers": workers})
    if workers == 1:
        for job in jobs:
            record(_prepare_case(*job))
    else:
        # Each case owns its RNG and cache/progress paths. Completion order never
        # changes case seeds or the ordered train/evaluation manifest.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_prepare_case, *job) for job in jobs]
            for future in as_completed(futures):
                record(future.result())
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    return manifest_path


def load_dataset(path):
    path = Path(path)
    root = path.parent if path.name == "manifest.json" else path
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Dataset is incomplete; resume its preparation before training")
    fields = asdict(Protocol("kpp2d", 2, "kpp", 2. ** -8))
    protocol = Protocol(**{name: manifest["protocol"][name] for name in fields})
    splits = {}
    for split in ("train", "eval"):
        splits[split] = []
        for item in manifest[split]:
            resolved = (root / item["file"]).resolve()
            if not resolved.is_relative_to(root.resolve()) or file_hash(resolved) != item["sha256"]:
                raise ValueError("Particle file escaped the dataset or its checksum differs")
            arrays, meta = read_npz(resolved)
            physics.check_cloud(arrays["target"], protocol.dimension, protocol.periodic)
            splits[split].append({**item, **arrays, "meta": meta})
    return protocol, manifest, splits


def import_archive(archive, out):
    """Keep all supplied caches and code as source evidence, outside canonical data.

    No archived Python is executed. Filename/config annotations do not certify
    which generator produced a cloud. Unresolved historical cases remain marked.
    """
    archive, out = Path(archive), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Archive import requires a new destination")
    out.mkdir(parents=True, exist_ok=True)
    index = {"source_archive": str(archive.resolve()), "source_sha256": file_hash(archive),
             "canonical": False, "items": [], "code": []}
    allowed = {"kpp2d", "kpp3d", "kpp3d_old_backup", "kpp3d_cache_test"}
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            parts = Path(entry.filename.replace("\\", "/")).parts
            if entry.filename.endswith(".npz"):
                group = parts[-2]
                if group not in allowed:
                    raise ValueError(f"Unexpected particle group {group}")
                raw = z.read(entry)
                with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                    meta = json.loads(data["meta"].item())
                    key = "qT" if "qT" in data else "xT"
                    cloud = physics.check_cloud(data[key], 2 if group == "kpp2d" else 3)
                    if np.any(cloud < 0) or np.any(cloud >= physics.TWO_PI):
                        raise ValueError("Archive contains particles outside its declared torus")
                    path = out / group / parts[-1]
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(raw)
                    index["items"].append({"group": group, "file": path.relative_to(out).as_posix(),
                        "source_entry": entry.filename, "sha256": hashlib.sha256(raw).hexdigest(),
                        "shape": list(cloud.shape), "level": -float(meta["log2sigma"]),
                        "cfg": meta["cfg"], "c_hat": meta["c_hat"], "status": "historical_protocol_requires_alignment",
                        "differences": ["resampling interval 256 instead of canonical 1", "old eigenvalue offset kappa instead of reaction_rate"]
                                       + (["3D sign/version provenance unresolved"] if group != "kpp2d" else [])})
            elif entry.filename.endswith(".py"):
                path = out / "source_snapshot" / parts[-1]
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(z.read(entry))
                index["code"].append({"file": path.relative_to(out).as_posix(), "sha256": file_hash(path)})
    write_json(out / "index.json", index)
    return out / "index.json"


def import_current_ks(run, out):
    """Adapt the audited existing KS3D data without changing cached tensors."""
    import torch
    run, out = Path(run), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("KS import requires a new destination")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    old = config["physics"]
    if old.get("physics_id") != "mainv2_positive_regularized_newtonian_current_gradient_v1":
        raise ValueError("Legacy KS kernel is not the current audited physical model")
    from .protocol import preset
    from dataclasses import replace
    training = torch.load(run / "training_data.pt", map_location="cpu", weights_only=True)
    evaluation = torch.load(run / "evaluation_data.pt", map_location="cpu", weights_only=True)
    if evaluation.get("physics") != old:
        raise ValueError("Training and evaluation physics differ")
    p = replace(preset("ks3d"), mu=old["mu"], chi=old["chi"], delta=old["delta"],
                final_time=old["final_time"], dt=old["dt"], n_train=len(training["shared_initial"]),
                n_eval=len(evaluation["cases"][0]["reference"]))
    provenance = {"source_run": str(run.resolve()), "source_files": {
        name: file_hash(run / name) for name in ("config.json", "training_data.pt", "evaluation_data.pt")},
        "source_physics": old}
    identity = {"protocol": p.metadata(), "purpose": "imported_existing_KS", "provenance": provenance}
    manifest = {**identity, "signature": fingerprint(identity), "evidence": p.evidence(), "status": "complete", "train": [], "eval": []}
    for split in ("train", "eval"):
        cases = ([{"level": float(s), "initial": training["shared_initial"].numpy(),
                   "prior": training["shared_initial"].numpy(), "target": target.numpy()}
                  for s, target in zip(training["sigmas"], training["target"])]) if split == "train" else [
                    {"level": c["sigma"], "initial": c.get("reference_initial", c["source"]).numpy(),
                     "prior": c["source"].numpy(), "target": c["reference"].numpy()} for c in evaluation["cases"]]
        for i, case in enumerate(cases):
            level = float(case.pop("level"))
            path = out / "clouds" / f"{split}_{i:02d}.npz"
            for array in case.values():
                physics.check_cloud(array, 3)
            write_npz(path, meta={"protocol": p.metadata(), "provenance": provenance, "trace": []}, **case)
            manifest[split].append({"level": level, "parameter": level, "file": path.relative_to(out).as_posix(), "sha256": file_hash(path)})
    write_json(out / "manifest.json", manifest)
    return out / "manifest.json"
