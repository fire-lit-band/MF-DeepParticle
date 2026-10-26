"""Explicit physical models; historical caches never imply model compatibility."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

VERSION = "particle_pipeline_v1"
NAMES = ("kpp2d", "kpp3d", "ks2d", "ks3d", "ks4d")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Protocol:
    name: str
    dimension: int
    family: str
    dt: float
    generations: int = 4096
    final_time: float = .02
    lam: float = 2.
    theta: float = 1.
    reaction_rate: float = 1.
    resample_every: int = 1
    mu: float = 1.
    chi: float = 16 * math.pi
    delta: float = .1
    flow_amplitude: float = 1.
    n_train: int = 20000
    n_eval: int = 20000

    def __post_init__(self):
        if self.name not in NAMES or self.family not in ("kpp", "ks"):
            raise ValueError("Unknown particle model")
        if self.dimension != int(self.name[-2]) or self.family != self.name[:-2]:
            raise ValueError("Model name, family and dimension differ")
        values = (self.dt, self.final_time, self.lam, self.theta, self.reaction_rate,
                  self.mu, self.chi, self.delta, self.flow_amplitude)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Physical parameters must be finite")
        if min(self.dt, self.final_time, self.lam, self.delta) <= 0:
            raise ValueError("Time steps, lambda and regularization must be positive")
        if min(self.reaction_rate, self.mu, self.chi, self.flow_amplitude) < 0:
            raise ValueError("Reaction, diffusion, interaction and amplitude must be nonnegative")
        if min(self.n_train, self.n_eval, self.generations, self.resample_every) < 1:
            raise ValueError("Particle counts, generations and resampling interval must be positive")
        if self.family == "kpp":
            if not math.isclose(round(1 / self.dt) * self.dt, 1., abs_tol=1e-12):
                raise ValueError("KPP dt must divide one forcing period exactly")
            if round(1 / self.dt) % self.resample_every:
                raise ValueError("Resampling interval must divide one period; no weights may be discarded")
        elif not math.isclose(round(self.final_time / self.dt) * self.dt,
                              self.final_time, abs_tol=1e-12):
            raise ValueError("KS dt must divide final_time exactly")

    @property
    def periodic(self):
        return self.family == "kpp"

    @property
    def train_levels(self):
        if self.periodic:
            return [2 + .25 * i for i in range(8)]
        return [15 * 10 ** (i / 7) for i in range(8)]

    @property
    def eval_levels(self):
        return [2 + .25 * i for i in range(11)] if self.periodic else list(range(10, 201, 10))

    @property
    def model_id(self):
        if self.periodic:
            return f"{self.name}_mainv2_plus_drift_plus_potential_reaction_v1"
        if self.dimension == 2:
            return "ks2d_archived_regularized_log_kernel_extension_v1"
        return f"{self.name}_mainv2_current_gradient_regularized_3d_kernel_v1"

    def parameter(self, level):
        level = float(level)
        value = 2. ** (-level) if self.periodic else level
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Physical conditioning parameter must be positive and finite")
        return value

    def condition(self, level):
        return (float(level) - 2) / 1.75 if self.periodic else math.log10(float(level) / 15)

    def metadata(self):
        return {"schema": VERSION, "model_id": self.model_id, **asdict(self),
                "level_definition": "s=-log2(kappa)" if self.periodic else "sigma=advection amplitude",
                "boundary": "torus [0,2*pi)^d" if self.periodic else "R^d",
                "phase": 0 if self.periodic else None,
                "potential": "kappa*lambda^2 + lambda*v1 + reaction_rate" if self.periodic else None,
                "eigenvalue": "sum log(mean weights)/elapsed physical time; no extra constant" if self.periodic else None,
                "resampler": "multinomial inverse CDF" if self.periodic else None}

    def evidence(self):
        return {"parameter_source": "Protocol dataclass defaults and explicit constructor overrides; preset() defines benchmark defaults",
                "configuration_fields": (["flow_amplitude", "lam", "theta", "dt", "n_train", "n_eval", "train_levels", "eval_levels", "resample_every"]
                                         if self.periodic else ["mu", "chi", "delta", "flow_amplitude", "train_levels", "eval_levels", "n_train", "final_time", "dt"]),
                "adopted_values": ({"reaction_rate": self.reaction_rate}
                                   if self.periodic else {"mu": self.mu, "chi": self.chi, "delta": self.delta}),
                "extension": ("Two-dimensional regularized logarithmic interaction kernel with explicitly configured parameters"
                              if self.name == "ks2d" else
                              "Regularized three-dimensional interaction kernel evaluated in four coordinates"
                              if self.name == "ks4d" else None)}


def preset(name):
    if name not in NAMES:
        raise ValueError(f"Choose one of {NAMES}")
    family, dimension = name[:-2], int(name[-2])
    return Protocol(name, dimension, family, 2. ** -8 if family == "kpp" else .005,
                    n_train=20000 if family == "kpp" else 15000,
                    n_eval=20000 if family == "kpp" else 10000)
