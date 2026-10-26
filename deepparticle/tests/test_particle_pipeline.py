"""Scientific invariants and a real MF/DP/FK integration check."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from particle_pipeline.protocol import preset, fingerprint
from particle_pipeline.physics import kpp_drift_potential, kpp_fk, ks_simulate, unit_ball
from particle_pipeline.geometry import emd_squared, exact_assignment, difference, costs
from particle_pipeline.models import Corrector
from particle_pipeline.data import prepare, load_dataset
from particle_pipeline.cli import parser
from particle_pipeline.engine import train, evaluate


class PhysicalContracts(unittest.TestCase):
    def test_memory_efficient_costs_preserve_ground_distances(self):
        rng = np.random.default_rng(44)
        for dimension in (2, 3, 4):
            a, b = rng.normal(size=(7, dimension)), rng.normal(size=(9, dimension))
            for periodic in (False, True):
                expected = np.square(difference(a[:, None], b[None], periodic)).sum(-1)
                np.testing.assert_allclose(costs(a, b, periodic), expected, rtol=1e-14, atol=1e-14)

    def test_fk_ensemble_matches_scalar_with_paired_numpy_noise(self):
        from particle_pipeline.fk_batch import kpp_fk_batch
        for name in ("kpp2d", "kpp3d"):
            p = replace(preset(name), dt=.125, generations=2, n_eval=64)
            jobs = [{"protocol": p, "particles": 64, "generations": 2, "level": level,
                     "fk_seed": seed, "cloud": cloud} for level, seed, cloud in
                    [(3., 11, None), (4.5, 12, np.full((64, p.dimension), .5)), (3., 11, None)]]
            serial = [kpp_fk(j["level"], p, particles=64, generations=2, seed=j["fk_seed"], initial=j["cloud"]) for j in jobs]
            for device in (("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)):
                ensemble = kpp_fk_batch(jobs, device=device)
                for result, (_, batched) in zip(serial, ensemble):
                    np.testing.assert_allclose(result["target"], batched["target"], rtol=0, atol=1e-6)
                    self.assertAlmostEqual(result["c_hat"], batched["c_hat"], places=11)
                np.testing.assert_array_equal(ensemble[0][1]["target"], ensemble[2][1]["target"])

    def test_three_dimensional_signed_operator_matches_manuscript(self):
        p = preset("kpp3d")
        drift, potential = kpp_drift_potential(np.array([[0., 0., np.pi / 2]]), 0., 3., p)
        np.testing.assert_allclose(drift, [[1.5, 0., 0.]], atol=1e-14)
        np.testing.assert_allclose(potential, [3.5], atol=1e-14)

    def test_no_flow_fk_has_analytic_eigenvalue_and_no_double_constant(self):
        p = replace(preset("kpp3d"), flow_amplitude=0., dt=.125, generations=3, n_eval=32)
        result = kpp_fk(3., p)
        self.assertAlmostEqual(result["mu_hat"], 1.5, places=12)
        self.assertAlmostEqual(result["c_hat"], .75, places=12)
        self.assertEqual([item["generation"] for item in result["trace"]], [1, 2, 3])

    def test_reaction_changes_eigenvalue_but_not_normalized_cloud(self):
        p = replace(preset("kpp2d"), dt=.125, generations=2, n_eval=32)
        first = kpp_fk(3., p, seed=11)
        second = kpp_fk(3., replace(p, reaction_rate=2.), seed=11)
        np.testing.assert_array_equal(first["target"], second["target"])
        self.assertAlmostEqual(second["mu_hat"] - first["mu_hat"], 1., places=12)

    def test_warm_start_is_used_and_source_not_mutated(self):
        p = replace(preset("kpp3d"), dt=.125, generations=1, n_eval=8)
        initial = np.full((8, 3), .5)
        saved = initial.copy()
        result = kpp_fk(3., p, initial=initial)
        np.testing.assert_array_equal(result["initial"], initial)
        np.testing.assert_array_equal(initial, saved)
        with self.assertRaises(ValueError):
            kpp_fk(3., p, initial=np.zeros((7, 3)))

    def test_periodic_distance_crosses_boundary_and_ot_transports_all_mass(self):
        a = np.array([[.05, .1], [.1, .2]])
        b = a + [2 * np.pi - .1, 0]
        self.assertAlmostEqual(emd_squared(a, b, True), .01, places=12)
        self.assertGreater(emd_squared(a, b, False), 30.)
        columns = exact_assignment(a, b[::-1], True)
        self.assertEqual(sorted(columns.tolist()), [0, 1])
        self.assertAlmostEqual(emd_squared([[0., 0.]], [[0., 0.], [4., 4.]]), 16.)

    def test_periodic_corrector_identity_and_geodesic_gradient(self):
        model = Corrector(2, True, width=8, blocks=1)
        points = torch.tensor([[.1, .2], [6., 6.1]])
        torch.testing.assert_close(model(points, torch.zeros(2, 1)), points, rtol=0, atol=0)
        x = torch.tensor([[.05]], requires_grad=True)
        error = difference(x, torch.tensor([[2 * np.pi - .05]]), True)
        error.square().sum().backward()
        self.assertAlmostEqual(x.grad.item(), .2, places=5)

    def test_all_ks_dimensions_and_shared_3d_kernel(self):
        for name in ("ks2d", "ks3d", "ks4d"):
            p = preset(name)
            cloud = unit_ball(12, p.dimension, np.random.default_rng(2))
            result = ks_simulate(cloud, 50., p, seed=11, chunk=4)
            self.assertEqual(result["target"].shape, (12, p.dimension))
            self.assertTrue(np.isfinite(result["target"]).all())
        from training.protocol import simulate_original
        p = replace(preset("ks3d"), mu=0.)
        initial = torch.tensor(unit_ball(12, 3, np.random.default_rng(2)), dtype=torch.float64)
        old = simulate_original(initial, 50., mu=0., chi=p.chi, delta=p.delta, chunk=4).numpy()
        new = ks_simulate(initial.numpy(), 50., p, seed=11, chunk=4)["target"]
        np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-6)

    def test_protocol_rejects_lost_weights_and_fingerprint_separates_models(self):
        with self.assertRaises(ValueError):
            replace(preset("kpp3d"), resample_every=3)
        self.assertNotEqual(fingerprint(preset("kpp3d").metadata()),
                            fingerprint(replace(preset("kpp3d"), reaction_rate=2.).metadata()))


class PipelineIntegration(unittest.TestCase):
    def test_original_ks_profile_matches_native_mf_training(self):
        from argparse import Namespace
        from training import core as original
        from particle_pipeline.engine import train_mf
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "old").mkdir()
            (root / "new").mkdir()
            p = replace(preset("ks3d"), n_train=24)
            cloud = unit_ball(24, 3, np.random.default_rng(2)).astype(np.float32)
            target = ks_simulate(cloud, 50., p, seed=11)["target"]
            args = parser().parse_args(["train", "--data", str(root), "--out", str(root),
                                       "--profile", "original_tuned", "--mf-steps", "2", "--dp-steps", "2",
                                       "--batch-size", "8", "--device", "cpu"])
            original.set_seed(88)
            old = original.Meanflow()
            native_args = Namespace(mf_steps=4, batch_size=8, n_train=24, train_sigmas=[50.],
                                    mf_lr=.001, time_mean=-.4, time_std=1., checkpoint_steps=[2], device="cpu")
            original.train_mf(old, torch.from_numpy(cloud), torch.from_numpy(target)[None], native_args, root / "old")
            new, _, _ = train_mf([{"prior": cloud, "target": target, "level": 50.}], p, args, root / "new", "test", 88)
            for key, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(new.condition_mode, "log10_sigma")

    def test_tuned_profile_keeps_pure_and_hybrid_alphas_on_reload(self):
        from particle_pipeline.engine import read_models
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = replace(preset("ks2d"), n_train=24, n_eval=24)
            data_path = prepare(p, root / "data", train_levels=[50.], eval_levels=[160.], purpose="smoke")
            args = parser().parse_args(["train", "--data", str(data_path), "--out", str(root / "run"),
                                       "--profile", "original_tuned", "--mf-steps", "2", "--dp-steps", "2",
                                       "--batch-size", "8", "--width", "8", "--dp-blocks", "4", "--alpha", ".1",
                                       "--hybrid-alpha", "10", "--residual-weight", ".1", "--seeds", "88", "--device", "cpu"])
            run = train(data_path, root / "run", args)
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            models = read_models(run, 88, config, p, "cpu")
            self.assertEqual(models["dp"].alpha, .1)
            self.assertEqual(models["mf_dp"].alpha, 10.)
            self.assertEqual(len(models["mf_dp"].blocks), 4)
            self.assertEqual(models["mf"].condition_mode, "log10_sigma")

    def test_parallel_cases_preserve_physical_seeds_clouds_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = replace(preset("kpp3d"), dt=.125, generations=2, n_train=24, n_eval=24)
            results = []
            for workers in (1, 3):
                path = prepare(p, root / str(workers), train_levels=[2., 3.], eval_levels=[3.75, 4.5],
                               device="cpu", purpose="pilot", workers=workers)
                results.append(load_dataset(path))
            self.assertEqual(results[0][1]["signature"], results[1][1]["signature"])
            for split in ("train", "eval"):
                for serial, parallel in zip(results[0][2][split], results[1][2][split]):
                    self.assertEqual(serial["level"], parallel["level"])
                    for field in ("initial", "target", "prior"):
                        np.testing.assert_array_equal(serial[field], parallel[field])
                    self.assertEqual(serial["meta"]["c_hat"], parallel["meta"]["c_hat"])

    def test_cache_identity_and_independent_physical_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = replace(preset("kpp2d"), dt=.125, generations=2, n_train=16, n_eval=16)
            path = prepare(p, temporary, train_levels=[3.], eval_levels=[3.], device="cpu", purpose="smoke")
            _, manifest, splits = load_dataset(path)
            self.assertFalse(np.array_equal(splits["train"][0]["target"], splits["eval"][0]["target"]))
            prepare(p, temporary, train_levels=[3.], eval_levels=[3.], device="cpu", purpose="smoke")
            with self.assertRaises(ValueError):
                prepare(replace(p, reaction_rate=2.), temporary, train_levels=[3.], eval_levels=[3.], device="cpu", purpose="smoke")
            cloud_path = Path(temporary) / manifest["train"][0]["file"]
            with cloud_path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_real_training_checkpoint_reload_metrics_and_fk_starts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p = replace(preset("kpp3d"), dt=.125, generations=2, n_train=24, n_eval=24)
            data_path = prepare(p, root / "data", train_levels=[3.], eval_levels=[4.25, 4.5], device="cpu", purpose="smoke")
            args = parser().parse_args(["train", "--data", str(data_path), "--out", str(root / "run"),
                                       "--mf-steps", "2", "--dp-steps", "2", "--batch-size", "8",
                                       "--width", "8", "--dp-blocks", "1", "--seeds", "88", "--device", "cpu"])
            run = train(data_path, root / "run", args)
            eval_args = parser().parse_args(["evaluate", "--run", str(run), "--device", "cpu", "--bins", "3",
                                            "--metric-particles", "8", "--fk", "--fk-generations", "2", "--fk-repeats", "1",
                                            "--fk-levels", "4.25", "4.5", "--fk-workers", "3"])
            report = json.loads(evaluate(run, eval_args).read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["purpose"], "smoke")
            self.assertEqual(report["metrics_rows"], 16)
            self.assertEqual(report["fk_trace_rows"], 20)
            self.assertTrue((run / "evaluation" / "fk_level_4.5.png").is_file())
            self.assertTrue((run / "evaluation" / "fk_level_4.25.png").is_file())
            summary = report["summary"]
            for row in summary:
                self.assertIsNone(row["w2_sd"])
                self.assertGreaterEqual(row["w2_squared_mean"], 0.)


if __name__ == "__main__":
    unittest.main()
