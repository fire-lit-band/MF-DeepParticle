"""100k KPP references using the unchanged, validated uploaded-protocol GPU port."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
DLL = os.add_dll_directory(str(Path(sys.prefix) / 'Library/bin')) if os.name == 'nt' else None
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper_reproduction.portability import verified_source_identity
SOURCE = ROOT / 'output/kpp_original_sampling_20261014/original_gpu_accelerator.py'
spec = importlib.util.spec_from_file_location('unchanged_kpp_gpu_100k', SOURCE)
gpu = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gpu
spec.loader.exec_module(gpu)
LEVELS = [2.0 + .25*i for i in range(11)]
SEED = 10999

def main():
    torch.set_num_threads(1)
    gpu.LEVELS, gpu.SEEDS = tuple(LEVELS), (SEED,)
    source_hash = gpu.sha256(SOURCE)
    validation_path = ROOT/'output/kpp_original_sampling_20261014/gpu_validation.json'
    prior_validation = json.loads(validation_path.read_text(encoding='utf-8'))
    assert prior_validation['status'] == 'passed'
    for filename, digest in prior_validation['source_hashes'].items():
        assert verified_source_identity(gpu.ARCHIVE/filename, root=ROOT)['source_sha256'] == digest
    plan = {'particles_per_reference': 100000, 'generations': 4096, 'levels': LEVELS,
            'physical_seed': SEED, 'accelerator_sha256': source_hash,
            'source_validation': str(validation_path.relative_to(ROOT)),
            'source_validation_sha256': gpu.sha256(validation_path),
            'training_data_reused': False, 'methods_retrained': False,
            'reference_uncertainty': 'One independent physical reference per level; parameter levels share random innovations.'}
    gpu.write_json(OUT/'reference_plan.json', plan)
    for dimension in (2, 3):
        gpu.OUT = OUT/'references'/f'kpp{dimension}d'
        gpu.OUT.mkdir(parents=True, exist_ok=True)
        manifest = gpu.OUT/'manifest.json'
        if manifest.exists():
            record = json.loads(manifest.read_text(encoding='utf-8'))
            assert record['plan'] == plan
            for row in record['outputs']:
                assert gpu.sha256(row['file']) == row['sha256']
            continue
        validation = gpu.validate_dimension(dimension)
        gpu.write_json(gpu.OUT/'validation.json', validation)
        checkpoint = gpu.OUT/f'original_gpu_checkpoint_{dimension}d.pt'
        cached = ROOT/'output/unified_ks_histogram_20261021/references/kpp2d/simulation_checkpoint.pt'
        if dimension == 2 and not checkpoint.exists() and cached.exists():
            old = torch.load(cached, map_location='cpu', weights_only=True)
            identity = old['identity']
            assert identity['source_sha256'] == source_hash
            assert identity['dimension'] == 2 and identity['particles'] == 100000
            assert identity['levels'] == LEVELS and identity['periods'] == 4096
            assert identity['physical_seeds'] == [10999, 110999]
            assert 0 < old['generation'] < 4096
            # Original jobs are seed-major. Retain the complete 100k-particle
            # ensemble and RNG stream of seed10999, across all eleven levels.
            state = {key: ([value[0].clone()] if key == 'rng' else value[:11].clone())
                     for key, value in old['state'].items()}
            protocol = gpu.protocol_record(2, 100000, 4096)
            digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            torch.save({'identity': digest, 'generation': old['generation'], 'state': state,
                        'traces': [[] for _ in LEVELS], 'seconds': 0.0}, checkpoint)
            gpu.write_json(gpu.OUT/'resume_provenance.json', {
                'checkpoint': str(cached.relative_to(ROOT)), 'sha256': gpu.sha256(cached),
                'generation': old['generation'], 'seed': SEED,
                'all_particles_retained': 100000, 'source_hash_matches': True,
                'selection': 'first 11 complete ensembles and first RNG; no particle subsampling',
                'old_directory_status': 'cancelled earlier; only validated simulation state reused, never old metric summaries'})
            print(f'Resuming 2D reference at period {old["generation"]}/4096.', flush=True)
            del old, state
        started = time.perf_counter()
        outputs = gpu.run_dimension(dimension, 100000, 4096, validation)
        assert len(outputs) == 11
        gpu.write_json(manifest, {'status': 'complete', 'plan': plan, 'validation': validation,
                                 'outputs': outputs, 'elapsed_this_invocation': time.perf_counter()-started})
    gpu.write_json(OUT/'references_complete.json', {'status':'complete', 'cases':22, 'particles':100000})

if __name__ == '__main__':
    main()
