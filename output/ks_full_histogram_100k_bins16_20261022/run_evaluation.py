"""Separate, frozen-checkpoint KS evaluation. Never edits manuscript assets."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
DLL = os.add_dll_directory(str(Path(sys.prefix) / 'Library/bin')) if os.name == 'nt' and (Path(sys.prefix) / 'Library/bin').is_dir() else None
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CLOUDS_ROOT = ROOT/'output/ks_full_histogram_100k_20261022/clouds'
sys.path.insert(0, str(ROOT / 'deepparticle'))
sys.path.insert(0, str(ROOT / 'output/dp_dplus1_four_benchmarks_20261016'))
import numpy as np
import ot
import psutil
from scipy.spatial.distance import cdist
import torch
from compact_architecture import CompactCorrector
from particle_pipeline.models import MeanFlow, generate
from particle_pipeline.physics import ks_simulate, unit_ball
from particle_pipeline.protocol import Protocol

BASE = ROOT / 'output/mf_matched_dp_20261016'
COMPACT = ROOT / 'output/dp_dplus1_four_benchmarks_20261016'
CACHE = ROOT / 'output/unified_ks_histogram_20261021'
LEVELS = list(range(10, 201, 10))
SEEDS = (88, 89, 90)
METHODS = ('mf', 'dp', 'mf_dp')
N, BINS = 100000, 16
STARTED = time.perf_counter()


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.new')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def event(stage, **data):
    record = {'stage': stage, 'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.perf_counter() - STARTED, **data}
    if stage != 'solve':
        save(OUT / f'progress_{stage}.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def protocol(name):
    value = js(BASE / name / 'run/config.json')['protocol']
    fields = Protocol.__dataclass_fields__
    return Protocol(**{k: v for k, v in value.items() if k in fields})


def checkpoint_path(name, seed, method):
    if method in ('mf', 'mf_init'):
        return BASE / name / 'run' / f'seed{seed}' / f'{method}.pt'
    return COMPACT / name / 'dplus1' / f'seed{seed}' / method / f'{method}.pt'


def load_models(name):
    p = protocol(name)
    signature = js(BASE / name / 'run/config.json')['dataset_signature']
    result = {}
    for seed in SEEDS:
        models = {}
        for method in (*METHODS, 'mf_init'):
            path = checkpoint_path(name, seed, method)
            item = torch.load(path, map_location='cpu', weights_only=True)
            assert item['dataset_signature'] == signature and item['protocol'] == p.metadata()
            assert item['step'] == (500 if method in ('mf_init', 'mf_dp') else 1000)
            if method in ('mf', 'mf_init'):
                model = MeanFlow(p.dimension, 64, 'log10_sigma', 3)
                expected = js(BASE / name / 'run/checkpoint_manifest.json')[str(seed)][method]['sha256']
            else:
                model = CompactCorrector(p.dimension, alpha=.1 if method == 'dp' else 10., condition_mode='log10_sigma')
                expected = js(path.parent / 'record.json')['checkpoint_sha256']
            assert sha(path) == expected
            model.load_state_dict(item['model'], strict=True)
            models[method] = model.cuda().eval().requires_grad_(False)
        result[seed] = models
    return result


@torch.no_grad()
def predict(source, level, p, models):
    outputs = {}
    for seed in SEEDS:
        for method in METHODS:
            chunks = []
            m = models[seed]
            for start in range(0, len(source), 10000):
                x = torch.from_numpy(source[start:start+10000]).cuda()
                cond = torch.full((len(x), 1), p.condition(level), device='cuda')
                mf_cond = torch.full_like(cond, math.log10(level))
                y = generate(method, x, cond, p, mf=m['mf'] if method == 'mf' else m['mf_init'],
                             dp=m['dp'] if method == 'dp' else m['mf_dp'], mf_condition=mf_cond)
                chunks.append(y.cpu().numpy())
            outputs[f'seed{seed}_{method}'] = np.concatenate(chunks)
    return outputs


def histogram(points, origin, widths):
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and np.isfinite(points).all()
    ids, counts = np.unique(np.floor((points-origin)/widths).astype(np.int64), axis=0, return_counts=True)
    coordinates = np.ascontiguousarray(origin + (ids+.5)*widths, dtype=np.float64)
    weights = counts.astype(np.float64)/len(points)
    assert int(counts.sum()) == len(points) and abs(weights.sum()-1) < 1e-12
    return coordinates, weights


def measures_for(clouds):
    ref = clouds['reference'].astype(np.float64)
    origin, widths = ref.min(0), np.ptp(ref, axis=0)/BINS
    assert np.all(widths > 0)
    measures = {key: histogram(value, origin, widths) for key, value in clouds.items()}
    return measures, origin, widths


def solve(first, second):
    t = time.perf_counter()
    costs = cdist(first[0], second[0], 'sqeuclidean')
    g, info = ot.emd(first[1], second[1], costs, numItermax=20000000, log=True, numThreads=1)
    if info['result_code'] != 1 or info['warning']:
        raise RuntimeError(f"Exact solver did not converge: {info['result_code']}, {info['warning']}")
    value = float(info['cost'])
    residual = max(float(np.max(np.abs(np.asarray(g.sum(1)).ravel()-first[1]))),
                   float(np.max(np.abs(np.asarray(g.sum(0)).ravel()-second[1]))))
    dual = float(first[1] @ info['u'] + second[1] @ info['v'])
    gap = abs(value-dual)
    assert residual < 1e-9 and gap < 1e-8 and np.isfinite(value) and value >= -1e-10
    return {'w2': math.sqrt(max(value, 0.)), 'w2_squared': max(value, 0.),
            'reference_occupied_bins': len(first[0]), 'generated_occupied_bins': len(second[0]),
            'solver_result_code': int(info['result_code']), 'marginal_residual': residual,
            'primal_dual_gap': gap, 'transport_nonzeros': int(np.count_nonzero(g)), 'seconds': time.perf_counter()-t,
            'solver': 'POT dense exact network simplex'}


def setup():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def preflight():
    setup()
    protected = [ROOT/'mainV2.tex', ROOT/'mainV2.pdf', ROOT/'sections/experiment_protocols.tex']
    protected += list((ROOT/'figures/selected_dp_results_20261017').glob('dplus1_ks*'))
    inputs = {}
    for name in ('ks3d', 'ks4d'):
        for seed in SEEDS:
            for method in (*METHODS, 'mf_init'):
                path = checkpoint_path(name, seed, method)
                inputs[str(path.relative_to(ROOT))] = sha(path)
    plan = {'material_passport': {'origin_skill': 'academic-research-suite', 'origin_mode': 'run + validate',
             'date': '2026-10-22', 'verification_status': 'PLANNED', 'version': 'ks-hist100k-bins16-v1'},
            'user_request': 'Evaluate both 3D and 4D KS using 100000-point full-dimensional histogram OT; do not replace manuscript results.',
            'models': ['ks3d', 'ks4d'], 'levels': LEVELS, 'training_seeds': list(SEEDS), 'methods': list(METHODS),
            'particles_per_distribution': N, 'base_bins_per_axis': BINS,
            'grid': f'For EACH case use the 100k reference coordinate min/max divided into {BINS} bins per axis; anchor at reference minimum and extend the same lattice without clipping any generated point. Share the grid across all methods and seeds. Apply this identical construction in 3D and 4D.',
            'distance': 'Unscaled squared Euclidean cost using all coordinates; exact weighted OT, square root reported as W2.',
            'solver': 'POT exact network simplex: dense for 3D and initial 4D rows; sparse column generation with a global dual certificate for remaining 4D rows. Same full squared-Euclidean OT objective, no entropic regularization. All solves check marginals and primal-dual gap. Dense-versus-sparse agreement is recorded in solver_comparison.json.',
            'reuse_clouds': str(CLOUDS_ROOT),
            'grid_choice': 'User chose a complete 16-bin-per-axis comparison after the 24-bin 4D evaluation proved too slow. Same 100k clouds retained.',
            'ks3d': 'Reuse frozen 100k reference and generated clouds from current manuscript checkpoints; recompute under the common reference-anchored grid rule (original manuscript uses its historical anchored grid).',
            'ks4d': 'Shared fresh 100k uniform-unit-ball source, seed10999. Same physical scheme as current benchmark, 100k interacting particles, Brownian seed10999+1009*index. Valid raw 100k cached physical clouds may be reused after source/signature checks and exact simulation replay at sigma10.',
            'training': 'None. MF1000 / DP1000 / MF500+DP500, fixed existing checkpoints. DP d+1, alpha .1 direct and 10 refinement.',
            'uncertainty': 'Three model seeds; report mean, sample SD and min/max conditional on fixed physical references. Parameter levels are not independent training replicates.',
            'physical_reference_caveat': '4D increases physical interaction ensemble from 10k to100k as well as evaluation size; comparison to old results is not a pure estimator-only ablation.',
            'protected_sha256': {str(p.relative_to(ROOT)): sha(p) for p in protected}, 'checkpoint_sha256': inputs,
            'runtime': {'python': sys.version, 'torch': str(torch.__version__), 'POT': ot.__version__, 'gpu': torch.cuda.get_device_name(), 'numpy': np.__version__}}
    if (OUT/'plan.json').exists():
        old = js(OUT/'plan.json')
        assert old['checkpoint_sha256'] == inputs and old['protected_sha256'] == plan['protected_sha256']
    else:
        save(OUT/'plan.json', plan)
    rng = np.random.default_rng(20261022)
    checks = []
    for dimension in (3, 4):
        x, y = rng.normal(size=(127, dimension)), rng.normal(size=(139, dimension))
        a, b = rng.random(127), rng.random(139)
        a, b = a/a.sum(), b/b.sum()
        lazy = solve((x, a), (y, b))
        dense, log = ot.lp.emd2_lazy(x, y, a, b, metric='sqeuclidean', log=True, numItermax=5000000)
        assert log['result_code'] == 1 and abs(lazy['w2_squared']-dense) < 1e-11
        checks.append({'dimension': dimension, 'dense_lazy_cost_difference': abs(lazy['w2_squared']-dense), **lazy})
    for name in ('ks3d', 'ks4d'):
        p, models = protocol(name), load_models(name)
        if name == 'ks4d':
            source = unit_ball(10000,4,np.random.default_rng(999)).astype(np.float32)
        else:
            with np.load(ROOT/'output/particle_count100k_budget_comparison_20261009/particles/sigma10_inputs.npz') as z:
                source=z['source'].copy()
        fresh = predict(source[:10000],10,p,models)
        with np.load(BASE/name/'evaluation/particles_00.npz') as base, np.load(COMPACT/name/'evaluation/particles_00.npz') as comp:
            for seed in SEEDS:
                for method in METHODS:
                    k=f'seed{seed}_{method}'
                    expected=(base if method=='mf' else comp)[k][:10000]
                    maximum=float(np.max(np.abs(fresh[k]-expected)))
                    assert maximum < 1e-5, (name,k,maximum)
                    checks.append({'model':name,'method':method,'seed':seed,'prediction_maximum_error':maximum})
        del models
        torch.cuda.empty_cache()
    save(OUT/'preflight.json', {'status':'passed','checks':checks})
    event('preflight', status='passed', checks=len(checks))


def write_case(name, level, clouds, metadata):
    raise RuntimeError('This run reuses the prepared 100k clouds read-only; preparation is disabled.')
    path=CLOUDS_ROOT/name/f'sigma{level}.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    for points in clouds.values():
        assert points.shape==(N,int(name[-2])) and np.isfinite(points).all()
    np.savez_compressed(path, meta=np.asarray(json.dumps(metadata)), **clouds)
    save(path.with_suffix('.json'), {'file':str(path.relative_to(OUT)), 'sha256':sha(path), **metadata})
    return path


def prepare():
    setup()
    assert js(OUT/'preflight.json')['status']=='passed'
    for level in LEVELS:
        path=OUT/'clouds/ks3d'/f'sigma{level}.npz'
        if path.exists():
            assert sha(path)==js(path.with_suffix('.json'))['sha256']
            continue
        i=level//10-1
        basepath=BASE/'ks3d/evaluation'/f'particles_{i:02d}.npz'
        comppath=COMPACT/'ks3d/evaluation'/f'particles_{i:02d}.npz'
        with np.load(basepath) as base, np.load(comppath) as comp:
            np.testing.assert_array_equal(base['reference'],comp['reference'])
            clouds={'reference':base['reference'].copy()}
            clouds.update({f'seed{s}_mf':base[f'seed{s}_mf'].copy() for s in SEEDS})
            clouds.update({f'seed{s}_{m}':comp[f'seed{s}_{m}'].copy() for s in SEEDS for m in ('dp','mf_dp')})
        write_case('ks3d',level,clouds,{'model':'ks3d','level':level,'particles':N,'dimension':3,'origin':'current frozen 100k manuscript clouds','inputs':{str(p.relative_to(ROOT)):sha(p) for p in (basepath,comppath)}})
        event('prepare',model='ks3d',level=level,completed=i+1,total=40)
    p=protocol('ks4d')
    models=load_models('ks4d')
    source=unit_ball(N,4,np.random.default_rng(10999)).astype(np.float32)
    for i,level in enumerate(LEVELS):
        path=OUT/'clouds/ks4d'/f'sigma{level}.npz'
        if path.exists():
            assert sha(path)==js(path.with_suffix('.json'))['sha256']
            continue
        noise_seed=10999+1009*i
        cached=CACHE/'references/ks4d'/f'level_{level}_seed{noise_seed}.npz'
        event('prepare',model='ks4d',level=level,completed=20+i,total=40,phase='physical reference',cached=cached.exists())
        t=time.perf_counter()
        replay_error=None
        if cached.exists():
            with np.load(cached) as z:
                meta=json.loads(str(z['meta']))
                np.testing.assert_array_equal(source,z['prior'])
                assert meta['protocol']==p.metadata() and meta['level']==level and meta['seed']==noise_seed
                reference=z['target'].copy()
            if i==0:
                replay=ks_simulate(source,level,p,seed=noise_seed,chunk=256,device='cuda')['target']
                replay_error=float(np.max(np.abs(replay-reference)))
                assert replay_error==0., replay_error
        else:
            reference=ks_simulate(source,level,p,seed=noise_seed,chunk=256,device='cuda')['target']
        clouds={'reference':reference,**predict(source,level,p,models)}
        metadata={'model':'ks4d','level':level,'particles':N,'dimension':4,'initial_seed':10999,'brownian_seed':noise_seed,
                  'origin':'100k interacting-particle physical simulation','physical_protocol':p.metadata(),
                  'cached_reference':str(cached.relative_to(ROOT)) if cached.exists() else None,
                  'cached_reference_sha256':sha(cached) if cached.exists() else None,
                  'cached_replay_maximum_error':replay_error,'seconds':time.perf_counter()-t}
        write_case('ks4d',level,clouds,metadata)
        event('prepare',model='ks4d',level=level,completed=21+i,total=40,phase='case complete',seconds=metadata['seconds'])
    save(OUT/'preparation_complete.json',{'status':'complete','cases':40,'particles':N})
    event('prepare',status='complete',completed=40,total=40)


def evaluate_case(name,level,workers):
    path=CLOUDS_ROOT/name/f'sigma{level}.npz'
    folder=OUT/'results'/name/f'sigma{level}'
    if (folder/'complete.json').exists():
        return
    while not path.with_suffix('.json').exists():
        if time.perf_counter()-STARTED>10800:
            raise TimeoutError('Evaluation preparation exceeded 3 hours')
        time.sleep(5)
    assert sha(path)==js(path.with_suffix('.json'))['sha256']
    with np.load(path) as z:
        clouds={k:z[k].copy() for k in z.files if k!='meta'}
    assert len(clouds)==10
    assert all(v.shape==(N,int(name[-2])) and np.isfinite(v).all() for v in clouds.values())
    measures,origin,widths=measures_for(clouds)
    save(folder/'grid.json',{'origin':origin.tolist(),'widths':widths.tolist(),'base_bins_per_axis':BINS,'dimension':int(name[-2]),'reference_particles':N,
                           'cloud_sha256':sha(path),'occupied_bins':{k:len(v[0]) for k,v in measures.items()}})
    del clouds
    def one(seed,method):
        target=folder/f'seed{seed}_{method}.json'
        if target.exists():
            return js(target)
        event('solve',model=name,level=level,seed=seed,method=method,phase='started')
        if name=='ks4d':
            from exact_sparse import solve as sparse_solve
            score=sparse_solve(measures['reference'],measures[f'seed{seed}_{method}'],grid=(origin,widths))
        else:
            score=solve(measures['reference'],measures[f'seed{seed}_{method}'])
        result={'model':name,'level':level,'seed':seed,'method':method,'particles':N,'dimension':int(name[-2]),'bins_per_axis':BINS,
                'metric':'full_dimensional_exact_histogram_W2',**score}
        save(target,result)
        print(f"SOLVED {name} sigma={level} seed={seed} {method} W2={result['w2']:.8f} seconds={result['seconds']:.1f}",flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,s,m) for s in SEEDS for m in METHODS]
        rows=[f.result() for f in as_completed(futures)]
    save(folder/'complete.json',{'status':'complete','rows':rows,'particles':N,'all_mass_retained':True})
    event('evaluate',model=name,level=level,cases_completed=sum(1 for _ in (OUT/'results').glob('*/*/complete.json')),total=40)


def evaluate(workers,names=('ks4d','ks3d'),levels=None):
    assert js(OUT/'preflight.json')['status']=='passed'
    # Prioritize the larger benchmark; 3D cases are ready while 4D is simulated.
    for name in names:
        for level in (LEVELS if levels is None else levels):
            evaluate_case(name,level,workers)
    count=sum(1 for _ in (OUT/'results').glob('*/*/complete.json'))
    if count==40:
        save(OUT/'evaluation_complete.json',{'status':'complete','cases':40,'rows':360})
    event('evaluate',status='complete' if count==40 else 'subset complete',cases_completed=count,total=40)


def table(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def report():
    plan=js(OUT/'plan.json')
    for rel,h in {**plan['protected_sha256'],**plan['checkpoint_sha256']}.items():
        assert sha(ROOT/rel)==h, f'Protected input changed: {rel}'
    rows=[]
    for name in ('ks3d','ks4d'):
        for level in LEVELS:
            data=js(OUT/'results'/name/f'sigma{level}'/'complete.json')
            assert data['status']=='complete' and len(data['rows'])==9
            rows.extend(data['rows'])
    assert len(rows)==360
    assert len({(r['model'],r['level'],r['seed'],r['method']) for r in rows})==360
    assert all(r['bins_per_axis']==BINS and r['particles']==N and r['dimension']==int(r['model'][-2]) for r in rows)
    assert all(r['solver_result_code']==1 and r['marginal_residual']<1e-9 and r['primal_dual_gap']<1e-8 for r in rows)
    table(OUT/'metrics_per_seed.csv',rows)
    summary=[]
    for name in ('ks3d','ks4d'):
        for level in LEVELS:
            for method in METHODS:
                values=[r['w2'] for r in rows if r['model']==name and r['level']==level and r['method']==method]
                summary.append({'model':name,'sigma':level,'method':method,'mean_w2':statistics.mean(values),'sd_w2':statistics.stdev(values),'min_w2':min(values),'max_w2':max(values),'training_seeds':3,'particles':N,'bins_per_axis':BINS})
    table(OUT/'summary.csv',summary)
    save(OUT/'summary.json',summary)
    oldrows=[]
    for name in ('ks3d','ks4d'):
        for source,methods in ((BASE,('mf',)),(COMPACT,('dp','mf_dp'))):
            with (source/name/'evaluation/metrics_per_seed.csv').open(encoding='utf-8-sig',newline='') as f:
                for r in csv.DictReader(f):
                    if r['method'] in methods:
                        oldrows.append({'model':name,'level':float(r['level']),'seed':int(r['seed']),'method':r['method'],'w2':float(r['w2']),'metric':r['metric']})
    comparisons=[]
    for r in summary:
        values=[x['w2'] for x in oldrows if x['model']==r['model'] and x['level']==r['sigma'] and x['method']==r['method']]
        assert len(values)==3,(r,len(values))
        old=statistics.mean(values)
        comparisons.append({**r,'old_mean_w2':old,'change_percent':100*(r['mean_w2']/old-1)})
    table(OUT/'old_vs_new.csv',comparisons)
    regional=[]
    for name in ('ks3d','ks4d'):
        for region,levels in [('all',LEVELS),('interpolation',[x for x in LEVELS if 15<=x<=150]),('high_extrapolation',[x for x in LEVELS if x>150]),('low_extrapolation',[10])]:
            for method in METHODS:
                seedmeans=[statistics.mean([r['w2'] for r in rows if r['model']==name and r['seed']==s and r['method']==method and r['level'] in levels]) for s in SEEDS]
                regional.append({'model':name,'region':region,'method':method,'mean_w2':statistics.mean(seedmeans),'sd_across_seed_means':statistics.stdev(seedmeans),'levels':len(levels),'training_seeds':3})
    table(OUT/'regional_summary.csv',regional)
    validate={'status':'passed','rows':360,'cases':40,'all_dimensions_included':True,'all_particles_retained':100000,'protected_inputs_unchanged':True,
              'max_marginal_residual':max(r['marginal_residual'] for r in rows),'max_primal_dual_gap':max(r['primal_dual_gap'] for r in rows),
              'optimal_solver_status_count':sum(r['solver_result_code']==1 for r in rows),'statistical_fallacy_scan':'11/11 considered: avoid pooling parameter points as independent replicates, no significance claims, no conclusions about dimensional difficulty solely from absolute errors; full fixed-grid sweep retained.',
              'physical_reference_limit':'4D uses a new larger physical reference. Single physical seed at each amplitude; bands across training seeds are not confidence intervals.'}
    save(OUT/'verification.json',validate)
    event('report',status='complete',**{k:v for k,v in validate.items() if k in ('rows','cases','max_primal_dual_gap')})


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('preflight','prepare','evaluate','report'))
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--models',nargs='+',choices=('ks3d','ks4d'),default=['ks4d','ks3d'])
    ap.add_argument('--levels',nargs='+',type=int,choices=LEVELS)
    args=ap.parse_args()
    if args.stage=='preflight': preflight()
    elif args.stage=='prepare': prepare()
    elif args.stage=='evaluate': evaluate(args.workers,args.models,args.levels)
    else: report()
