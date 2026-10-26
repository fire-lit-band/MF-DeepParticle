"""Standalone 100k / 16-bin evaluation. No manuscript or training changes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
_DLL_DIRECTORY = Path(sys.prefix)/'Library/bin'
DLL = os.add_dll_directory(str(_DLL_DIRECTORY)) if os.name == 'nt' and _DLL_DIRECTORY.is_dir() else None
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0,str(ROOT/'deepparticle'))
sys.path.insert(0,str(ROOT/'output/dp_dplus1_four_benchmarks_20261016'))
import numpy as np
import ot
import torch
from scipy.spatial.distance import cdist
from compact_architecture import CompactCorrector
from particle_pipeline.models import MeanFlow, generate
from particle_pipeline.protocol import Protocol, file_hash
from particle_pipeline.data import prior, read_npz, write_npz, write_json
from non_kpp.experiment import read_models as read_non_kpp
from non_kpp.sweep import DiffusionProtocol, dkey, infer as non_kpp_infer
from paper_reproduction.portability import resolve_recorded_path

N, BINS = 100000, 16
SEEDS, METHODS = (88,89,90), ('mf','dp','mf_dp')
KPP_LEVELS = [2+.25*i for i in range(11)]
NON_LEVELS = [.25,.375,.5,.75,1.,1.5,2.]
BASE = ROOT/'output/mf_matched_dp_20261016'
SELECTED = ROOT/'output/dp_alpha01_refinement_alpha10_20261017'
NON = ROOT/'output/non_kpp_diffusion_sweep_20261020'

def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def reference_path(name, record):
    """Resolve a frozen reference record inside this checkout's model folder."""
    if name not in ('kpp2d', 'kpp3d'):
        raise ValueError(f'Unknown KPP reference model: {name}')
    return resolve_recorded_path(record['file'], root=ROOT,
                                 allowed_root=OUT/'references'/name)

def csv_write(path, rows):
    path=Path(path)
    if not rows:
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

def setup():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

def make_plan():
    paths=[ROOT/'mainV2.tex',ROOT/'mainV2.pdf',ROOT/'sections/experiment_protocols.tex',
           ROOT/'sections/non_kpp_cubic.tex']
    checkpoints={}
    manifest=js(SELECTED/'checkpoint_manifest.json')
    for name in ('kpp2d','kpp3d'):
        for seed in SEEDS:
            for method in (*METHODS,'mf_init'):
                item=manifest[name][str(seed)][method]
                assert file_hash(ROOT/item['file'])==item['sha256']
                checkpoints[item['file']]=item['sha256']
    for reaction in ('cubic','arrhenius'):
        for seed in SEEDS:
            expected=js(NON/'runs'/reaction/f'seed{seed}/complete.json')['sha256']
            for method in (*METHODS,'mf_init'):
                p=NON/'runs'/reaction/f'seed{seed}'/f'{method}.pt'
                assert file_hash(p)==expected[method]
                checkpoints[str(p.relative_to(ROOT))]=expected[method]
    plan={'material_passport':{'origin_skill':'academic-research-suite','mode':'run + validate',
                              'date':'2026-10-23','status':'PLANNED'},
          'scope':['kpp2d','kpp3d','non_kpp_cubic','non_kpp_arrhenius'],
          'excluded':'cancer', 'destination':'independent results only; manuscript unchanged',
          'particles_per_reference_and_generated_cloud':N,'bins_per_axis':BINS,
          'kpp_levels':KPP_LEVELS,'non_kpp_diffusion_levels':NON_LEVELS,
          'methods':METHODS,'training_seeds':SEEDS,'new_training':False,
          'KPP_grid':'fixed 16^d torus grid on [0,2pi)^d; periodic squared distance in every coordinate',
          'non_KPP_grid':'full 100k reference min/max, 16 bins per coordinate range, same lattice extended without clipping',
          'non_KPP_normalization':'physical W2 retained; normalized W2 = physical W2 / 5 for comparison with current appendix',
          'KPP_source':'100k fresh Normal(0,pi) particles, seed411002, shared by levels and methods; direct DP input wrapped as in current paper',
          'KPP_reference':'100k interacting FK particles to T4096, original dt2^-8, phase0, physical seed10999, unchanged numerical convention',
          'non_KPP_reference':'same fixed 5-million-particle numerical fields and same 100k evaluation draws as previous 24-bin evaluation',
          'solver':'unregularized exact weighted quadratic OT, primal-dual and marginal checks; square root reported',
          'uncertainty':'mean, sample SD, min/max over 3 model seeds, conditional on fixed reference; no formal significance claim',
          'comparison_limit':'KPP physical ensemble and metric both change; old/new differences cannot isolate the metric alone',
          'protected_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in paths},
          'checkpoint_sha256':checkpoints}
    path=OUT/'plan.json'
    if path.exists():
        old=js(path)
        assert old['checkpoint_sha256']==plan['checkpoint_sha256']
        assert old['protected_sha256']==plan['protected_sha256']
    else:
        write_json(path,plan)
    return plan

def kpp_models(name):
    manifest=js(SELECTED/'checkpoint_manifest.json')[name]
    config=js(BASE/name/'run/config.json')
    p=Protocol(**{k:v for k,v in config['protocol'].items() if k in Protocol.__dataclass_fields__})
    models={}
    for seed in SEEDS:
        models[seed]={}
        for method in (*METHODS,'mf_init'):
            record=manifest[str(seed)][method]
            path=resolve_recorded_path(record['file'], root=ROOT)
            assert file_hash(path)==record['sha256']
            payload=torch.load(path,map_location='cpu',weights_only=True)
            assert payload['protocol']==config['protocol']
            assert payload['dataset_signature']==config['dataset_signature']
            assert payload['step']==(500 if method in ('mf_init','mf_dp') else 1000)
            if method in ('mf','mf_init'):
                model=MeanFlow(p.dimension,64,'protocol',3)
            else:
                alpha=.1 if method=='dp' else 10.
                assert payload['architecture']['alpha']==alpha
                model=CompactCorrector(p.dimension,alpha=alpha,condition_mode='protocol')
            model.load_state_dict(payload['model'],strict=True)
            models[seed][method]=model.cuda().eval().requires_grad_(False)
    return p, models

@torch.no_grad()
def predict_kpp(source, level, p, models):
    outputs={}
    for seed in SEEDS:
        m=models[seed]
        for method in METHODS:
            chunks=[]
            for begin in range(0,len(source),10000):
                x=torch.from_numpy(source[begin:begin+10000]).cuda()
                if method=='dp':
                    x=torch.remainder(x,2*math.pi)
                cond=torch.full((len(x),1),p.condition(level),device='cuda')
                y=generate(method,x,cond,p,mf=m['mf'] if method=='mf' else m['mf_init'],
                           dp=m['dp'] if method=='dp' else m['mf_dp'])
                chunks.append(y.cpu().numpy())
            outputs[f'seed{seed}_{method}']=np.concatenate(chunks)
    return outputs

def prepare_kpp():
    make_plan(); checks=[]
    for name in ('kpp2d','kpp3d'):
        p,models=kpp_models(name)
        arrays,_=read_npz(ROOT/'output/kpp_original_training_total1k_20261014/datasets'/name/'clouds/eval_00.npz')
        reproduced=predict_kpp(arrays['prior'][:10000],2.,p,models)
        old,_=read_npz(BASE/name/'evaluation/particles_00.npz')
        corrected,_=read_npz(ROOT/'output/dp_dplus1_four_benchmarks_20261016/validated_periodic'/name/'evaluation/particles_00.npz')
        for key,points in reproduced.items():
            expected=(old if key.endswith('_mf') else corrected)[key][:10000]
            error=float(np.max(np.abs(points-expected)))
            assert error < 1e-5,(name,key,error)
            checks.append({'model':name,'key':key,'old_prediction_max_error':error})
        source=prior(p,N,411002)
        folder=OUT/'generated'/name
        folder.mkdir(parents=True,exist_ok=True)
        write_npz(folder/'source.npz',meta={'seed':411002,'distribution':'Normal(0,pi)','particles':N},source=source)
        for i,level in enumerate(KPP_LEVELS):
            path=folder/f'level_{i:02d}.npz'
            if path.exists():
                continue
            generated=predict_kpp(source,level,p,models)
            assert all(v.shape==(N,p.dimension) and np.isfinite(v).all() for v in generated.values())
            write_npz(path,meta={'model':name,'level':level,'particles':N,'source_seed':411002,
                                'checkpoints':'current selected manuscript checkpoints'},**generated)
            print(f'Generated {name} level={level:g}: 9 x 100000 particles.',flush=True)
        del models
        torch.cuda.empty_cache()
    write_json(OUT/'prediction_verification.json',{'status':'passed','checks':checks})

def nested_indices(pool, count, old_seed, extra_seed):
    first=np.random.default_rng(old_seed).choice(pool,2000,replace=False)
    remaining=np.setdiff1d(np.arange(pool),first)
    extra=np.random.default_rng(extra_seed).choice(remaining,count-2000,replace=False)
    return np.concatenate((first,extra))

def histogram(points, origin, width, periodic):
    points=np.asarray(points,dtype=np.float64)
    assert points.shape[0]==N and np.isfinite(points).all()
    if periodic:
        points=np.remainder(points,2*np.pi)
    index,counts=np.unique(np.floor((points-origin)/width).astype(np.int64),axis=0,return_counts=True)
    assert counts.sum()==N
    return np.ascontiguousarray(origin+(index+.5)*width),counts.astype(np.float64)/N

def solve(first,second,periodic):
    start=time.perf_counter()
    x,a=first; y,b=second
    if periodic:
        cost=np.zeros((len(x),len(y)),dtype=np.float64)
        for axis in range(x.shape[1]):
            delta=(x[:,None,axis]-y[None,:,axis]+np.pi)%(2*np.pi)-np.pi
            cost+=delta*delta
    else:
        cost=cdist(x,y,'sqeuclidean')
    g,info=ot.emd(a,b,cost,numItermax=20000000,log=True,numThreads=1)
    assert info['result_code']==1 and not info['warning'],info
    value=float(info['cost'])
    residual=max(float(np.max(np.abs(g.sum(1)-a))),float(np.max(np.abs(g.sum(0)-b))))
    raw_dual_violation=float(np.max(info['u'][:,None]+info['v'][None,:]-cost))
    # Repair floating-point potential infeasibility by the exact c-transform.
    # This leaves both the primal coupling and objective unchanged and yields
    # a valid lower-bound certificate for the full, unregularized OT problem.
    certified_u=np.minimum(info['u'],np.min(cost-info['v'][None,:],axis=1))
    dual=float(a@certified_u+b@info['v'])
    gap=abs(value-dual)
    dual_violation=float(np.max(certified_u[:,None]+info['v'][None,:]-cost))
    assert residual < 1e-9 and gap < 1e-8 and dual_violation < 1e-8, {
        'marginal_residual':residual,'primal_dual_gap':gap,'dual_violation':dual_violation,
        'primal_cost':value,'max_ground_cost':float(cost.max())}
    return {'w2':math.sqrt(max(0.,value)),'w2_squared':max(0.,value),
            'marginal_residual':residual,'primal_dual_gap':gap,'dual_violation':dual_violation,
            'raw_dual_violation':raw_dual_violation,
            'reference_occupied_bins':len(a),'generated_occupied_bins':len(b),
            'solver_result_code':int(info['result_code']),'seconds':time.perf_counter()-start}

def evaluate_clouds(name,level,clouds,periodic,scale,provenance):
    folder=OUT/'results'/name/('level_'+format(level,'.6g').replace('.','p'))
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'complete.json'
    if path.exists():
        return js(path)['rows']
    ref=clouds['reference'].astype(np.float64)
    d=ref.shape[1]
    assert all(v.shape==(N,d) and np.isfinite(v).all() for v in clouds.values())
    origin=np.zeros(d) if periodic else ref.min(0)
    width=np.full(d,2*np.pi/BINS) if periodic else np.ptp(ref,axis=0)/BINS
    measures={k:histogram(v,origin,width,periodic) for k,v in clouds.items()}
    write_json(folder/'inputs.json',{'model':name,'level':level,'N':N,'bins':BINS,'dimension':d,
               'periodic':periodic,'origin':origin.tolist(),'width':width.tolist(),'provenance':provenance})
    jobs=[(f'seed{seed}_{method}',seed,method,'target_error') for seed in SEEDS for method in METHODS]
    if 'independent_reference' in measures:
        jobs.append(('independent_reference',0,'reference','reference_self'))
    partial=folder/'partial.json'
    rows=js(partial) if partial.exists() else []
    done={(r['seed'],r['method'],r['kind']) for r in rows}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures={executor.submit(solve,measures['reference'],measures[key],periodic):(seed,method,kind)
                 for key,seed,method,kind in jobs if (seed,method,kind) not in done}
        for future in as_completed(futures):
            seed,method,kind=futures[future]
            result=future.result()
            row={'model':name,'level':level,'seed':seed,'method':method,'kind':kind,
                 'particles':N,'bins':BINS,'dimension':d,'periodic':periodic,'normalization_scale':scale,
                 **result,'w2_normalized':result['w2']/scale}
            rows.append(row)
            write_json(partial,rows)
    rows.sort(key=lambda r:(r['kind'],r['seed'],r['method']))
    write_json(path,{'status':'complete','rows':rows})
    means={m:statistics.mean(r['w2_normalized'] for r in rows if r['method']==m) for m in METHODS}
    print(json.dumps({'model':name,'level':level,'means':means}),flush=True)
    return rows

def evaluate_non_kpp():
    make_plan()
    plan=js(NON/'plan.json')
    args=SimpleNamespace(**plan['training'])
    si=nested_indices(plan['n_eval_per_D'],N,7000,700000)
    ti=nested_indices(plan['n_eval_per_D'],N,8000,800000)
    checks=[]
    for reaction in ('cubic','arrhenius'):
        name='non_kpp_'+reaction
        cfg=js(NON/'runs'/reaction/'config.json')
        models={s:read_non_kpp(NON/'runs'/reaction,s,args,cfg['dataset_signature']) for s in SEEDS}
        p=DiffusionProtocol(reaction,plan['particles'])
        for level in NON_LEVELS:
            path=NON/'datasets'/reaction/dkey(level)/'samples.npz'
            with np.load(path,allow_pickle=False) as data:
                source=data['eval_prior'][si]
                clouds={'reference':data['eval_target'][ti],
                        'independent_reference':data['eval_target_independent'][si]}
            for seed in SEEDS:
                for method in METHODS:
                    generated=np.concatenate([non_kpp_infer(method,source[i:i+10000],level,p,models[seed],'cuda')
                                              for i in range(0,N,10000)])
                    clouds[f'seed{seed}_{method}']=generated
                    if level==1.:
                        with np.load(NON/'evaluation'/reaction/dkey(level)/f'seed{seed}_metric_clouds.npz') as old:
                            np.testing.assert_array_equal(source[:2000],old['prior'][0])
                            np.testing.assert_array_equal(clouds['reference'][:2000],old['target'][0])
                            error=float(np.max(np.abs(generated[:2000]-old[method][0])))
                            assert error<5e-5,(reaction,seed,method,error)
                            checks.append({'reaction':reaction,'seed':seed,'method':method,'prediction_max_error':error})
            evaluate_clouds(name,level,clouds,False,5.,{'dataset':str(path.relative_to(ROOT)),
                'dataset_sha256':file_hash(path),'source_indices_sha256':hashlib.sha256(si.tobytes()).hexdigest(),
                'target_indices_sha256':hashlib.sha256(ti.tobytes()).hexdigest(),'budget':2000})
        del models
        torch.cuda.empty_cache()
    write_json(OUT/'non_kpp_prediction_verification.json',{'status':'passed','checks':checks})

def evaluate_kpp(name):
    make_plan()
    manifest=js(OUT/'references'/name/'manifest.json')
    assert manifest['status']=='complete'
    for i,level in enumerate(KPP_LEVELS):
        record=next(r for r in manifest['outputs'] if float(r['level'])==level)
        path=reference_path(name, record)
        assert file_hash(path)==record['sha256']
        ref,meta=read_npz(path)
        assert meta['particles']==N and meta['generations']==4096 and meta['selection_count']==4096
        clouds,_=read_npz(OUT/'generated'/name/f'level_{i:02d}.npz')
        clouds['reference']=ref['target']
        evaluate_clouds(name,level,clouds,True,1.,{'reference':str(path.relative_to(ROOT)),
            'reference_sha256':record['sha256'],'generated_sha256':file_hash(OUT/'generated'/name/f'level_{i:02d}.npz')})

def collect():
    plan=make_plan()
    rows=[]
    for path in sorted((OUT/'results').glob('*/*/complete.json')):
        rows.extend(js(path)['rows'])
    csv_write(OUT/'metrics_per_seed.csv',rows)
    summary=[]
    for key in sorted({(r['model'],r['level'],r['method'],r['kind']) for r in rows}):
        group=[r for r in rows if (r['model'],r['level'],r['method'],r['kind'])==key]
        values=[r['w2'] for r in group]; normalized=[r['w2_normalized'] for r in group]
        scale=group[0]['normalization_scale']
        summary.append(dict(zip(('model','level','method','kind'),key))|{
            'w2_mean':statistics.mean(values),'w2_sd':statistics.stdev(values) if len(values)>1 else 0.,
            'w2_min':min(values),'w2_max':max(values),'w2_normalized_mean':statistics.mean(normalized),
            'w2_normalized_sd':statistics.stdev(normalized) if len(values)>1 else 0.,
            'seeds':len(values),'particles':N,'bins':BINS,'normalization_scale':scale})
    csv_write(OUT/'summary.csv',summary)
    write_json(OUT/'summary.json',summary)
    counts={name:sum(r['kind']=='target_error' and r['model']==name for r in rows) for name in plan['scope']}
    expected={'kpp2d':99,'kpp3d':99,'non_kpp_cubic':63,'non_kpp_arrhenius':63}
    complete=counts==expected
    if complete:
        for name in expected:
            required=KPP_LEVELS if name.startswith('kpp') else NON_LEVELS
            actual={(r['level'],r['seed'],r['method']) for r in rows
                    if r['model']==name and r['kind']=='target_error'}
            assert actual=={(level,seed,method) for level in required for seed in SEEDS for method in METHODS}
        for filename in ('prediction_verification.json','non_kpp_prediction_verification.json',
                         'independent_metric_checks.json','references_complete.json'):
            assert js(OUT/filename)['status'] in ('passed','complete')
        assert js(OUT/'references/kpp2d/resume_slicing_validation.json')['status']=='passed'
        for name in ('kpp2d','kpp3d'):
            assert js(OUT/'references'/name/'validation.json')['status']=='passed'
            manifest=js(OUT/'references'/name/'manifest.json')
            assert manifest['status']=='complete' and len(manifest['outputs'])==11
    protected=all(file_hash(ROOT/p)==h for p,h in plan['protected_sha256'].items())
    checkpoints=all(file_hash(ROOT/p)==h for p,h in plan['checkpoint_sha256'].items())
    assert protected and checkpoints
    write_json(OUT/'verification.json',{'status':'passed' if complete else 'partial','counts':counts,
        'expected':expected,'protected_manuscript_files_unchanged':protected,'model_checkpoints_unchanged':checkpoints,
        'all_particles_retained':all(r['particles']==N for r in rows),
        'max_marginal_residual':max((r['marginal_residual'] for r in rows),default=0),
        'max_primal_dual_gap':max((r['primal_dual_gap'] for r in rows),default=0),
        'max_dual_violation':max((r['dual_violation'] for r in rows),default=0),
        'all_solver_results_optimal':all(r['solver_result_code']==1 for r in rows),
        'fallacy_scan':'11/11 considered: no pooling of levels as independent repeats; no significance or causal claims; fixed final checkpoints and all planned levels retained; reference and metric change together; other human-study categories inapplicable.'})
    print(json.dumps({'complete':complete,'counts':counts}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','non_kpp','kpp2d','kpp3d','collect'])
    cli=parser.parse_args()
    setup()
    if cli.command=='prepare': prepare_kpp()
    elif cli.command=='non_kpp': evaluate_non_kpp(); collect()
    elif cli.command in ('kpp2d','kpp3d'): evaluate_kpp(cli.command); collect()
    else: collect()
