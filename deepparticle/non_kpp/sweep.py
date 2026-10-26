"""Train D-conditioned non-KPP models and plot physical diffusion versus W2.

This is a new conditional experiment, not a connection of separately fitted
single-D models. Held-out D values never enter the training case list.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import statistics
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from particle_pipeline import engine
from particle_pipeline.models import MeanFlow, generate
from particle_pipeline.protocol import fingerprint, file_hash
from . import experiment as fixed
from .models import CompactCorrector
from .physics import Physics, write_json


TRAIN_D = [.25, .5, 1., 2.]
EVAL_D = [.25, .375, .5, .75, 1., 1.5, 2.]
METHODS = ('mf', 'dp', 'mf_dp', 'mf_init')


def dkey(d):
    return 'D_' + format(d, '.6g').replace('.', 'p')


class DiffusionProtocol:
    dimension, periodic, family = 3, False, 'non_kpp'

    def __init__(self, reaction, particles):
        self.physics = Physics(reaction, particles=particles)
        self.name = reaction + '3d_diffusion'

    def condition(self, level):
        if not math.isfinite(level) or level <= 0:
            raise ValueError('Positive diffusion required')
        return (math.log2(level) + 2.) / 3.

    def metadata(self):
        physical = asdict(self.physics)
        physical.pop('diffusion')
        return {'schema': 'non_kpp_diffusion_conditional_v1', 'physics_except_D': physical,
                'dimension': 3, 'periodic': False, 'diffusion': 'level',
                'condition': '(log2(D)+2)/3', 'train_D': TRAIN_D,
                'coordinate_scale': self.physics.initial_radius,
                'target': 'normalized SGIP final density at the requested D',
                'prior': 'uniform unit 3-ball in scaled coordinates'}


def factory(dimension, periodic, width, blocks, alpha):
    assert dimension == 3 and not periodic
    model = CompactCorrector(dimension, width=width, alpha=alpha)
    model.architecture.pop('constant_condition')
    model.architecture.update(condition='(log2(D)+2)/3', conditioning='protocol.condition(level)')
    return model


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def get_data(root, reaction, cli):
    cases, records = [], []
    for d in EVAL_D:
        destination = root/'datasets'/reaction/dkey(d)
        physics = Physics(reaction, diffusion=d, particles=cli.particles)
        old = cli.existing/'datasets'/reaction
        if d == 1. and not destination.exists() and (old/'simulation.json').exists():
            if read_json(old/'simulation.json')['physics'] == asdict(physics):
                destination.mkdir(parents=True)
                for name in ('field.npz', 'simulation.json'):
                    shutil.copy2(old/name, destination/name)
        write_json(root/'progress.json', {'reaction':reaction, 'stage':'data', 'D':d})
        samples, signature = fixed.prepare(physics, destination, cli.n_train, cli.n_eval, cli.device)
        sim = read_json(destination/'simulation.json')
        assert all(r['discarded'] == 0 for r in sim['records']), 'Boundary truncation requires review'
        if d in TRAIN_D:
            cases.append({'level':d, 'prior':samples['train_prior']/physics.initial_radius,
                          'target':samples['train_target']/physics.initial_radius})
        records.append({'D':d, 'role':'training' if d in TRAIN_D else 'interpolation',
                        'sample_signature':signature, 'field_sha256':file_hash(destination/'field.npz'),
                        'total_mass':sim['total_mass']})
    signature = fingerprint({'protocol':DiffusionProtocol(reaction,cli.particles).metadata(),
                              'training_data':[r for r in records if r['D'] in TRAIN_D]})
    write_json(root/'datasets'/reaction/'manifest.json', {'training_signature':signature, 'cases':records})
    return cases, signature


def train(root, protocol, cases, signature, args, seeds):
    folder=root/'runs'/protocol.physics.reaction
    folder.mkdir(parents=True,exist_ok=True)
    config={'protocol':protocol.metadata(),'dataset_signature':signature,'training':vars(args),
            'seeds':seeds, 'train_D':TRAIN_D, 'eval_D':EVAL_D, 'status':'training',
            'sampling':'MF samples one D per update; DP optimizes the summed batches over all four training Ds',
            'budget_caveat':'Matched optimizer updates, not matched FLOPs or number of examples'}
    if (folder/'config.json').exists():
        old=read_json(folder/'config.json')
        assert {k:v for k,v in old.items() if k!='status'}=={k:v for k,v in config.items() if k!='status'}
    write_json(folder/'config.json',config)
    for seed in seeds:
        current=folder/f'seed{seed}'
        current.mkdir(exist_ok=True)
        if (current/'complete.json').exists():
            info=read_json(current/'complete.json')
            assert all(file_hash(current/f'{m}.pt')==info['sha256'][m] for m in METHODS)
            continue
        write_json(root/'progress.json', {'reaction':protocol.physics.reaction,'stage':'training','seed':seed})
        timings={}
        _, initial_state, timings['mf']=engine.train_mf(cases,protocol,args,current,signature,seed)
        mf_init=MeanFlow(3,args.width).to(args.device)
        mf_init.load_state_dict(initial_state)
        mf_init.eval().requires_grad_(False)
        prefix_before={k:v.clone() for k,v in mf_init.state_dict().items()}
        for method in ('dp','mf_dp'):
            _,timings[method]=engine.train_dp(cases,protocol,args,current,signature,seed,
                mf=mf_init if method=='mf_dp' else None,corrector_factory=factory)
        assert all(torch.equal(prefix_before[k],v) for k,v in mf_init.state_dict().items())
        write_json(current/'timings.json',timings)
        write_json(current/'complete.json',{'sha256':{m:file_hash(current/f'{m}.pt') for m in METHODS},
                                           'frozen_MF_unchanged':True})
    config['status']='complete'
    write_json(folder/'config.json',config)


@torch.no_grad()
def infer(method, physical_prior, d, protocol, models, device):
    x=torch.from_numpy(physical_prior/protocol.physics.initial_radius).to(device)
    condition=torch.full((len(x),1),protocol.condition(d),device=device)
    output=generate('mf' if method=='mf_init' else method,x,condition,protocol,
        mf=models['mf'] if method=='mf' else models['mf_init'],
        dp=models['mf_dp'] if method=='mf_dp' else models['dp'])
    result=(output*protocol.physics.initial_radius).cpu().numpy()
    assert np.isfinite(result).all()
    return result


def evaluate(root,protocol,signature,args,cli):
    from .histogram_ot import Grid, histogram, score
    reaction=protocol.physics.reaction
    output=root/'evaluation'/reaction
    output.mkdir(parents=True,exist_ok=True)
    folder=root/'runs'/reaction
    models_by_seed={s:fixed.read_models(folder,s,args,signature) for s in cli.seeds}
    source_indices=[np.random.default_rng(7000+i).choice(cli.n_eval,cli.metric_particles,replace=False)
                    for i in range(cli.metric_repeats)]
    target_indices=[np.random.default_rng(8000+i).choice(cli.n_eval,cli.metric_particles,replace=False)
                    for i in range(cli.metric_repeats)]
    rows, floors, validation=[],[],[]
    for d in EVAL_D:
        write_json(root/'progress.json',{'reaction':reaction,'stage':'evaluation','D':d})
        destination=output/dkey(d)
        destination.mkdir(exist_ok=True)
        with np.load(root/'datasets'/reaction/dkey(d)/'samples.npz') as data:
            sources=[data['eval_prior'][idx] for idx in source_indices]
            targets=[data['eval_target'][idx] for idx in target_indices]
            independent=[data['eval_target_independent'][idx] for idx in source_indices]
        grid=Grid.from_reference(targets[0])
        write_json(destination/'evaluation_grid.json',grid.metadata())
        reference_measures=[histogram(t,grid) for t in targets]
        floor=[score(t,histogram(r,grid))['w2_squared'] for t,r in zip(reference_measures,independent)]
        floors.append({'reaction':reaction,'D':d,'w2':statistics.mean(math.sqrt(v) for v in floor),
                       'w2_squared':statistics.mean(floor)})
        for seed in cli.seeds:
            archive={'prior':np.stack(sources),'target':np.stack(targets)}
            for method in METHODS:
                predictions=[infer(method,s,d,protocol,models_by_seed[seed],args.device) for s in sources]
                values=[score(t,histogram(g,grid))['w2_squared'] for t,g in zip(reference_measures,predictions)]
                row={'reaction':reaction,'D':d,'role':'training' if d in TRAIN_D else 'interpolation',
                     'seed':seed,'method':method,'w2':statistics.mean(math.sqrt(v) for v in values),
                     'w2_squared':statistics.mean(values),'metric_particles':cli.metric_particles,
                     'metric_repeats':cli.metric_repeats,'metric':'full_coordinate_ks_histogram','bins_per_axis':24}
                row.update({f'w2_squared_repeat{i+1}':v for i,v in enumerate(values)})
                rows.append(row)
                archive[method]=np.stack(predictions)
            np.savez_compressed(destination/f'seed{seed}_metric_clouds.npz',**archive)
        engine.csv_write(output/'metrics_per_seed.csv',rows)
        engine.csv_write(output/'reference_sampling.csv',floors)
        print(f'{reaction} D={d:g} evaluation complete',flush=True)
    # Reload independently and reproduce saved predictions at a held-out D.
    checked_d=.75
    for seed in cli.seeds:
        models=fixed.read_models(folder,seed,args,signature)
        with np.load(output/dkey(checked_d)/f'seed{seed}_metric_clouds.npz') as archived:
            for method in METHODS:
                reproduced=infer(method,archived['prior'][0],checked_d,protocol,models,args.device)
                np.testing.assert_allclose(reproduced,archived[method][0],atol=2e-5,rtol=1e-6)
                payload=torch.load(folder/f'seed{seed}'/f'{method}.pt',map_location='cpu',weights_only=True)
                assert payload['protocol']==protocol.metadata()
                expected=args.mf_steps if method in ('mf_init','mf_dp') else args.mf_steps+args.dp_steps
                assert payload['step']==expected
                validation.append({'reaction':reaction,'seed':seed,'method':method,'step':expected,
                    'reload_max_error':float(abs(reproduced-archived[method][0]).max())})
    summary=[]
    for d in EVAL_D:
        for method in METHODS:
            subset=[r for r in rows if r['D']==d and r['method']==method]
            result={'reaction':reaction,'D':d,'role':'training' if d in TRAIN_D else 'interpolation',
                    'method':method,'seeds':len(cli.seeds)}
            for metric in ('w2','w2_squared'):
                values=[r[metric] for r in subset]
                result[metric+'_mean']=statistics.mean(values)
                result[metric+'_sd']=statistics.stdev(values) if len(values)>1 else 0.
            summary.append(result)
    write_json(output/'verification.json',{'status':'passed','checkpoints':validation,
                'train_D':TRAIN_D,'held_out_D':[d for d in EVAL_D if d not in TRAIN_D],
                'same_model_across_all_D':True,'physical_coordinates':True})
    engine.csv_write(output/'summary.csv',summary)
    return summary,floors


def plot(root,summary,floors,cli):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'mf':'#5470C6','dp':'#D78A32','mf_dp':'#268A72'}
    labels={'mf':'MF','dp':'DP','mf_dp':'MF + DP refinement'}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.2,5.4))
    fig.subplots_adjust(left=.065,right=.985,bottom=.18,top=.77,wspace=.22)
    for ax,reaction in zip(axes,('cubic','arrhenius')):
        for method in ('mf','dp','mf_dp'):
            group=sorted([r for r in summary if r['reaction']==reaction and r['method']==method],key=lambda r:r['D'])
            x=np.array([r['D'] for r in group]); y=np.array([r['w2_mean'] for r in group])
            sd=np.array([r['w2_sd'] for r in group])
            ax.plot(x,y,'o-',color=colors[method],label=labels[method],lw=1.8,ms=4.5)
            ax.fill_between(x,y-sd,y+sd,color=colors[method],alpha=.13,linewidth=0)
        reference=[r for r in floors if r['reaction']==reaction]
        ax.plot([r['D'] for r in reference],[r['w2'] for r in reference],ls='--',color='#777777',lw=1.3,label='Reference vs reference')
        ax.plot(TRAIN_D,[.02]*len(TRAIN_D),marker='|',ms=10,ls='none',color='#333333',
                transform=ax.get_xaxis_transform(),clip_on=True,label='Training D')
        ax.set_title(reaction.capitalize())
        ax.set_xlabel('Diffusion coefficient D',labelpad=15)
        ax.set_ylabel(r'$W_2$ (physical coordinates)')
        ax.set_xticks([.25,.5,.75,1.,1.5,2.])
        ax.grid(axis='y',alpha=.2)
        ax.set_xlim(.2,2.05)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.9),ncol=5,frameon=False,fontsize=10)
    fig.suptitle(f'Non-KPP: conditional models | {cli.steps:,} total updates | mean +/- SD over {len(cli.seeds)} seeds',fontsize=13,y=.98)
    fig.savefig(root/'diffusion_w2.png',dpi=200)
    fig.savefig(root/'diffusion_w2.pdf')
    plt.close(fig)
    lines=['Non-KPP conditional diffusion sweep','',
        'Horizontal axis: physical diffusion coefficient D in u_t = D Laplacian(u) + w dot grad(u) + r(u).',
        'Vertical axis: exact KS-style full-coordinate histogram W2, 24 initial bins/axis, in physical coordinates.',
        f'Train D: {TRAIN_D}. Evaluation D: {EVAL_D}. D=0.375,0.75,1.5 are held out interpolation points.',
        'One conditional model per method/reaction/seed is used for every D. This is a new run, not the old D=1 model.',
        'Conditioning: (log2(D)+2)/3. Coordinates divided by the radius 5 during learning, restored for evaluation.',
        f'Physical simulation: {cli.particles:,} particles, 100^3 bins, T=20, dt=0.5, initial radius 5, domain [-60,60]^3.',
        'Only D changes; drift, reaction, initial density, numerical settings and T are held fixed.',
        'Reference: seeded PyTorch SGIP port of supplied code. No FDM or convergence study was run.',
        f'Seeds: {cli.seeds}. MF and DP: {cli.steps} updates; hybrid: MF {cli.steps//2} + frozen-MF DP {cli.steps//2}.',
        'Budget is matched optimizer updates, not matched FLOPs/examples: MF samples one D each step; DP sums four D batches.',
        'MF/DP networks, learning rates, batch/OT refresh periods and penalties match the preceding experiment.',
        f'{cli.n_train:,} training samples per training D. Independent evaluation draws from each same fixed SGIP density.',
        f'W2 uses {cli.metric_particles:,} points and {cli.metric_repeats} fixed subsamples per model.',
        'Shading: sample SD over model seeds; does not include PDE reference uncertainty.',
        'Gray dashed line: independent reference-vs-reference sampling diagnostic; not a rigorous error lower bound.',
        'Short black ticks inside the bottom edge mark training D. Curves connecting test points are guides to the eye.',
        'No held-out D samples were used for training, hyperparameter tuning, or checkpoint selection.',
        'Metric source/target/generated point clouds and individual model checkpoints are saved.','',
        'Reproduce from deepparticle/:',
        '  python -m non_kpp.sweep --out ../output/non_kpp_diffusion_new','',
        'reaction,D,method,W2_mean,W2_SD']
    lines += [f'{r["reaction"]},{r["D"]},{r["method"]},{r["w2_mean"]:.8f},{r["w2_sd"]:.8f}' for r in summary]
    (root/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--existing',type=Path,default=Path('../output/non_kpp_20261020'))
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--seeds',nargs='+',type=int,default=[88,89,90])
    parser.add_argument('--particles',type=int,default=5_000_000)
    parser.add_argument('--n-train',type=int,default=20000)
    parser.add_argument('--n-eval',type=int,default=200000)
    parser.add_argument('--metric-particles',type=int,default=100000)
    parser.add_argument('--metric-repeats',type=int,default=1)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    cli=parser.parse_args()
    assert cli.steps>=2 and cli.steps%2==0 and 0<cli.metric_particles<=cli.n_eval
    assert len(set(cli.seeds))==len(cli.seeds) and cli.metric_repeats>0
    root=cli.out.resolve(); root.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(mf_steps=cli.steps//2,dp_steps=cli.steps//2,mf_lr=.001,dp_lr=.0001,
        batch_size=2000,width=64,mf_width=64,mf_hidden_layers=3,dp_blocks=0,alpha=.1,hybrid_alpha=10.,
        profile='original_tuned',residual_weight=.1,batch_refresh=500,ot_refresh=500,device=cli.device)
    plan={'training':vars(args),'train_D':TRAIN_D,'eval_D':EVAL_D,'seeds':cli.seeds,
          'particles':cli.particles,'n_train_per_D':cli.n_train,'n_eval_per_D':cli.n_eval,
          'metric_particles':cli.metric_particles,'metric_repeats':cli.metric_repeats,
          'conditioning':'(log2(D)+2)/3','selection':'fixed final steps; no test-based tuning'}
    if (root/'plan.json').exists():
        assert read_json(root/'plan.json')==plan,'Existing experiment settings differ'
    write_json(root/'plan.json',plan)
    torch.set_num_threads(4)
    prototype=DiffusionProtocol('cubic',cli.particles)
    np.testing.assert_allclose([prototype.condition(d) for d in TRAIN_D],[0,1/3,2/3,1])
    assert [d for d in EVAL_D if d not in TRAIN_D]==[.375,.75,1.5]
    source_root=Path(__file__).resolve().parents[1]
    source_files=list((source_root/'non_kpp').rglob('*.py'))+list((source_root/'particle_pipeline').glob('*.py'))+list((source_root/'training').glob('*.py'))
    write_json(root/'source_manifest.json',{str(p.relative_to(source_root)):file_hash(p) for p in source_files})
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path,str(path.relative_to(source_root)))
    summary,floors=[],[]
    for reaction in ('cubic','arrhenius'):
        cases,signature=get_data(root,reaction,cli)
        assert [c['level'] for c in cases]==TRAIN_D
        protocol=DiffusionProtocol(reaction,cli.particles)
        train(root,protocol,cases,signature,args,cli.seeds)
        these,refs=evaluate(root,protocol,signature,args,cli)
        summary+=these; floors+=refs
        engine.csv_write(root/'summary.csv',summary)
        engine.csv_write(root/'reference_sampling.csv',floors)
    plot(root,summary,floors,cli)
    write_json(root/'progress.json',{'stage':'complete'})
    print(str(root/'diffusion_w2.png'),flush=True)


if __name__=='__main__':
    main()
