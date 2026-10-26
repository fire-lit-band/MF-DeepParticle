"""Adopt verified 100k/16-bin KS scores and match the two-step MF ablation."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
EVAL=ROOT/'output/ks_full_histogram_100k_bins16_20261022'
FIG=ROOT/'figures/ks_histogram_100k_bins16_20261023'
sys.path.insert(0,str(EVAL))
import run_evaluation as r
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def prepare_ablation():
    r.setup()
    plan=r.js(EVAL/'plan.json')
    assert r.js(EVAL/'verification.json')['status']=='passed'
    for rel,digest in plan['checkpoint_sha256'].items():
        assert r.sha(ROOT/rel)==digest
    # Validate the source and weights by replaying the retained one-step outputs.
    source_root=ROOT/'output/particle_count100k_budget_comparison_20261009/particles'
    models=r.load_models('ks3d')
    rows=[]
    checks=[]
    sources=[]
    for level in r.LEVELS:
        source_path=source_root/f'sigma{level}_inputs.npz'
        with np.load(source_path) as z:
            source=z['source'].copy()
            source_reference=z['reference'].copy()
        assert source.shape==(100000,3)
        sources.append({'sigma':level,'path':str(source_path),'sha256':r.sha(source_path)})
        cloud_path=r.CLOUDS_ROOT/'ks3d'/f'sigma{level}.npz'
        assert r.sha(cloud_path)==r.js(cloud_path.with_suffix('.json'))['sha256']
        with np.load(cloud_path) as z:
            reference=z['reference'].copy()
            baseline={s:z[f'seed{s}_mf'].copy() for s in r.SEEDS}
        np.testing.assert_array_equal(reference,source_reference)
        clouds={'reference':reference}
        with torch.no_grad():
            for seed in r.SEEDS:
                model=models[seed]['mf']
                two=[]
                one=[]
                for start in range(0,100000,10000):
                    x=torch.from_numpy(source[start:start+10000]).cuda()
                    c=torch.full((len(x),1),math.log10(level),device='cuda')
                    one.append((x-model(x,torch.zeros_like(c),torch.ones_like(c),c)).cpu().numpy())
                    for t,s in ((1.,.5),(.5,0.)):
                        x=x-(t-s)*model(x,torch.full_like(c,s),torch.full_like(c,t),c)
                    two.append(x.cpu().numpy())
                generated=np.concatenate(two)
                difference=float(np.max(np.abs(np.concatenate(one)-baseline[seed])))
                assert difference<1e-5,(level,seed,difference)
                check={'sigma':level,'seed':seed,'one_step_replay_max_error':difference}
                if level==160:
                    previous=ROOT/'output/dp_alpha01_refinement_alpha10_20261017/multistep_particles/sigma160.npz'
                    with np.load(previous) as old:
                        error=float(np.max(np.abs(generated-old[f'seed{seed}_mf_two_step'])))
                    assert error<1e-5
                    check['two_step_replay_max_error']=error
                checks.append(check)
                clouds[f'seed{seed}_mf_two_step']=generated
        measures,origin,widths=r.measures_for(clouds)
        grid=r.js(EVAL/'results/ks3d'/f'sigma{level}'/'grid.json')
        np.testing.assert_array_equal(origin,grid['origin'])
        np.testing.assert_array_equal(widths,grid['widths'])
        with ThreadPoolExecutor(max_workers=3) as pool:
            scores=list(pool.map(lambda seed:r.solve(measures['reference'],measures[f'seed{seed}_mf_two_step']),r.SEEDS))
        for seed,score in zip(r.SEEDS,scores):
            rows.append({'model':'ks3d','sigma':level,'seed':seed,'method':'mf_two_step','particles':100000,'bins_per_axis':16,**score})
        print(f'Two-step MF sigma={level}: '+str([round(s['w2'],6) for s in scores]),flush=True)
    r.table(OUT/'multistep_metrics_per_seed.csv',rows)
    summary=[]
    for level in r.LEVELS:
        values=[row['w2'] for row in rows if row['sigma']==level]
        summary.append({'model':'ks3d','sigma':level,'method':'mf_two_step','mean_w2':statistics.mean(values),
                        'sd_w2':statistics.stdev(values),'min_w2':min(values),'max_w2':max(values),
                        'training_seeds':3,'particles':100000,'bins_per_axis':16})
    r.table(OUT/'multistep_summary.csv',summary)
    r.save(OUT/'multistep_verification.json',{'status':'passed','rows':60,'new_training':False,
        'sources':sources,'schedule':[[1,.5],[.5,0]],
        'grid':'Identical to the adopted 3D main comparison at each amplitude','checks':checks,
        'max_marginal_residual':max(row['marginal_residual'] for row in rows),
        'max_primal_dual_gap':max(row['primal_dual_gap'] for row in rows)})

def figures():
    summary=read(EVAL/'summary.csv')
    multi=read(OUT/'multistep_summary.csv')
    assert len(summary)==120 and len(multi)==20
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,
                         'axes.linewidth':.8,'legend.fontsize':9,'axes.labelsize':12})
    colors={'mf':'#2864B4','dp':'#D37B2F','mf_dp':'#168173','mf_two_step':'#D37B2F'}
    labels={'mf':'MF','dp':'Direct DP','mf_dp':'DP refinement','mf_two_step':'MF (2 steps)'}
    markers={'mf':'o','dp':'^','mf_dp':'s','mf_two_step':'^'}
    plotted=[]
    global_max=max(float(row['max_w2']) for row in summary)
    for name,methods,stem in [('ks3d',('mf','dp','mf_dp'),'ks3d_w2'),
                              ('ks4d',('mf','dp','mf_dp'),'ks4d_w2'),
                              ('ks3d',('mf','mf_two_step','mf_dp'),'ks3d_multistep_w2')]:
        fig,ax=plt.subplots(figsize=(5.2,3.6),layout='constrained')
        selected=[]
        for method in methods:
            values=sorted([row for row in summary+multi if row['model']==name and row['method']==method],key=lambda row:int(row['sigma']))
            assert len(values)==20
            x=np.array([int(row['sigma']) for row in values])
            y=np.array([float(row['mean_w2']) for row in values])
            low=np.array([float(row['min_w2']) for row in values])
            high=np.array([float(row['max_w2']) for row in values])
            assert (low<=y).all() and (y<=high).all()
            line,=ax.plot(x,y,color=colors[method],marker=markers[method],ms=3.5,lw=1.6,label=labels[method],zorder=3)
            np.testing.assert_array_equal(line.get_ydata(),y)
            ax.fill_between(x,low,high,color=colors[method],alpha=.13,lw=0,zorder=1)
            selected+=values
            plotted.extend({'figure':stem,**row} for row in values)
        top=global_max if 'multistep' not in stem else max(float(row['max_w2']) for row in selected)
        ax.set(xlabel=r'$\sigma$',ylabel=r'$W_2$',xlim=(5,205),ylim=(0,top*1.05))
        ax.set_xticks([20,50,100,150,200])
        ax.axvline(150,color='#777777',ls='--' if 'multistep' in stem else ':',lw=1.3)
        ax.grid(alpha=.23,ls='--')
        ax.legend(frameon=False,loc='upper left')
        for ext in ('pdf','png'):
            fig.savefig(FIG/f'{stem}.{ext}',dpi=250,facecolor='white')
        plt.close(fig)
    r.table(OUT/'plotted_values.csv',plotted)
    lookup={(row['model'],int(row['sigma']),row['method']):float(row['mean_w2']) for row in summary+multi}
    facts={}
    for name in ('ks3d','ks4d'):
        facts[name]={'refinement_below_MF':[s for s in r.LEVELS if lookup[name,s,'mf_dp']<lookup[name,s,'mf']],
                     'refinement_below_DP':[s for s in r.LEVELS if lookup[name,s,'mf_dp']<lookup[name,s,'dp']]}
    facts['ablation']={
        'two_step_below_MF':[s for s in r.LEVELS if lookup['ks3d',s,'mf_two_step']<lookup['ks3d',s,'mf']],
        'refinement_below_two_step':[s for s in r.LEVELS if lookup['ks3d',s,'mf_dp']<lookup['ks3d',s,'mf_two_step']],
        'sigma160':{m:lookup['ks3d',160,m] for m in ('mf','mf_two_step','mf_dp')}}
    r.save(OUT/'figure_verification.json',{'status':'passed','main_evaluations':360,'ablation_evaluations':60,
        'source_summary_sha256':r.sha(EVAL/'summary.csv'),'facts':facts,
        'figures':{str(path.relative_to(ROOT)):r.sha(path) for path in FIG.glob('*')}})
    print(json.dumps(facts,indent=2),flush=True)

if __name__=='__main__':
    if not (OUT/'multistep_verification.json').exists():prepare_ablation()
    figures()
