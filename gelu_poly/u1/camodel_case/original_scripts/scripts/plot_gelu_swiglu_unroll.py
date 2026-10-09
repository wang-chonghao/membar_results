import json
import os
from pathlib import Path
import sys

ROOT=Path('/mnt/e/vfsimulator')
sys.path.insert(0,str(ROOT))
from api.ub_dependency_experiment import run_cce_ub_dependency_comparison

cases=[('gelu_poly_i96_unroll_spill_20260911','GeLU_poly', (1,2,4,6,8,12), {12}),
       ('swiglu_grad_fp32_i96/unroll_scan_20260914','SwiGLUGrad FP32', (1,2,4,6,8), {6,8})]
datasets=[]
for rel,title,factors,spilled in cases:
    base=ROOT/'results/ub_address_dependency_experiment'/rel
    prior={r['unroll']:r for r in json.loads((base/'summary.json').read_text())}
    rows=[]
    for u in factors:
        case=base/f'u{u}'
        source=case/'compiled_replay/source/compiled_pc_replay.cce' if u in spilled else next((case/'source').glob('*.cce'))
        result=run_cce_ub_dependency_comparison(source,soc='dav-3510',
            baseline_out_dir=case/'plot_mode_comparison_20260915/global',
            local_out_dir=case/'plot_mode_comparison_20260915/local')
        (case/'plot_mode_comparison_20260915/summary.json').write_text(json.dumps(result,indent=2)+'\n')
        rows.append(dict(unroll=u,camodel=prior[u]['timing']['vf_total_cycles'],
            global_cycles=result['baseline']['vf_end_cycle'],local_cycles=result['local']['vf_end_cycle'],
            source=str(source),input='compiled spill replay' if u in spilled else 'source CCE'))
        print(title,rows[-1],flush=True)
    datasets.append((base,title,rows,spilled))

sys.path.insert(0,'/tmp/vfsim_dag_plot_deps')
os.environ.setdefault('MPLCONFIGDIR','/tmp/vfsim_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':17,'axes.titlesize':21,'axes.labelsize':19,'legend.fontsize':16})
for base,title,rows,spilled in datasets:
    folder=base/'plots'
    folder.mkdir(exist_ok=True)
    (folder/'unroll_cycles_data_20260915.json').write_text(json.dumps(rows,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(14,8))
    x=[r['unroll'] for r in rows]
    for key,label,color,marker,style in (
        ('camodel','A5 CAmodel','#dc3545','o','-'),
        ('global_cycles','VfSim global Membar','#2563eb','s','-'),
        ('local_cycles','VfSim local UB dependency','#109447','^','--')):
        ax.plot(x,[r[key] for r in rows],label=label,color=color,marker=marker,
                linestyle=style,linewidth=2.8,markersize=9)
        for r in rows:
            if key=='global_cycles' and r['global_cycles']==r['local_cycles']:
                continue
            offset=12 if key=='camodel' else -25
            if r['unroll'] not in spilled and r['global_cycles']>r['camodel']:
                offset=-25 if key=='camodel' else 12
            ax.annotate(str(r[key]),(r['unroll'],r[key]),xytext=(0,offset),
                        textcoords='offset points',ha='center',color=color,fontsize=14)
    ax.set(title=f'{title}: Unroll vs. VF Cycles (A5, 96 iterations)',xlabel='Unroll factor',
           ylabel='VF cycles',xticks=x,xlim=(.5,max(x)+.65),ylim=(0,max(r['camodel'] for r in rows)*1.13))
    ax.grid(alpha=.23)
    ax.legend(loc='upper left')
    spill_text='/'.join('U'+str(u) for u in sorted(spilled))
    fig.text(.08,.06,f'{spill_text}: compiled spill replay; other points: source CCE. Blue/green overlap when equal.',fontsize=12)
    fig.text(.08,.032,'Only successful CAmodel cases shown. CAmodel uses archived measurements; VfSim excludes scalar execution time.',fontsize=12)
    fig.tight_layout(rect=(0,.1,1,1))
    for ext in ('png','svg'):
        fig.savefig(folder/f'unroll_cycles_comparison_20260915.{ext}',dpi=180)
    plt.close(fig)
    print(folder/'unroll_cycles_comparison_20260915.png',flush=True)
