import json
import os
from pathlib import Path
import sys

ROOT = Path('/mnt/e/vfsimulator')
sys.path.insert(0, str(ROOT))
from api.ub_dependency_experiment import run_cce_ub_dependency_comparison

OUT = ROOT/'results/ub_address_dependency_experiment/adam_apply_one_fused_i48_unroll'
rows = []
for u in (1, 2, 4, 6, 8):
    case = OUT/f'u{u}'
    original = json.loads((case/'summary.json').read_text())
    if u < 6:
        comparison = run_cce_ub_dependency_comparison(
            case/f'source/adam_i48_u{u}.cce', soc='dav-3510', kernel_name='adam_fused_vf',
            baseline_out_dir=case/'mode_comparison/global', local_out_dir=case/'mode_comparison/local')
        (case/'mode_comparison/summary.json').write_text(json.dumps(comparison, indent=2)+'\n')
        assert comparison['baseline']['vf_end_cycle'] == original['vfsim_cycles']
        origin = 'source CCE (no compiler spill)'
    else:
        comparison = json.loads((case/'compiled_replay/summary.json').read_text())
        origin = 'compiled PC replay (with spill)'
    rows.append(dict(unroll=u, camodel=original['vf_cycles'],
                     global_cycles=comparison['baseline']['vf_end_cycle'],
                     local_cycles=comparison['local']['vf_end_cycle'], input=origin))
    print(rows[-1], flush=True)

sys.path.insert(0, '/tmp/vfsim_dag_plot_deps')
os.environ.setdefault('MPLCONFIGDIR','/tmp/vfsim_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = OUT/'plots'
folder.mkdir(exist_ok=True)
(folder/'unroll_cycles_data.json').write_text(json.dumps(rows, indent=2)+'\n')
plt.rcParams.update({'font.size':17, 'axes.titlesize':21, 'axes.labelsize':19, 'legend.fontsize':16})
fig, ax = plt.subplots(figsize=(14,8))
x = [r['unroll'] for r in rows]
for key,label,color,marker,style in (
    ('camodel','A5 CAmodel','#dc3545','o','-'),
    ('global_cycles','VfSim global Membar','#2563eb','s','-'),
    ('local_cycles','VfSim local UB dependency','#109447','^','--')):
    ax.plot(x,[r[key] for r in rows],label=label,color=color,marker=marker,
            linestyle=style,linewidth=2.8,markersize=9)
    for r in rows:
        offset = 12 if key=='camodel' else -25 if key=='local_cycles' else -25
        if key=='global_cycles' and r['unroll']<6:
            continue
        ax.annotate(str(r[key]),(r['unroll'],r[key]),xytext=(0,offset),
                    textcoords='offset points',ha='center',color=color,fontsize=14)
ax.set(title='AdamApplyOne FP32: Unroll vs. VF Cycles (A5, 48 iterations)',
       xlabel='Unroll factor',ylabel='VF cycles',xticks=x,xlim=(0.6,8.5),ylim=(0,4200))
ax.grid(alpha=.23)
ax.legend(loc='upper left')
fig.text(.08,.06,'U1/U2/U4: source CCE; U6/U8: compiled spill replay. Overlapping blue/green points share values.',fontsize=12)
fig.text(.08,.032,'U12/U16 omitted: CAmodel compile failure. VfSim includes VSQRT timing fallback; scalar execution time excluded.',fontsize=12)
fig.tight_layout(rect=(0,.1,1,1))
for ext in ('png','svg'):
    fig.savefig(folder/f'unroll_cycles_comparison.{ext}',dpi=180)
print(folder/'unroll_cycles_comparison.png')
