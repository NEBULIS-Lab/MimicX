"""Draw readable website-only figures from the released evidence CSVs.

Contract: show direct tracking accuracy, scene transfer, execution length,
feedback cost and training dynamics as distinct pieces of evidence. Export
editable SVG plus PNG, explicit dark/light palettes and no embedded titles.
All uncertainty and sample units are stated in the HTML captions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = [('tennis', 'Tennis'), ('football1', 'Football'), ('dance2', 'Dance'), ('kongfu1', 'Kung Fu')]
METHODS = ['m0_open_loop', 'm1_policy_window', 'm2_task_hierarchy', 'm3_full_mimicx']


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def import_evidence(paper, directory):
    direct = paper / 'figures/direct_update/direct_update.csv'
    values = rows(direct)
    for row in values:
        row['source'] = '/'.join(Path(row['source']).parts[-3:])
    with (directory / 'direct_policy_comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    source = paper / 'figures/collision/collision_results.csv'
    (directory / 'collision_scene_results.csv').write_bytes(source.read_bytes())
    provenance = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (direct, source)}
    (directory / 'paper_evidence_sources.json').write_text(json.dumps(provenance, indent=2)+'\n')


def build(directory, output):
    output.mkdir(parents=True, exist_ok=True)
    direct = rows(directory / 'direct_policy_comparison.csv')
    collision = rows(directory / 'collision_scene_results.csv')
    core = rows(directory / 'core_trial_results.csv')
    timing = rows(directory / 'hloop_timings.csv')
    dynamics = rows(directory / 'tennis_training_dynamics_3000_points.csv')
    manifest = []
    for theme in ['light', 'dark']:
        dark = theme == 'dark'
        bg, fg, grid = ('#1e2125', '#f1f0ee', '#3c424a') if dark else ('#f4f5f6', '#22262c', '#d7dce1')
        colors = ['#81b5df', '#b49ad9', '#82c7b0', '#ef8b7b'] if dark else ['#3B78A8', '#8266A6', '#23866B', '#D45B4C']
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18,
            'svg.fonttype': 'none', 'svg.hashsalt': 'mimicx-web-evidence', 'axes.labelsize': 18,
            'xtick.labelsize': 16, 'ytick.labelsize': 17, 'text.color': fg,
            'axes.labelcolor': fg, 'xtick.color': fg, 'ytick.color': fg,
            'axes.edgecolor': grid, 'figure.facecolor': bg, 'axes.facecolor': bg,
            'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False})

        def canvas(wide=False):
            fig, ax = plt.subplots(figsize=(13.5, 4.6) if wide else (7, 6.4))
            fig.subplots_adjust(left=.1 if wide else .30, right=.95, top=.94, bottom=.18)
            ax.set_axisbelow(True)
            ax.grid(axis='x', color=grid, linewidth=.7)
            ax.tick_params(axis='y', length=0, pad=10)
            return fig, ax

        def save(fig, name, sources, contract):
            stem = f'{name}-{theme}'
            fig.savefig(output/f'{stem}.svg', metadata={'Date': None, 'Creator': 'MimicX'})
            fig.savefig(output/f'{stem}.png', dpi=180, metadata={'Software': 'MimicX'})
            manifest.append(dict(file=stem+'.svg', sha256=hashlib.sha256((output/f'{stem}.svg').read_bytes()).hexdigest(), sources=sources, contract=contract))
            if not name.startswith('learning'):
                fig.set_size_inches(5, 5.4)
                fig.subplots_adjust(left=.38, right=.96, top=.94, bottom=.20)
                if name == 'execution-horizon':
                    fig.axes[0].set_xlabel('Failure horizon (steps)')
                mobile_stem = f'{name}-mobile-{theme}'
                fig.savefig(output/f'{mobile_stem}.svg', metadata={'Date': None, 'Creator': 'MimicX'})
                fig.savefig(output/f'{mobile_stem}.png', dpi=180, metadata={'Software': 'MimicX'})
                manifest.append(dict(file=mobile_stem+'.svg', sha256=hashlib.sha256((output/f'{mobile_stem}.svg').read_bytes()).hexdigest(), sources=sources, contract=contract+'; mobile layout'))
            plt.close(fig)

        fig, ax = canvas()
        order = [('m0_open_loop', 'Fixed\nReference', colors[0]), ('BeyondMimic_MjLab', 'BeyondMimic\n(MjLab)', colors[1]), ('SONIC', 'SONIC', colors[2]), ('m3_full_mimicx', 'MimicX', colors[3])]
        for i, (method, label, color) in enumerate(order):
            vals = np.array([float(r['root_local_fk_body_mean_m'])*100 for r in direct if r['task']=='tennis' and r['method']==method])
            assert len(vals)==3
            ax.scatter(vals, i+np.array([-.09, 0, .09]), s=38, color=color, alpha=.65, zorder=3)
            ax.errorbar(vals.mean(), i, xerr=vals.std(ddof=1), fmt='D', ms=8, color=color, capsize=5, lw=2.3, zorder=4)
            ax.text(min(vals.mean()+vals.std(ddof=1)+1.3, 26), i-.2, f'{vals.mean():.1f}', color=color, fontsize=17)
        ax.set_yticks(range(4), [x[1] for x in order])
        ax.invert_yaxis()
        ax.set_xlim(0, 28)
        ax.set_ylim(3.6, -.65)
        ax.set_xlabel('Body error (cm)')
        save(fig, 'direct-tennis', ['direct_policy_comparison.csv'], 'Three recorded runs per method; mean and sample SD; common reference; lower is better')

        fig, ax = canvas()
        for i, row in enumerate(collision):
            a, b = float(row['body_reduction_percent']), float(row['root_reduction_percent'])
            ax.plot([a, b], [i, i], color=grid, lw=3)
            ax.scatter(a, i, s=100, color=colors[3], zorder=3)
            ax.scatter(b, i, s=90, marker='s', color=colors[2], zorder=3)
        ax.set_yticks(range(5), ['Track', 'Stairs', 'Forest', 'Platform', 'Parkour'])
        ax.set_ylim(4.6, -.6)
        ax.set_xlim(-4, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('p95 error reduction (%)')
        save(fig, 'scene-gains', ['collision_scene_results.csv'], 'Body circle / root square; paired pre-failure prefixes; one training seed and three evaluation seeds')

        fig, ax = canvas()
        for i, (task, label) in enumerate(TASKS):
            for j, seed in enumerate(['101', '202', '303']):
                pair = [next(r for r in core if r['task_id']==task and r['train_seed']==seed and r['method_id']==m) for m in ['m0_open_loop', 'm3_full_mimicx']]
                a, b = [float(r['worst_first_failure_step']) for r in pair]
                y = i+(j-1)*.16
                ax.plot([a,b], [y,y], lw=1.5, color=grid)
                ax.scatter(a,y,color=colors[0],s=45,zorder=3)
                ax.scatter(b,y,color=colors[3],s=58,marker='D',zorder=3)
        ax.set_yticks(range(4), [x[1] for x in TASKS])
        ax.set_ylim(3.55,-.55)
        ax.set_xlim(0,850)
        ax.set_xticks([0,200,400,600,800])
        ax.set_xlabel('First-failure horizon (steps)')
        save(fig, 'execution-horizon', ['core_trial_results.csv'], 'All twelve paired task-seed results, worst repeat per trial; H+1 encodes no failure')

        fig, ax = canvas()
        for i, (method,color) in enumerate(zip(['sequential','bulk_sync','e0'], [colors[0],colors[2],colors[3]])):
            vals = np.array([float(r['wall_time_seconds']) for r in timing if r['scheduler']==method])
            ax.scatter(vals, i+np.linspace(-.16,.16,5), s=65, color=color, alpha=.7)
            ax.plot([np.median(vals)]*2,[i-.26,i+.26],color=color,lw=3)
            ax.text(np.median(vals)-4,i-.42,f'{np.median(vals):.1f} s',ha='right',color=color,fontsize=18)
        ax.set_yticks(range(3), ['Sequential','Bulk-sync','HLoop'])
        ax.set_ylim(2.65,-.8)
        ax.set_xlim(100,260)
        ax.set_xticks([120,160,200,240])
        ax.set_xlabel('Feedback wall time (s)')
        save(fig, 'feedback-time', ['hloop_timings.csv'], 'Five measured workloads per executor; vertical tick is median; no PPO updates')

        fig, ax = canvas(wide=True)
        ax.spines['left'].set_visible(True)
        ax.grid(axis='y',color=grid,lw=.7)
        for method,color in zip(METHODS,colors):
            seeds=[]
            for seed in ['101','202','303']:
                trace=sorted([r for r in dynamics if r['method']==method and r['train_seed']==seed],key=lambda r:int(r['relative_iteration']))
                x=np.array([int(r['relative_iteration']) for r in trace])
                y=np.array([float(r['error_body_pos']) for r in trace])
                ax.plot(x,y,lw=.8,alpha=.20,color=color)
                seeds.append(y)
            ax.plot(x,np.mean(seeds,axis=0),color=color,lw=2.3)
        ax.set_yscale('log')
        ax.set_yticks([.1,.2,.5,1], ['0.1','0.2','0.5','1.0'])
        ax.set_ylim(.07,1.2)
        ax.set_xlim(0,249)
        ax.set_xlabel('Continuation iteration')
        ax.set_ylabel('Body error (m; log scale)')
        save(fig, 'learning', ['tennis_training_dynamics_3000_points.csv'], 'All 3,000 logged values, unsmoothed; thin lines individual seeds, thick lines three-seed means')
        # A narrow-screen derivative changes layout, never the sampled data.
        fig, ax = canvas()
        fig.set_size_inches(5, 5.4)
        fig.subplots_adjust(left=.23, right=.96, bottom=.17)
        ax.spines['left'].set_visible(True)
        ax.grid(axis='y', color=grid, lw=.7)
        for method, color in zip(METHODS, colors):
            traces = []
            for seed in ['101', '202', '303']:
                trace = sorted([r for r in dynamics if r['method']==method and r['train_seed']==seed], key=lambda r:int(r['relative_iteration']))
                x = [int(r['relative_iteration']) for r in trace]
                y = [float(r['error_body_pos']) for r in trace]
                ax.plot(x, y, lw=.7, alpha=.2, color=color)
                traces.append(y)
            ax.plot(x, np.mean(traces,axis=0), color=color, lw=2)
        ax.set_yscale('log')
        ax.set_yticks([.1,.2,.5,1], ['0.1','0.2','0.5','1.0'])
        ax.set_ylim(.07,1.2)
        ax.set_xlim(0,249)
        ax.set_xlabel('Continuation iteration')
        ax.set_ylabel('Body error (m; log scale)')
        save(fig, 'learning-mobile', ['tennis_training_dynamics_3000_points.csv'], 'Same unsmoothed 3,000 observations; mobile layout')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',type=Path)
    parser.add_argument('--data',type=Path,default=ROOT/'docs/assets/results')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/assets/media/evidence')
    args=parser.parse_args()
    if args.paper: import_evidence(args.paper,args.data)
    build(args.data,args.output)
