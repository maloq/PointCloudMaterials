"""Illustrative final-frame sections from explicitly selected Al replay pairs."""
import argparse
import os
from pathlib import Path
import json

os.environ['OVITO_THREAD_COUNT'] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.analysis.al_replay import structure
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.transfer import write_json
from src.experiment_runner.metric_docs import fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source-ids',type=int,nargs='+',required=True)
    args=parser.parse_args()
    protocol=json.loads((args.root/'technical/protocol.json').read_text())
    records={r['source_id']:r for r in protocol['records']}
    colors=['#b9bdc5','#219e78','#e8b44c','#9164b7','#407fc0']
    names=['Other','FCC','HCP','BCC','ICO']
    fig,axes=plt.subplots(len(args.source_ids),2,figsize=(11,5.4*len(args.source_ids)),squeeze=False,layout='constrained')
    receipts=[]
    for row,sid in enumerate(args.source_ids):
        record=records[sid]
        for col,(label,directory) in enumerate([('Original',record['parent_directory']),('Rerun',record['new_directory'])]):
            trajectory=ShootingBinaryTrajectory.load(Path(directory)/'trajectory_binary_float16')
            lengths=(trajectory.box_high[-1]-trajectory.box_low[-1]).astype(float)
            positions=np.mod(trajectory.positions[-1].astype(float),lengths)
            labels,stats=structure(positions,lengths)
            mask=abs(positions[:,2]-lengths[2]/2)<4
            ax=axes[row,col]
            ax.scatter(positions[mask,0],positions[mask,1],c=labels[mask],cmap=ListedColormap(colors),vmin=-.5,vmax=4.5,s=5,linewidths=0)
            ax.set(xlim=(0,lengths[0]),ylim=(0,lengths[1]),xlabel='x (Å)',ylabel='y (Å)',
                title=f'{label} • parent {sid} • 600 ps\nFull-cell crystalline fraction: {stats["crystal_fraction"]:.1%}')
            ax.set_aspect('equal')
            receipts.append(dict(source_id=sid,version=label,frame=int(trajectory.frame_count-1),
                binary_manifest_sha256=fingerprint(trajectory.root/'manifest.json'),
                full_cell_stats=stats,display_atoms=int(mask.sum()),slice_half_thickness_A=4))
    fig.legend(handles=[Patch(color=color,label=name) for color,name in zip(colors,names,strict=True)],loc='outside lower center',ncol=5)
    fig.suptitle('Same prepared liquid, different crystallization outcome\n8 Å midplane sections; full-cell PTM, RMSD cutoff 0.1',fontsize=14)
    fig.savefig(args.root/'plots/opposite_outcome_examples.png',dpi=180)
    write_json(args.root/'technical/example_sections.json',dict(
        selection='Illustrative opposite-sign extremes of final potential-energy change, selected after inspecting all 20 pairs; not a representative subset.',
        producer_sha256=fingerprint(Path(__file__)),frames=receipts))


if __name__=='__main__':main()
