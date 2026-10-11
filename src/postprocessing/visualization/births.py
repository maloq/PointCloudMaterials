"""Portable birth-history movies and scientific property views of saved MD."""
import argparse
import base64
import gzip
import html
import io
import json
from pathlib import Path

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha,write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.postprocessing.histories.common import root
FAMILY = 'nucleus_histories'
from src.postprocessing.io import read

COLORS=['#93a1b0','#32cc95','#ffb453','#bd8cff','#9da6af','#9da6af','#9da6af']


def packet(folder):
    done=read(folder/'complete.json')
    if any(sha(folder/n)!=h for n,h in done['files'].items()):raise ValueError('Changed birth packet')
    with np.load(folder/'movie.npz') as a:movie={k:a[k] for k in a.files}
    return read(folder/'birth.json'),read(folder/'movie-frames.json'),movie,read(folder/'graph.json'),read(folder/'family-history.json')


def slices(movie,i):
    start,end=movie['offsets'][i:i+2]
    return {k:v[start:end] for k,v in movie.items() if k!='offsets'}


def render(c,slug):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    dest=root(c)/'analyses/births'/slug
    birth,frames,movie,graph,history=packet(dest)
    plots=dest/'plots';plots.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.facecolor':'#101923','axes.facecolor':'#152230','savefig.facecolor':'#101923',
        'text.color':'#e6edf4','axes.labelcolor':'#c7d5e3','xtick.color':'#b5c4d2','ytick.color':'#b5c4d2',
        'axes.edgecolor':'#4c6479','grid.color':'#354959','font.size':10})
    first=birth['first_time_ps'];confirm=birth['establishment_confirmation_ps']-first
    t=np.asarray([f['relative_time_ps'] for f in frames]);cuts=np.flatnonzero(np.diff(t)>.100001)+1
    segments=np.split(np.arange(len(t)),cuts)
    def line(ax,y,**kwargs):
        for n,idx in enumerate(segments):ax.plot(t[idx],np.asarray(y,dtype=float)[idx],**{k:v for k,v in kwargs.items() if k!='label' or n==0})
    def decorate(ax):
        ax.axvline(0,c='#ffffff',lw=.8,ls=':');ax.axvline(confirm,c='#ed82bc',lw=.8,ls=':')
        for cut in cuts:ax.axvspan(t[cut-1],t[cut],color='#7b8693',alpha=.15)
        ax.grid(alpha=.3);ax.set_xlabel('Time relative to catalogue embryo onset (ps)')
    fig,axes=plt.subplots(3,2,figsize=(13,11),layout='constrained')
    ht=np.array([h['time_ps']-first for h in history]);hy=np.array([h['size'] for h in history])
    # Break missing observations, including original trace gaps, rather than interpolate.
    for ii in np.split(np.arange(len(ht)),np.flatnonzero(np.diff(ht)>.100001)+1):axes[0,0].plot(ht[ii],hy[ii],c='#32cc95',lw=1.5)
    for val in (8,64):axes[0,0].axhline(val,c='#7890a4',ls='--',lw=.7)
    visible_sizes=hy[(ht>=min(t))&(ht<=max(t))]
    axes[0,0].set(xlim=(min(t),max(t)),ylim=(0,max(72,float(visible_sizes.max()))*1.08),ylabel='Largest tracked component (atoms)',title='Embryo → established nucleus')
    counts=np.array([[np.count_nonzero(slices(movie,i)['ptm']==k) for k in (1,2,3)] for i in range(len(frames))])
    for k,name in enumerate(('FCC','HCP','BCC')):line(axes[0,1],counts[:,k],c=COLORS[k+1],label=name)
    axes[0,1].set(ylabel='Atoms within the 14 Å crop',title='Crystal structures');axes[0,1].legend()
    for key,color,name in [('q6','#5dc8fa','q₆'),('qbar6','#f7d975','q̄₆'),('coherence6','#e08ec1','Neighbour coherence')]:
        line(axes[1,0],[f[key] for f in frames],c=color,label=name)
    axes[1,0].set(ylabel='Order parameter',title='Bond order around the tracked atom');axes[1,0].legend()
    line(axes[1,1],[f['radius_gyration_A'] for f in frames],c='#7de0cf',label='Gyration radius')
    axes[1,1].set(ylabel='Visible component radius (Å)',title='Component shape within the crop')
    twin=axes[1,1].twinx();line(twin,[f['anisotropy'] for f in frames],c='#ffb453');twin.set(ylabel='Anisotropy (orange)',ylim=(0,1.05))
    for key,color,name in [('founder','#f6ce6b','Original founder atoms'),('precursor','#67dfea','Linked precursor atoms'),('catalogued_family','#ee9bd3','Tracked family atoms')]:
        line(axes[2,0],[int(slices(movie,i)[key].sum()) for i in range(len(frames))],c=color,label=name)
    line(axes[2,0],[int((slices(movie,i)['founder']&np.isin(slices(movie,i)['ptm'],[1,2,3])).sum()) for i in range(len(frames))],c='#ffffff',ls=':',label='Crystalline founder atoms')
    axes[2,0].set(ylabel='Visible atoms',title='Membership and retention');axes[2,0].legend()
    line(axes[2,1],[f['crystal_fraction'] for f in frames],c='#32cc95');axes[2,1].set(ylabel='Crystalline fraction in crop',title='Local crystallinity and spacing',ylim=(0,1))
    twin=axes[2,1].twinx();line(twin,[f['first_shell_A'] for f in frames],c='#6dbdf4');twin.set_ylabel('Mean first-shell spacing (Å, blue)')
    for ax in axes.flat:decorate(ax)
    fig.suptitle(slug+' | actual saved frames, 0.1 ps cadence\nGrey bands: no detailed atom movie; dotted lines: first embryo and establishment confirmation',fontsize=13)
    fig.savefig(plots/'properties.png',dpi=150);fig.savefig(plots/'properties.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    for ii in np.split(np.arange(len(ht)),np.flatnonzero(np.diff(ht)>.100001)+1):ax.plot(ht[ii],hy[ii],c='#32cc95',lw=1)
    ax.axhline(64,c='#819bad',ls='--');ax.axvline(confirm,c='#ed82bc',ls=':');ax.set(xlabel='Time since first embryo (ps)',ylabel='Largest family component (atoms)',title='Full original recorded family history');ax.grid(alpha=.3)
    fig.savefig(plots/'full-history.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained');nodes={n['id']:n for n in graph['nodes']}
    for edge in graph['edges']:
        a,b=nodes[edge['parent']],nodes[edge['child']];kind=edge['kind']
        ax.plot([a['time_ps']-first,b['time_ps']-first],[a['size'],b['size']],
            c='#59d6a5' if kind=='direct' else '#ffb453' if kind=='gap' else '#7b8da0',ls='--' if kind=='gap' else '-',alpha=.55 if kind!='weak_overlap' else .25,lw=1)
    for status,color in [('direct','#59d6a5'),('gap','#ffb453'),('weak_only','#8190a2')]:
        subset=[n for n in nodes.values() if n['path_status']==status]
        ax.scatter([n['time_ps']-first for n in subset],[n['size'] for n in subset],s=16,c=color,label=status)
    ax.set(xlabel='Time before first embryo (ps)',ylabel='Component size (atoms)',title='Precursor branches: shared atom identity, including splits and mergers');ax.legend();ax.grid(alpha=.3)
    fig.savefig(plots/'precursor-graph.png',dpi=150);plt.close(fig)
    # Two separate GIFs avoid a fabricated continuation through disjoint excerpts.
    for phase,indices in [('early-appearance',np.flatnonzero((t>=max(min(t),graph['earliest_with_gaps_ps']-first-1)-1e-8)&(t<=c['early_after_ps']+1e-8))),
                          ('establishment',np.flatnonzero((t>=confirm-c['establishment_before_ps']-1e-8)&(t<=confirm+c['establishment_after_ps']+1e-8)))]:
        chosen=indices[::2].tolist()
        if indices[-1]!=chosen[-1]:chosen.append(int(indices[-1]))
        images=[]
        for i in chosen:
            frame=slices(movie,i);q=frame['positions'];angle=.45;X=q[:,0]*np.cos(angle)+q[:,2]*np.sin(angle);Z=-q[:,0]*np.sin(angle)+q[:,2]*np.cos(angle)
            Y=q[:,1]*.92-Z*.38;screen=np.column_stack((360+X*18,305-Y*18))
            im=Image.new('RGB',(1024,650),'#101923');draw=ImageDraw.Draw(im)
            for j in np.argsort(Z):
                crystal=int(frame['ptm'][j]);tracked=frame['precursor'][j] or frame['catalogued_family'][j]
                rad=4.8 if crystal in (1,2,3) else 1.3;color=COLORS[crystal] if crystal in (1,2,3) else '#405369'
                x,y=screen[j];draw.ellipse((x-rad,y-rad,x+rad,y+rad),fill=color,outline='#f6f8ff' if tracked else None)
            draw.text((25,18),slug+' / '+phase,fill='#e6edf4',font_size=23)
            draw.text((25,51),f"t = {frames[i]['time_ps']:.1f} ps | embryo-relative {t[i]:+.1f} ps",fill='#c4d5e3',font_size=19)
            info=[('FCC',COLORS[1]),('HCP',COLORS[2]),('BCC',COLORS[3]),('Liquid / other','#93a1b0'),('White rim: linked family','#ffffff')]
            for j,(label,color) in enumerate(info):draw.text((727,135+29*j),label,fill=color,font_size=17)
            text=[f"Tracked size: {frames[i]['largest_family']} atoms",f"Local crystal: {100*frames[i]['crystal_fraction']:.1f}%",f"q6: {frames[i]['q6']:.3f}",f"qbar6: {frames[i]['qbar6']:.3f}",f"14 A crop; saved MD",f"0.2 ps GIF playback stride"]
            for j,label in enumerate(text):draw.text((727,324+29*j),label,fill='#c4d5e3',font_size=17)
            draw.line((28,606,995,606),fill='#405369',width=4);frac=(i-chosen[0])/max(1,chosen[-1]-chosen[0]);draw.line((28,606,28+967*frac,606),fill='#59d6a5',width=4)
            images.append(im)
        images[0].save(plots/(phase+'.gif'),save_all=True,append_images=images[1:],duration=130,loop=0,optimize=False)
    payload=dict(birth=birth,frames=frames,graph=graph,history=history,
        movie=[{k:np.round(v,3).tolist() if k=='positions' else v.astype(int).tolist() for k,v in slices(movie,i).items()} for i in range(len(frames))])
    packed=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':')).encode())).decode()
    template=Path(__file__).with_name('birth_history_view.html').read_text()
    (dest/'viewer.html').write_text(template.replace('__PAYLOAD__',packed).replace('__SLUG__',html.escape(slug)))
    write_metric_rows(frames,dest,family=c.get('metric_family',FAMILY),name='movie-properties')
    write_json(dest/'technical/visualization.json',dict(renderer_sha256=sha(Path(__file__)),template_sha256=sha(Path(__file__).with_name('birth_history_view.html')),source_packet_sha256=sha(dest/'complete.json'),gif_stride_ps=.2,files={str(f.relative_to(dest)):sha(f) for f in sorted(plots.iterdir())}))
    print('Rendered',slug,flush=True)


def index(c):
    r=root(c);p=read(r/'technical/plan.json');rows=[];completed=0;keys=set()
    for e in p['events']:
        d=r/'analyses/births'/e['slug'];ready=(d/'complete.json').exists();view=(d/'viewer.html').exists()
        if ready:
            s=read(d/'summary.json');rows.append(s);completed+=1;keys.add(s['physical_birth_key'])
    if rows:write_metric_rows(rows,r,family=FAMILY,name='birth-histories')
    cards=[]
    for e in sorted(p['events'],key=lambda e:(e['slug'] not in p['examples'],e['first_time_ps'])):
        d=r/'analyses/births'/e['slug'];href='analyses/births/'+e['slug']
        ready=(d/'complete.json').exists();view=(d/'viewer.html').exists()
        title=html.escape(e['slug']);links=f'<a href="{href}/viewer.html">Interactive viewer</a> · <a href="{href}/plots/properties.png">Properties</a> · <a href="{href}/plots/precursor-graph.png">Precursor graph</a> · <a href="{href}/plots/early-appearance.gif">Early GIF</a> · <a href="{href}/plots/establishment.gif">Growth GIF</a>' if view else ('History reconstructed; atom packet available' if ready else 'Reconstruction pending')
        cards.append(f'<article><b>{title}</b><p>{html.escape(e["source"])} · {html.escape(e["role"])} · first embryo {e["first_time_ps"]:.1f} ps → establishment {e["establishment_confirmation_ps"]:.1f} ps</p>{links}</article>')
    page='''<!doctype html><meta charset="utf-8"><title>Nucleus births and precursor histories</title><style>body{font:16px system-ui;background:#101923;color:#dce8f3;max-width:1180px;margin:40px auto;padding:0 22px}a{color:#7dd7ff}article{padding:18px;background:#182735;border-radius:10px;margin:12px 0}p{color:#acbed0;line-height:1.6}h1{font-size:32px}</style><h1>Nucleus births & precursor histories</h1>'''
    page+=f'<p>{completed}/{len(p["events"])} original birth records reconstructed; {len(keys)} distinct physical onset keys among completed views. Source-view duplicates are aliases, not independent births.</p>'
    page+='<p>Each entry is an existing established nucleus. Earlier 1–7 atom components are history nodes, not additional births. The three illustrated examples have short formation intervals on different ancestors; they are not a representative performance sample. All figures use saved MD at 0.1 ps. Detailed movies cover the early precursor and establishment excerpts; the full size trace spans the intervening history. Uncertain links and missing movie intervals stay explicit.</p>'
    page+=''.join(cards);(r/'index.html').write_text(page)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('render','examples','index'));p.add_argument('--config',required=True);p.add_argument('--event');a=p.parse_args();c=read(a.config)
    if a.stage=='render':render(c,a.event)
    elif a.stage=='examples':
        for slug in read(root(c)/'technical/plan.json')['examples']:render(c,slug)
    index(c)


if __name__=='__main__':main()
