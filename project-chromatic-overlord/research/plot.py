#!/usr/bin/env python3
"""Standalone publication/export figures from CSV; never reruns the engine."""
import argparse,csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/chromatic-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
MODELS=['zero128','lichess100','lichess1000','lichess10000'];OS=['Linux','Windows'];BITS='CAFTP'
def rows(p):return list(csv.DictReader(p.open()))
def name(m):return ''.join(c for i,c in enumerate(BITS) if m>>i&1) or 'base'
def export(fig,out,stem):
    fig.savefig(out/(stem+'.svg'),bbox_inches='tight');fig.savefig(out/(stem+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();out=a.analysis/'figures';out.mkdir(exist_ok=True)
    data=rows(a.analysis/'matrix.csv');screen=[r for r in data if r['stage']=='screen'];plan=json.loads(a.plan.read_text())
    fig,ax=plt.subplots(figsize=(14,4.3));matrix=np.full((8,32),np.nan)
    for r in screen:matrix[OS.index(r['os'])*4+MODELS.index(r['model']),int(r['mask'])]=100*float(r['accuracy'])
    im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=100,cmap='viridis')
    for i in range(8):
        for j in range(32):
            if np.isfinite(matrix[i,j]):ax.text(j,i,str(int(matrix[i,j])),ha='center',va='center',fontsize=6,color='white' if matrix[i,j]<55 else 'black')
    ax.set(xticks=range(32),xticklabels=[name(m) for m in range(32)],yticks=range(8),yticklabels=[f'{o} / {m}' for o in OS for m in MODELS],
           title='100 ms full factorial screening: exact expected response matches / 100 (discovery set)')
    plt.setp(ax.get_xticklabels(),rotation=65,ha='right',fontsize=8);ax.axvline(15.5,color='white',linestyle='--',linewidth=1);ax.axhline(3.5,color='white',linewidth=1)
    fig.colorbar(im,ax=ax,label='Accuracy (%)',fraction=.02);fig.text(.1,-.01,'C: counterexample   A: adaptive   F: forcing   T: tactical   P: two threads (other cells use one)',fontsize=9);export(fig,out,'factorial-screen')
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True)
    for i,o in enumerate(OS):
        for j,m in enumerate(['zero128','lichess10000']):
            ax=axes[i,j];subset=[r for r in screen if r['os']==o and r['model']==m]
            for r in subset:
                mask=int(r['mask']);ax.scatter(float(r['mean_cpu_ms']),100*float(r['accuracy']),marker='^' if mask&16 else 'o',c='#9ba9b4' if mask&16 else '#2471a3',s=24)
                if mask in {0,8,15,plan['best_serial'][m]}:ax.annotate(name(mask),(float(r['mean_cpu_ms']),100*float(r['accuracy'])),xytext=(3,3),textcoords='offset points',fontsize=8)
            ax.set(title=f'{o} / {m}',xlabel='Mean process CPU ms / puzzle',ylabel='Exact matches (%)');ax.grid(alpha=.2)
    fig.suptitle('Equal requested time does not imply equal CPU cost (100 ms discovery set)');fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color='#2471a3',label='1 thread'),Line2D([],[],marker='^',linestyle='',color='#9ba9b4',label='2 threads')],loc='lower center',ncol=2);fig.tight_layout(rect=(0,.03,1,.96));export(fig,out,'accuracy-cpu-cost')
    data=rows(a.analysis/'shapley.csv');fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
    colors=['#2e86c1','#d68910','#1e8449','#8e44ad']
    for i,o in enumerate(OS):
        for j,m in enumerate(['zero128','lichess10000']):
            ax=axes[i,j];rr=[r for r in data if r['stage']=='screen' and r['os']==o and r['model']==m and r['design']=='serial4']
            rr.sort(key=lambda r:BITS.index(r['factor']))
            v=np.array([float(r['shapley_pp']) for r in rr]);lo=np.array([float(r['lo95_pp']) for r in rr]);hi=np.array([float(r['hi95_pp']) for r in rr])
            ax.bar(range(len(rr)),v,color=colors[:len(rr)]);ax.errorbar(range(len(rr)),v,yerr=[np.maximum(0,v-lo),np.maximum(0,hi-v)],fmt='none',color='black',capsize=3)
            ax.set(title=f'{o} / {m}',xticks=range(len(rr)),xticklabels=[r['factor'] for r in rr],ylabel='Shapley accuracy contribution (pp)');ax.axhline(0,color='black',linewidth=.6);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Serial four-factor Shapley: discovery metric, pointwise cluster-bootstrap 95% intervals');fig.tight_layout();export(fig,out,'serial-shapley')
if __name__=='__main__':main()
