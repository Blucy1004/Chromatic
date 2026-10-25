"""Paired estimands; dependencies limited to NumPy. Units are puzzles/source games."""
import itertools, math
import numpy as np

def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[1.]*len(values);last=0.
    for j,i in enumerate(order):
        last=max(last,min(1.,values[i]*(len(values)-j)));out[i]=last
    return out

def mcnemar(gains,losses):
    n=int(gains+losses)
    if not n:return 1.
    # Integer arithmetic avoids overflow or cancellation in the exact binomial tail.
    return min(1.,2*sum(math.comb(n,k) for k in range(min(gains,losses)+1))/2**n)

def cluster_bootstrap(values,groups=None,repetitions=5000,seed=240125,alpha=.05):
    """Percentile intervals of puzzle-weighted means, resampling source-game clusters.

    Equal game weights are NOT substituted for puzzle weights. One game can have
    multiple old puzzles; holdout has exactly one puzzle per game. A simultaneous
    interval uses the bootstrap maximum absolute studentized centered error.
    """
    x=np.asarray(values,dtype=float)
    if x.ndim==1:x=x[:,None]
    n,k=x.shape
    if not n:raise ValueError('empty estimand')
    if groups is None:groups=list(range(n))
    if len(groups)!=n:raise ValueError('group length')
    unique=sorted(set(groups));lookup={v:i for i,v in enumerate(unique)}
    sums=np.zeros((len(unique),k));counts=np.zeros(len(unique))
    for row,g in zip(x,groups):sums[lookup[g]]+=row;counts[lookup[g]]+=1
    rng=np.random.default_rng(seed);samples=[]
    # Chunk to bound memory for full-factorial contribution vectors.
    for start in range(0,repetitions,250):
        indices=rng.integers(len(unique),size=(min(250,repetitions-start),len(unique)))
        samples.append(sums[indices].sum(axis=1)/counts[indices].sum(axis=1)[:,None])
    draws=np.concatenate(samples);mean=x.mean(axis=0);se=draws.std(axis=0,ddof=1)
    pointwise=np.quantile(draws,[alpha/2,1-alpha/2],axis=0).T
    active=se>1e-14
    radius=float(np.quantile(np.max(np.abs((draws[:,active]-mean[active])/se[active]),axis=1),1-alpha)) if active.any() else 0.
    simultaneous=np.column_stack([mean-radius*se,mean+radius*se])
    return {'mean':mean.tolist(),'ci95':pointwise.tolist(),'simultaneous_ci95':simultaneous.tolist(),
            'clusters':len(unique),'puzzles':n,'bootstrap_repetitions':repetitions,'seed':seed,
            'method':'source-game cluster bootstrap; percentile pointwise / max studentized centered-error simultaneous'}

def paired(new,old,groups=None,**kwargs):
    a=np.asarray(new,dtype=int);b=np.asarray(old,dtype=int)
    if a.shape!=b.shape:raise ValueError('unpaired shapes')
    gains=int(((a==1)&(b==0)).sum());losses=int(((a==0)&(b==1)).sum())
    r=cluster_bootstrap(a-b,groups,**kwargs)
    return {'n':len(a),'new_hits':int(a.sum()),'old_hits':int(b.sum()),'gains':gains,'losses':losses,
            'difference':r['mean'][0],'ci95':r['ci95'][0],'p_exact_mcnemar':mcnemar(gains,losses),'clusters':r['clusters']}

def mobius(cells,mask,background=0):
    """Empty-background interaction unless background explicitly supplied."""
    if mask&background:raise ValueError('overlapping interaction/background')
    r=0;subset=mask
    while True:
        r=r+(-1)**(mask.bit_count()-subset.bit_count())*np.asarray(cells[background|subset],float)
        if subset==0:return r
        subset=(subset-1)&mask

def shapley(cells,bits=range(5),background=0):
    bits=list(bits);n=len(bits);out=[]
    for bit in bits:
        others=[x for x in bits if x!=bit];v=0
        for size in range(n):
            weight=math.factorial(size)*math.factorial(n-size-1)/math.factorial(n)
            for subset in itertools.combinations(others,size):
                mask=background|sum(1<<x for x in subset)
                v=v+weight*(np.asarray(cells[mask|1<<bit],float)-np.asarray(cells[mask],float))
        out.append(v)
    return np.stack(out,axis=-1)
