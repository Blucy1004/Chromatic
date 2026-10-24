"""Bounded release experiments. No external dataset or teacher invocation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zero.verify import verify
from zero.train import legacy_train
from zero.loop import loop
from zero.arena import arena
from zero.data import atomic_json
from bench import benchmark
from compare_search import compare
from tactics import tactics
from draw_audit import audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',help='new reproduction directory');args=p.parse_args()
    root=Path(args.root).resolve();engine=root/'build/project_zero.exe';bench=root/'build/chromatic_bench.exe'
    model=root/'baseline/default.zero';reports=(root/args.output).resolve() if args.output else root/'reports/final'
    if args.output and reports.exists() and any(reports.iterdir()):raise ValueError('choose an empty reproduction directory')
    reports.mkdir(parents=True,exist_ok=True)
    cycle=reports/'selfplay-cycle' if args.output else root/'runs/overdrive'
    compare(root/'baseline/project_zero.exe',engine,model,reports/'scalar-search-equivalence.json')
    for name,executable,backend,head in [('baseline',root/'build/baseline_bench.exe',None,None),
            ('full-scalar',bench,'scalar','full'),('legal-scalar',bench,'scalar','legal'),('legal-simd',bench,'simd','legal')]:
        benchmark(executable,model,reports/(name+'.json'),repeats=5,backend=backend,head=head)
    benchmark(root/'build/baseline_bench.exe',model,reports/'baseline-memory.json',repeats=3,evaluations=100,simulations=4096)
    benchmark(bench,model,reports/'chromatic-memory.json',repeats=3,evaluations=100,simulations=4096,backend='scalar')
    for name,checkpoint in [('baseline',root/'runs/bootstrap/model.pt'),
                            ('tiny',root/'reports/model-scaling/tiny/model.pt'),('small',root/'reports/model-scaling/small/model.pt')]:
        for backend in ('Scalar','SIMD'):
            verify(engine,checkpoint,reports/f'{name}-{backend.lower()}-parity.json',positions=100,backend=backend)
    for name,options in [('default',{}),('deep-proof',{'MateProofNodes':12000,'MateProofPlies':5}),
                          ('fpu-025',{'FPUReduction':.25})]:
        tactics(engine,model,root/'benchmarks/tactics.json',reports/(name+'-tactics.json'),options=options)
    state=loop(engine,root/'runs/bootstrap/model.pt',cycle,iterations=1,games=4,simulations=64,
               steps=100,batch_size=32,pairs=2,max_plies=512,seed=2026,engine_options={'InferenceBackend':'SIMD'},
               opening_plies=6,arena_milliseconds=10,opening_suite=root/'benchmarks/openings.json')
    again=loop(engine,root/'runs/bootstrap/model.pt',cycle,iterations=1,games=4,simulations=64,
               steps=100,batch_size=32,pairs=2,max_plies=512,seed=2026,engine_options={'InferenceBackend':'SIMD'},
               opening_plies=6,arena_milliseconds=10,opening_suite=root/'benchmarks/openings.json')
    if state!=again:raise AssertionError('completed generation changed on restart')
    arena(engine,model,model,reports/'engine-match.json',pairs=4,incumbent_executable=root/'baseline/project_zero.exe',
          milliseconds=10,max_plies=256,opening_suite=root/'benchmarks/openings.json')
    arena(engine,root/'reports/model-scaling/small/model.zero',root/'reports/model-scaling/tiny/model.zero',
          reports/'model-match.json',pairs=2,milliseconds=10,max_plies=256,opening_suite=root/'benchmarks/openings.json')
    old_shards=sorted((root/'reports/baseline-games').glob('game_*.jsonl'))
    audit(old_shards,[root/'runs/bootstrap/model.pt',root/'runs/release-validation/generation_0000/candidate.pt'],reports/'draw-audit.json')
    legacy_output=reports/'compatibility/chromatic-legacy-resumed.pt' if args.output else root/'reports/compatibility/chromatic-legacy-resumed.pt'
    legacy_train([root/'runs/bootstrap/teacher.jsonl'],legacy_output,
                 steps=201,batch_size=32,seed=2026,save_every=50,resume=root/'runs/bootstrap/model.pt')
    atomic_json(reports/'execution.json',{'completed':True,'completed_generation_restart_equal':True,
                'selfplay_games':4,'training_steps':100,'engine_match_games':8,'model_match_games':4,
                'maximum_selfplay_plies_per_game':512,'maximum_match_plies_per_game':256,
                'teacher_used_after_bootstrap':False,'external_dataset_downloaded':False})
    print(json.dumps({'experiments_completed':True,'report_directory':str(reports)},indent=2))
