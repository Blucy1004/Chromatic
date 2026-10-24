"""Check actual native SIMD sparse-policy inference against PyTorch, read only."""
from pathlib import Path
import argparse
from contextlib import closing
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from shardtrain.evaluate import (_connect,_meta,_load_model,checkpoint,Native,torch,np,features,action,
                                read_manifest,verify_bundle,OPTIONS)
from shardtrain.model_adapter import restore_board
from shardtrain.storage import atomic_json,digest


def check(validation_root,engine,output,positions=10,tolerance=2e-5):
    if not 1<=positions<=100 or not 0<tolerance<=2e-5:raise ValueError('bounded positions and strict absolute tolerance required')
    started=time.perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    base=Path(validation_root).resolve();engine=Path(engine).resolve();output=Path(output).resolve()
    with closing(_connect(base/'positions-10000/data')) as db:
        records=[(identity,json.loads(payload)) for identity,payload in db.execute(
            "SELECT id,payload FROM records WHERE split='test' ORDER BY seq LIMIT ?",(positions,))]
    if len(records)!=positions:raise ValueError('insufficient fixed test positions')
    boards=[restore_board(record) for _,record in records]
    models={};source_files={'engine':engine,'champion_checkpoint':ROOT/'champion/model.pt',
                             'champion_native':ROOT/'champion/default.zero'}
    champion,actual=checkpoint.load(source_files['champion_checkpoint'])
    models['champion']=(champion,source_files['champion_native'],str(actual),False)
    for size in [100,1000,10000]:
        candidate=base/f'positions-{size}'/'candidate'
        with closing(_connect(base/f'positions-{size}'/'data')) as db:
            head=Path(_meta(db,'current_manifest') or '').resolve()
            if not head.is_relative_to(candidate/'checkpoints'):raise ValueError('unexpected candidate manifest location')
            manifest=read_manifest(head);state,mirror=verify_bundle(head)
        native=Path(manifest['files']['model']['path']);models[str(size)]=(state,native,str(head),mirror)
        source_files[str(size)+'_manifest']=head
        for name,spec in manifest['files'].items():
            path=Path(spec['path'])
            if path.is_file():source_files[str(size)+'_'+name]=path
    before={key:digest(path) for key,path in source_files.items()};results={}
    for name,(state,native_path,checkpoint_path,mirror) in models.items():
        model=_load_model(state);maximum={'features':0.,'legal_policy_logits':0.,'wdl_probability':0.};cases=[]
        with Native(engine,native_path,timeout=20) as native:
            native.pin_cpu(2);native.configure(OPTIONS)
            if 'Threads' in native.options:native.configure({'Threads':1})
            for (identity,record),board in zip(records,boards):
                native.position(board);actual=native.command('zero evallegal')
                expected_legal={move.uci():action(move) for move in board.legal_moves}
                if {item['uci']:item['action'] for item in actual['legal']}!=expected_legal:
                    raise AssertionError(name+' native legal moves/action encoding mismatch')
                inputs=features(board)
                with torch.no_grad():
                    policy,wdl=model(torch.from_numpy(inputs[None,:]))
                    expected_policy=policy[0].numpy();expected_wdl=torch.softmax(wdl,-1)[0].numpy()
                indices=list(expected_legal.values())
                if not indices:raise ValueError('fixed parity test position has no legal moves')
                errors={'features':float(np.max(np.abs(np.asarray(actual['features'])-inputs))),
                        'legal_policy_logits':float(np.max(np.abs(np.asarray(actual['policy'])[indices]-expected_policy[indices]))),
                        'wdl_probability':float(np.max(np.abs(np.asarray(actual['wdl'])-expected_wdl)))}
                if any(not np.isfinite(value) for value in errors.values()):raise AssertionError('nonfinite parity error')
                if errors['features']>1e-7 or max(errors['legal_policy_logits'],errors['wdl_probability'])>tolerance:
                    raise AssertionError(name+' native/PyTorch absolute tolerance exceeded: '+str(errors))
                for key,error in errors.items():maximum[key]=max(maximum[key],error)
                cases.append({'record_id':identity,'root_fen':record['root_fen'],'moves':record['moves'],
                              'fen':board.fen(en_passant='fen'),'legal_actions':len(indices),'max_abs_error':errors})
        results[name]={'passed':True,'checkpoint':checkpoint_path,'native_model':str(native_path),
                       'used_mirror':mirror,'step':state['step'],'config':state['config'],
                       'max_abs_error':maximum,'cases':cases}
    after={key:digest(path) for key,path in source_files.items()}
    if before!=after:raise RuntimeError('model, manifest, checkpoint, or engine changed during inference parity check')
    report={'format':'chromatic-native-inference-parity-v1','passed':True,'cases':positions*len(models),
            'positions_per_model':positions,'source':'10k run frozen test split ordered by record sequence',
            'absolute_tolerance':{'features':1e-7,'legal_policy_logits':tolerance,'wdl_probability':tolerance},
            'backend':'SIMD','sparse_policy':True,'search_threads':1,'affinity_mask':2,
            'models':results,'preservation':{'before':before,'after':after,'unchanged':True},
            'arithmetic':'FP32 native SIMD and PyTorch CPU may differ in reduction order; comparisons use absolute error',
            'elapsed_seconds':time.perf_counter()-started}
    atomic_json(output,report);return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-root',type=Path,default=ROOT.parent/'chromatic-shard-validation')
    parser.add_argument('--engine',type=Path,default=ROOT.parent/'project-chromatic-memory/dist/chromatic-memory.exe')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/inference-parity.json')
    parser.add_argument('--positions',type=int,default=10);parser.add_argument('--tolerance',type=float,default=2e-5)
    report=check(**vars(parser.parse_args(argv)))
    print(json.dumps({'passed':report['passed'],'cases':report['cases'],'elapsed_seconds':report['elapsed_seconds'],
                      'max_abs_error':{name:model['max_abs_error'] for name,model in report['models'].items()}},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
