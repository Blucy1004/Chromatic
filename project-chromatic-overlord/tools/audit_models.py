#!/usr/bin/env python3
"""Read trusted repository checkpoints; bind their exported tensors to .zero files."""
import argparse,hashlib,json,struct,tempfile
from pathlib import Path
import torch
from experiment import ROOT,sha,save
REPO=ROOT.parent

def json_file(path):return json.loads((REPO/path).read_text())
def export_digest(state):
 width=state['config']['width'];blocks=state['config']['blocks'];h=hashlib.sha256()
 h.update(b'ZERO0001');h.update(struct.pack('<4I',1280,width,blocks,4672))
 names=['input',*[f'residual.{i}' for i in range(blocks)],'policy','wdl']
 for layer in names:
  for suffix in ['weight','bias']:
   tensor=state['model'][f'{layer}.{suffix}'].detach().cpu()
   if not bool(torch.isfinite(tensor).all()):raise ValueError('Nonfinite checkpoint')
   h.update(tensor.numpy().astype('<f4').tobytes())
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('Audit already exists')
 artifacts={}
 specs={
  'zero_initial':('chromatic-shard-trainer/champion/model.pt','chromatic-shard-trainer/champion/default.zero'),
  'lichess_10000':('chromatic-shard-trainer/candidates/positions-10000.pt','chromatic-shard-trainer/candidates/positions-10000.zero')}
 for name,(pt,zero) in specs.items():
  checkpoint=torch.load(REPO/pt,map_location='cpu',weights_only=False)
  data=(REPO/zero).read_bytes();dims=struct.unpack('<4I',data[8:24]);exported=export_digest(checkpoint)
  if exported!=sha(REPO/zero):raise AssertionError('PT tensors do not match deployed .zero')
  artifacts[name]={'checkpoint_path':str((REPO/pt).resolve()),'checkpoint_sha256':sha(REPO/pt),
    'model_path':str((REPO/zero).resolve()),'model_sha256':sha(REPO/zero),'exported_checkpoint_tensor_sha256':exported,
    'checkpoint_export_matches_model':True,'architecture':{'format':data[:8].decode(),'inputs':dims[0],'width':dims[1],'residual_blocks':dims[2],'policy_actions':dims[3],'wdl_outputs':3,'parameters':(len(data)-24)//4},
    'checkpoint_format':checkpoint['format'],'step':checkpoint['step'],'seed':checkpoint.get('seed'),
    'dataset':checkpoint['dataset'],'settings':checkpoint['settings'],
    'progress':checkpoint.get('progress'),'initial_model_sha256':checkpoint.get('initial_model_sha256')}
 teacher=REPO/'project-chromatic/runs/bootstrap/teacher.jsonl';teacher_hash=sha(teacher);teacher_meta=json_file('project-chromatic/runs/bootstrap/teacher.meta.json')
 teacher_rows=sum(bool(line.strip()) for line in teacher.read_text().splitlines())
 if teacher_hash!=artifacts['zero_initial']['dataset'][0]['sha256'] or teacher_rows!=128:raise AssertionError('Bootstrap corpus binding failed')
 artifacts['zero_initial'].update(training_positions=128,corpus_sha256=teacher_hash,teacher=teacher_meta,
   training_report=json_file('project-chromatic/runs/bootstrap/model.train.json'))
 candidate=artifacts['lichess_10000'];report=json_file('chromatic-shard-validation/positions-10000/candidate/training-report.json')
 if candidate['initial_model_sha256']!=artifacts['zero_initial']['checkpoint_sha256']:raise AssertionError('Initial checkpoint hash mismatch')
 if report['positions']!=10000 or candidate['progress']['positions_processed']!=10000 or report['global_step']!=candidate['step']:raise AssertionError('10k training report/checkpoint mismatch')
 source=REPO/'chromatic-shard-trainer/fixtures/official-evaluations.jsonl.zst'
 if candidate['settings']['source_provenance']['local_sha256']!=sha(source):raise AssertionError('Training fixture hash mismatch')
 candidate.update(training_positions=10000,training_report=report,source_file_sha256=sha(source))
 manifest=REPO/'chromatic-shard-validation/positions-10000/candidate/checkpoints'/report['candidate_manifest'].replace('\\','/').split('/')[-1]
 bundle=json.loads(manifest.read_text());bindings={}
 for name,entry in bundle['files'].items():
  path=manifest.parent/entry['path'].replace('\\','/').split('/')[-1];actual=sha(path)
  if actual!=entry['sha256']:raise AssertionError('Retained bundle hash mismatch')
  bindings[name]={'path':str(path.resolve()),'sha256':actual,'manifest_sha256_matches':True}
 candidate['retained_bundle']={'manifest_path':str(manifest.resolve()),'manifest_sha256':sha(manifest),'files':bindings,'step':bundle['step'],'commit_seq':bundle['commit_seq']}
 historical=json_file('project-chromatic-overlord/reports/puzzles-final.json')
 if historical['model_sha256']!=artifacts['zero_initial']['model_sha256']:raise AssertionError('Historical model was not initial champion')
 historical_rows=[]
 for row in historical['summary']:
  if row['ms']==100:
   historical_rows.append({**row,'model_file_path_source_inferred':str(REPO/'project-chromatic-memory/dist/default.zero'),
     'recorded_model_file_sha256':historical['model_sha256'],'recorded_engine_sha256':historical['engine_sha256'],
     'actual_loaded_tensor_sha256_at_historical_run':None,
     'memory_attestation_status':'not recorded; cannot retroactively attest historical process memory'})
 aliases=[str(p.relative_to(REPO)) for p in REPO.rglob('*.zero') if '.git' not in p.parts and sha(p)==artifacts['zero_initial']['model_sha256']]
 artifact_paths=[p for p in REPO.rglob('*') if p.is_file() and p.suffix in ['.pt','.zero','.exe'] and '.git' not in p.parts and 'project-chromatic-overlord' not in p.parts]
 save(a.out,{'artifacts':artifacts,'initial_champion_identical_aliases':aliases,'historical_100ms_experiments':historical_rows,
   'historical_fixture_sha256':historical['fixture_sha256'],
   'protected_artifact_sha256':{str(p.relative_to(REPO)):sha(p) for p in artifact_paths},
   'limits':['Model .pt/.zero and retained training logs/corpus hashes agree. Training was not re-executed by this audit.',
    'Stockfish teacher identity/hash is historical metadata; availability of that exact teacher binary is not established.',
    'Historical in-process loaded tensor hashes were not recorded; new matrix runs attest them explicitly.']})
 print('Checkpoint exports, bootstrap corpus, 10k ancestry and fixture hash verified')
if __name__=='__main__':main()
