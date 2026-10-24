"""Strict one-time migration of a separately copied, verified CPU shard run."""
import hashlib
from pathlib import Path
from .storage import canonical,training_settings

EXECUTION={'device':'cpu','resolved_device':'cpu','cpu_threads':1,'precision':'fp32','prefetch':0,
           'cache_mode':'auto','transfer_mode':'copy','optimizer':'default','gradient_accumulation':1,'fallback':'error'}

def authorize(marker,config,previous_signature,state=None):
    if not isinstance(marker,dict) or marker.get('format')!='chromatic-legacy-import-v1':
        raise ValueError('No verified legacy import marker; use a new run or the import tool')
    if marker.get('status') not in ['pending_arc_migration','migrated']:
        raise ValueError('Legacy import did not finish; incomplete copies cannot resume')
    if marker.get('copy_complete') is not True:raise ValueError('Legacy copy has not passed its final verification')
    legacy=marker['legacy_settings'];old_signature=marker['legacy_settings_signature']
    if hashlib.sha256(canonical(legacy).encode()).hexdigest()!=old_signature:
        raise ValueError('Legacy settings hash does not match import evidence')
    for key,value in EXECUTION.items():
        if config.get(key)!=value:raise ValueError('Legacy import first resumes as CPU FP32 with original execution semantics: '+key)
    for key in ['data_directory','checkpoint_directory']:
        if str(Path(config[key]).resolve())!=str(Path(marker['target_'+key]).resolve()):
            raise ValueError('Legacy import destination differs: '+key)
    for key,value in legacy.items():
        expected=marker['target_'+key] if key in ['data_directory','checkpoint_directory'] else value
        if key=='trainer_version':
            if value!=1 or config.get(key)!=2:raise ValueError('Only the verified trainer v1 CPU contract can migrate to ARC v2')
        elif key not in config or config[key]!=expected:
            raise ValueError('Legacy import changed a training or source setting: '+key)
    if state is not None:
        if state.get('settings_signature')!=old_signature or state.get('settings')!=legacy:
            raise ValueError('Legacy model checkpoint disagrees with copied ledger settings')
        if state.get('rng_xpu') is not None:raise ValueError('Legacy CPU import unexpectedly contains XPU RNG')
        if state['step']>marker['imported_step']:raise ValueError('Legacy checkpoint step exceeds verified import evidence')
    if previous_signature!=old_signature:
        # A prior process may have committed the new settings signature just
        # before interruption, while the old immutable model head still exists.
        current=training_settings(config)
        if previous_signature!=hashlib.sha256(canonical(current).encode()).hexdigest():
            raise ValueError('Neither original nor authorized ARC signature matches')
    return True
