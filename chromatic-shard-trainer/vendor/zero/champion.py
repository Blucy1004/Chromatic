import json
from pathlib import Path
from .checkpoint import digest
from .data import atomic_json

def rollback(output,steps=1):
    root=Path(output);path=root/'state.json';state=json.loads(path.read_text(encoding='utf-8'))
    history=state.get('history',[])
    if steps<1 or steps>len(history):raise ValueError('not enough verified promotion history to roll back')
    target=history[-steps]
    checkpoint=root/target['previous_champion']
    if not checkpoint.resolve().is_relative_to(root.resolve()):raise ValueError('champion path escaped run directory')
    if digest(checkpoint)!=target['previous_sha256']:raise ValueError('rollback checkpoint checksum mismatch')
    state.setdefault('rollbacks',[]).append({'from':state['champion'],'to':target['previous_champion'],'promotions':steps})
    state['champion']=target['previous_champion'];state['champion_sha256']=target['previous_sha256']
    state['history']=history[:-steps];atomic_json(path,state);return state
