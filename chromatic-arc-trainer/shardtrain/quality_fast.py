"""Experimental ranked evaluation selection with unchanged selected labels.

The default quality parser remains untouched. Only metadata for every analysis
and every PV of attempted candidates are validated here. Unused lower-ranked
analyses are explicitly marked uninspected, never certified as valid.
"""
from __future__ import annotations
import math

from . import quality as base

AUDIT_KEYS=frozenset({'rejected_evaluations','validation_mode','selected_evaluation_index',
    'metadata_checked_evaluations','metadata_rejected_evaluations','full_validation_attempted_indices',
    'full_validation_rejected_evaluations','skipped_lower_evaluation_indices',
    'skipped_lower_evaluations','rejected_evaluations_scope'})


def semantic_record(record):
    """All label/board/provenance fields except explicitly changed audit scope."""
    result=dict(record)
    result['source_eval']={k:v for k,v in record['source_eval'].items() if k not in AUDIT_KEYS}
    return result


def normalize_selected(raw,config=None,calibration=None,source=None):
    config=config or base.QualityConfig()
    # Preserve the base parser's configuration and malformed top-level behavior.
    if (config.min_depth<1 or config.min_knodes<1 or config.max_pvs<1 or config.max_pv_plies<1
        or not math.isfinite(config.policy_temperature_cp) or config.policy_temperature_cp<=0):
        return base.parse_record(raw,calibration=calibration,config=config,source=source)
    if not isinstance(raw,dict):return base.parse_record(raw,calibration=calibration,config=config,source=source)
    entries=raw.get('evals')
    if not isinstance(entries,list) or not entries or len(entries)>256:
        return base.parse_record(raw,calibration=calibration,config=config,source=source)
    candidates=[];metadata_rejected=[]
    for index,entry in enumerate(entries):
        try:
            # Order and exact helper are shared with the base parser: malformed
            # knodes is rejected before low depth, even when depth is too low.
            if not isinstance(entry,dict):raise base.QualityError('invalid_evaluation')
            depth=base._integer(entry.get('depth'),'depth',1)
            knodes=base._integer(entry.get('knodes'),'knodes',1)
            if depth<config.min_depth:raise base.QualityError('low_depth')
            if knodes<config.min_knodes:raise base.QualityError('low_nodes')
            pvs=entry.get('pvs')
            if not isinstance(pvs,list) or not pvs or len(pvs)>config.max_pvs:raise base.QualityError('invalid_pv_count')
            # The historical parser can raise an uncaught numeric overflow for
            # huge JSON integer CP/mate scores. Preserve that record-level
            # behavior even if the analysis would otherwise be skipped. This
            # inspects score convertibility only, not unused PV legality.
            for pv in pvs:
                if isinstance(pv,dict) and ('cp' in pv)!=('mate' in pv):
                    value=pv['cp'] if 'cp' in pv else pv['mate']
                    if isinstance(value,(int,float)) and not isinstance(value,bool):
                        float(value if 'cp' in pv else abs(value))
            candidates.append(((depth,knodes,len(pvs),-index),index,entry))
        except base.QualityError as exc:metadata_rejected.append({'index':index,'reason':exc.reason})
        except Exception:
            # JSON integer overflow/type corner cases can raise outside
            # QualityError. Replay the base parser to retain FEN/error priority.
            return base.parse_record(raw,calibration=calibration,config=config,source=source)
    attempted=[];full_rejected=[]
    for rank,index,entry in sorted(candidates,key=lambda candidate:candidate[0],reverse=True):
        attempted.append(index)
        try:
            selected=base.parse_record({'fen':raw.get('fen'),'evals':[entry]},calibration=calibration,config=config,source=source)
        except base.QualityError as exc:
            full_rejected.append({'index':index,'reason':exc.reason})
            # These failures are independent of the selected evaluation. Avoid
            # repeating an invalid FEN for every analysis in the same record.
            if exc.reason in ['invalid_fen','invalid_fen_fields','illegal_position','terminal_position']:
                return base.parse_record(raw,calibration=calibration,config=config,source=source)
            continue
        skipped=[i for _,i,_ in candidates if i not in attempted]
        known_rejected=sorted(metadata_rejected+full_rejected,key=lambda item:item['index'])
        selected['source_eval'].update(
            rejected_evaluations=[item['reason'] for item in known_rejected],
            validation_mode='selected-ranked-v1',selected_evaluation_index=index,
            metadata_checked_evaluations=len(entries),metadata_rejected_evaluations=metadata_rejected,
            full_validation_attempted_indices=attempted,full_validation_rejected_evaluations=full_rejected,
            skipped_lower_evaluation_indices=skipped,skipped_lower_evaluations=len(skipped),
            rejected_evaluations_scope='metadata checked for all analyses; full PV validation only for attempted ranked candidates; lower skipped analyses are uninspected')
        return selected
    # Preserve the original record rejection reason and order if none survives.
    return base.parse_record(raw,calibration=calibration,config=config,source=source)
