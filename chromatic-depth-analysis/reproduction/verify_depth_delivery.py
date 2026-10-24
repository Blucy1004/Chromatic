"""Bounded release/telemetry/UCI validation. Run after the measurement batch.

This script does not rebuild, change models, or modify either engine. All engine
searches are sequential and pinned to the same logical CPU as the depth batch.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'outputs/project-chromatic'
PROJECT = ROOT / 'outputs/project-chromatic-depth'
OUT = ROOT / 'outputs/chromatic-depth-analysis'
RAW = OUT / 'validation-uci-raw'
sys.path.insert(0, str(ORIGINAL))
sys.dont_write_bytecode = True
import chess
from zero.native import Native

MODEL = ORIGINAL / 'dist/default.zero'
NEW_EXE = PROJECT / 'build/chromatic_depth.exe'
ORIGINAL_EXE = ORIGINAL / 'dist/chromatic.exe'
FLOAT_TOLERANCE = 1e-8
AFFINITY_MASK = 2
CASES = {
    'start': ([], 4096),
    'before_Rg1_100ms': (['c2c3', 'd7d6', 'a2a3', 'g8f6', 'g1f3', 'b7b6', 'e2e4', 'e7e6'], 6363),
    'before_Rg1_1s': (['c2c3', 'g8f6', 'a2a3', 'd7d6', 'g1f3', 'e7e6'], 4096),
    'before_Ra6_10s': (['g2g4', 'a7a5', 'c2c3'], 4096),
}


def save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def board_for(moves):
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return board


class LoggedNative(Native):
    """Retain commands and every consumed response using the existing client."""
    def __init__(self, executable, tag):
        self.protocol = []
        self.tag = tag
        self.began = time.perf_counter()
        super().__init__(executable, MODEL, timeout=20)

    def record(self, direction, line):
        self.protocol.append({'direction': direction, 'elapsed_ms': (time.perf_counter()-self.began)*1000,
                              'line': line})

    def send(self, text):
        self.record('stdin', text)
        return super().send(text)

    def until(self, predicate, timeout=None, errors=None):
        def traced(line):
            self.record('stdout', line)
            return predicate(line)
        return super().until(traced, timeout=timeout, errors=errors)

    def close(self):
        try:
            super().close()
        finally:
            RAW.mkdir(parents=True, exist_ok=True)
            with (RAW / (self.tag+'.protocol.jsonl')).open('w', encoding='utf-8') as stream:
                for row in self.protocol:
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')


def integrity():
    manifest = json.loads((ORIGINAL/'release-manifest.json').read_text(encoding='utf-8'))
    changed = []
    for row in manifest['files']:
        path = ORIGINAL / row['path']
        if not path.is_file() or digest(path) != row['sha256']:
            changed.append(row['path'])
    check(not changed, 'production manifest changed: '+repr(changed))
    baseline = []
    for path in sorted((PROJECT/'baseline').rglob('*')):
        if path.is_file():
            relative = path.relative_to(PROJECT/'baseline')
            original = ORIGINAL / relative
            row = {'path': relative.as_posix(), 'baseline_sha256': digest(path),
                   'production_sha256': digest(original)}
            check(row['baseline_sha256'] == row['production_sha256'], 'baseline source differs: '+str(relative))
            baseline.append(row)
    check(len(baseline) == 10, 'expected 10 preserved baseline files')
    copied_model = PROJECT/'dist/default.zero'
    check(digest(MODEL) == digest(copied_model), 'copied depth model changed')
    return {'production_manifest_files_checked': len(manifest['files']),
            'changed_or_missing_production_files': changed, 'baseline_files': baseline,
            'production_engine_sha256': digest(ORIGINAL_EXE), 'new_engine_sha256': digest(NEW_EXE),
            'production_model_sha256': digest(MODEL), 'depth_model_sha256': digest(copied_model),
            'baseline_all_exactly_equal': True}


def configure(engine, enabled=None):
    options = {'MateProofNodes': 0, 'TreeMemoryMB': 64, 'MaxNodes': 1000000,
               'InferenceBackend': 'SIMD', 'SparsePolicyHead': True,
               'CPuct': '1.5', 'FPUReduction': '0'}
    if enabled is not None:
        options.update(SearchTelemetry=enabled, TelemetryInterval=4096)
    engine.pin_cpu(AFFINITY_MASK)
    engine.configure(options)


def histogram_summary(histogram):
    n = sum(histogram)
    weighted = sum(depth*visits for depth, visits in enumerate(histogram))
    occupied = [depth for depth, visits in enumerate(histogram) if visits]
    if not n:
        return {'samples': 0, 'sum_depth': 0, 'mean_depth': 0, 'median_depth': 0, 'max_depth': 0}
    lower, upper = (n-1)//2, n//2
    before, left, right = 0, None, None
    for depth, visits in enumerate(histogram):
        if left is None and before+visits > lower:
            left = depth
        if before+visits > upper:
            right = depth
            break
        before += visits
    return {'samples': n, 'sum_depth': weighted, 'mean_depth': weighted/n,
            'median_depth': (left+right)/2, 'max_depth': max(occupied)}


def check_summary(actual, histogram):
    expected = histogram_summary(histogram)
    for key, value in expected.items():
        check(math.isclose(actual[key], value, rel_tol=1e-10, abs_tol=1e-9), 'histogram summary mismatch: '+key)


def check_telemetry(t, simulations, result=None):
    check(t['schema_version'] == 1, 'unexpected telemetry schema')
    check(not t['proof_only'], 'bounded diagnostic unexpectedly became proof-only')
    h = t['leaf_depth']
    check(sum(h) == simulations and h[0] == 0, 'one non-root endpoint per simulation')
    check_summary(t['depth'], h)
    tail = 0
    for depth in range(len(h)-1, -1, -1):
        tail += h[depth]
        check(t['traversal_depth'][depth] == tail, 'traversal histogram tail mismatch')
        check(t['pre_failure_leaf_depth'][depth]+t['post_failure_leaf_depth'][depth] == h[depth], 'phase histogram mismatch')
        check(sum(root['leaf_depth'][depth] for root in t['roots']) == h[depth], 'root histogram partition mismatch')
    for root in t['roots']:
        check(sum(root['leaf_depth']) == root['visits'], 'root endpoint count differs from visits')
    check(sum(root['visits'] for root in t['roots']) == simulations, 'telemetry root visit total')
    c = t['counters']
    fails = c['child_fail_node_limit']+c['child_fail_byte_capacity']+c['expansion_fail_byte_capacity']
    event = t['first_allocation_failure']
    check(event['occurred'] == bool(fails), 'failure event/counter mismatch')
    if event['occurred']:
        check(sum(t['pre_failure_leaf_depth']) == event['simulation']-1, 'first failure phase boundary')
    else:
        check(t['pre_failure_leaf_depth'] == h and sum(t['post_failure_leaf_depth']) == 0, 'unexpected post-failure samples')
    previous = [0]*len(h)
    previous_ms = 0
    for snapshot in t['snapshots']:
        check_summary(snapshot['depth'], snapshot['leaf_depth'])
        check_summary(snapshot['window_depth'], snapshot['window_leaf_depth'])
        check(snapshot['simulation'] == sum(snapshot['leaf_depth']), 'snapshot not after completed simulation')
        check(snapshot['elapsed_ms'] >= previous_ms, 'snapshot clock order')
        check(all(old+window == current for old, window, current in zip(previous, snapshot['window_leaf_depth'], snapshot['leaf_depth'])), 'snapshot window partition')
        check(sum(root['visits'] for root in snapshot['roots']) == snapshot['simulation'], 'snapshot root visits')
        previous = snapshot['leaf_depth']
        previous_ms = snapshot['elapsed_ms']
    check(previous == h, 'final snapshot does not include all endpoints')
    if result is not None:
        check(c['node_creations']+1 == result['arena_nodes'], 'actual new node count')
        check(c['neural_evaluations'] == result['neural_evals'], 'NN evaluation count')
        check(t['depth']['max_depth'] == result['max_depth'], 'legacy max depth mismatch')
        lookup = {row['uci']: row for row in result['visits']}
        for root in t['roots']:
            row = lookup[root['uci']]
            check(row['n'] == root['visits'], 'root result visit mismatch')
            check(abs(row['q']-root['q']) <= FLOAT_TOLERANCE and abs(row['prior']-root['prior']) <= FLOAT_TOLERANCE,
                  'root result Q/policy mismatch')
    return {'samples': simulations, 'max_depth': t['depth']['max_depth'],
            'mean_depth': t['depth']['mean_depth'], 'median_depth': t['depth']['median_depth'],
            'snapshots': len(t['snapshots']), 'all_histogram_and_counter_checks_passed': True}


def same_result(a, b):
    exact_fields = ['bestmove', 'terminal', 'simulations', 'neural_evals', 'memory_bytes',
                    'max_depth', 'proof_nodes', 'mate_proven', 'proof_plies']
    if 'arena_nodes' in a and 'arena_nodes' in b:
        exact_fields.append('arena_nodes')
    for key in exact_fields:
        check(a[key] == b[key], 'result differs: '+key)
    maximum_error = abs(a['value']-b['value'])
    check(maximum_error <= FLOAT_TOLERANCE, 'root value differs')
    left, right = a['visits'], b['visits']
    check(len(left) == len(right), 'root legal move count differs')
    for x, y in zip(left, right):
        for key in ['uci', 'action', 'n']:
            check(x[key] == y[key], 'root '+key+' differs')
        for key in ['q', 'prior']:
            difference = abs(x[key]-y[key])
            maximum_error = max(maximum_error, difference)
            check(difference <= FLOAT_TOLERANCE, 'root '+key+' exceeds serialization tolerance')
    return {'passed': True, 'numeric_absolute_tolerance': FLOAT_TOLERANCE,
            'max_serialized_float_difference': maximum_error}


def consume_bestmove(engine, board, *, start):
    info, telemetry = {}, None
    first_info_ms = None
    telemetry_received_ms = None
    lines = []

    def response(line):
        nonlocal telemetry, first_info_ms, telemetry_received_ms
        received_ms = (time.perf_counter()-start)*1000
        lines.append(line)
        if line.startswith('info string telemetry '):
            telemetry_received_ms = received_ms
            telemetry = json.loads(line[len('info string telemetry '):])
        elif line.startswith('info '):
            if first_info_ms is None:
                first_info_ms = received_ms
            for field in ['nodes', 'time', 'nps']:
                match = re.search(r'\b'+field+r' (\d+)', line)
                if match:
                    info[field] = int(match[1])
        return line.startswith('bestmove ')

    errors = []
    line = engine.until(response, timeout=10, errors=errors)
    check(not errors, 'UCI error: '+repr(errors))
    bestmove_ms = (time.perf_counter()-start)*1000
    bestmove = line.split()[1]
    check(chess.Move.from_uci(bestmove) in board.legal_moves, 'UCI returned an illegal bestmove')
    if telemetry is not None:
        check_telemetry(telemetry, info['nodes'])
    return {'bestmove': bestmove, 'info': info, 'elapsed_to_info_ms': first_info_ms,
            'elapsed_to_telemetry_ms': telemetry_received_ms, 'elapsed_to_bestmove_ms': bestmove_ms,
            'telemetry_received': telemetry is not None,
            'protocol_sequence': ['telemetry' if row.startswith('info string telemetry ') else row for row in lines],
            'telemetry_check': {'passed': True, 'samples': telemetry['depth']['samples']} if telemetry else None}


def uci_search(engine, board, command, expected_telemetry):
    engine.position(board)
    engine.ready()
    start = time.perf_counter()
    engine.send(command)
    result = consume_bestmove(engine, board, start=start)
    check(result['telemetry_received'] == expected_telemetry, 'UCI telemetry enable/disable mismatch')
    if expected_telemetry:
        sequence = result['protocol_sequence']
        check(sequence[-2] == 'telemetry' and sequence[-1].startswith('bestmove '), 'telemetry JSON must precede bestmove')
    return result


def main():
    OUT.mkdir(exist_ok=True)
    RAW.mkdir(exist_ok=True)
    validation = {'status': 'running', 'scope': 'sequential bounded validation; no training/build/deployment change',
                  'settings': {'seed_for_zero_search': 2026, 'noise': 0, 'temperature': 0, 'proof_nodes': 0,
                               'max_nodes': 1000000, 'memory_mib': 64, 'threads': 1, 'affinity_mask': AFFINITY_MASK,
                               'backend': 'SIMD', 'sparse_policy': True,
                               'uci_go_default_seed': 1, 'uci_go_seed_note': 'go has no seed option; noise=0/temperature=0 so no RNG is consumed'},
                  'cases': {}, 'raw_protocol_directory': RAW.relative_to(OUT).as_posix()}
    try:
        validation['integrity_before'] = integrity()
        for name, (moves, n) in CASES.items():
            board = board_for(moves)
            results = {}
            for variant, exe, enabled in [('production', ORIGINAL_EXE, None), ('new_off', NEW_EXE, False), ('new_on', NEW_EXE, True)]:
                with LoggedNative(exe, name+'-'+variant) as engine:
                    configure(engine, enabled)
                    engine.position(board)
                    results[variant] = engine.search(n, seed=2026, noise=0, temperature=0)
            baseline_vs_off = same_result(results['production'], results['new_off'])
            off_vs_on = same_result(results['new_off'], results['new_on'])
            check(results['new_off']['telemetry_enabled'] is False and 'telemetry' not in results['new_off'], 'off payload unexpectedly contains telemetry')
            on = results['new_on']
            check(on['telemetry_enabled'], 'on payload lacks enabled flag')
            telemetry_check = check_telemetry(on['telemetry'], n, on)
            check(on['selected']['uci'] == on['bestmove'], 'selected move serialization mismatch')
            ordered = sorted(on['visits'], key=lambda row: (-row['n'], -row['prior']))[:5]
            check(on['top5'] == ordered, 'top5 serialization mismatch')
            validation['cases'][name] = {'fen': board.fen(en_passant='fen'), 'moves': moves, 'simulations': n,
                                         'bestmove': on['bestmove'], 'production_vs_new_off': baseline_vs_off,
                                         'new_off_vs_new_on': off_vs_on, 'telemetry': telemetry_check,
                                         'selected': on['selected'], 'top5': on['top5']}
            save(OUT/'validation-uci.json', validation)

        with LoggedNative(NEW_EXE, 'new-uci-protocol') as engine:
            handshake = [row['line'] for row in engine.protocol if row['direction'] == 'stdout']
            check('option name SearchTelemetry type check default false' in handshake, 'missing/default-on SearchTelemetry option')
            check('option name TelemetryInterval type spin default 4096 min 64 max 1000000' in handshake, 'missing TelemetryInterval option')
            configure(engine, False)
            board = chess.Board()
            validation['uci_handshake'] = {'search_telemetry_default_false': True, 'telemetry_interval_default': 4096,
                                            'engine_name': engine.name, 'options': sorted(engine.options)}
            validation['go_nodes_off'] = uci_search(engine, board, 'go nodes 4096', False)
            engine.configure({'SearchTelemetry': True})
            validation['go_nodes_on'] = uci_search(engine, board, 'go nodes 4096', True)
            check(validation['go_nodes_on']['info']['nodes'] == 4096, 'go nodes did not consume exact requested simulations')
            check(validation['go_nodes_off']['bestmove'] == validation['go_nodes_on']['bestmove'], 'go nodes off/on selected differently')

            # Output costs are measured separately: info precedes JSON serialization,
            # while bestmove receipt includes telemetry formatting and pipe delivery.
            timings = []
            for repeat in range(1, 4):
                for enabled in ([False, True] if repeat % 2 else [True, False]):
                    engine.configure({'SearchTelemetry': enabled})
                    result = uci_search(engine, board, 'go movetime 100', enabled)
                    timings.append({'repeat': repeat, 'enabled': enabled, **result})
            validation['uci_100ms_response_runs'] = timings
            validation['uci_100ms_response_summary'] = []
            for enabled in [False, True]:
                group = [row for row in timings if row['enabled'] == enabled]
                validation['uci_100ms_response_summary'].append({
                    'enabled': enabled, 'repeats': len(group),
                    'median_info_time_ms': statistics.median(row['info']['time'] for row in group),
                    'median_elapsed_to_info_ms': statistics.median(row['elapsed_to_info_ms'] for row in group),
                    'median_elapsed_to_bestmove_ms': statistics.median(row['elapsed_to_bestmove_ms'] for row in group),
                    'min_elapsed_to_bestmove_ms': min(row['elapsed_to_bestmove_ms'] for row in group),
                    'max_elapsed_to_bestmove_ms': max(row['elapsed_to_bestmove_ms'] for row in group),
                    'median_after_info_delivery_ms': statistics.median(row['elapsed_to_bestmove_ms']-row['elapsed_to_info_ms'] for row in group),
                    'note': 'response check only; native off/on overhead is measured in the separate paired fixed-N batch'})

            engine.configure({'SearchTelemetry': True})
            engine.position(board)
            engine.ready()
            start = time.perf_counter()
            engine.send('go infinite')
            time.sleep(.05)
            stop_sent = time.perf_counter()
            engine.send('stop')
            stopped = consume_bestmove(engine, board, start=start)
            stopped['stop_response_ms'] = (time.perf_counter()-stop_sent)*1000
            check(stopped['telemetry_received'] and stopped['info']['nodes'] > 0, 'infinite/stop did not return accounted search')
            engine.ready()
            stopped['subsequent_readyok'] = True
            validation['infinite_stop'] = stopped
            validation['after_stop_go_nodes'] = uci_search(engine, board, 'go nodes 128', True)
            check(validation['after_stop_go_nodes']['info']['nodes'] == 128, 'engine state did not recover after stop')
        validation['status'] = 'passed'
    except BaseException as exc:
        validation['status'] = 'failed'
        validation['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        validation['integrity_after'] = integrity()
        check(validation['integrity_before'] == validation['integrity_after'], 'input engines/models changed during validation')
        save(OUT/'validation-uci.json', validation)
    print(json.dumps({'status': validation['status'], 'cases': len(validation['cases']),
                      'production_manifest_files_unchanged': validation['integrity_after']['production_manifest_files_checked'],
                      'baseline_files_unchanged': len(validation['integrity_after']['baseline_files']),
                      'uci_100ms_response_summary': validation.get('uci_100ms_response_summary'),
                      'output': str(OUT/'validation-uci.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
