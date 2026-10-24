import copy
import hashlib
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

import zstandard
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shardtrain.streaming import (RecordStream,SourceSpec,SourceChanged,StreamError,StreamLimit,
                                 DataCorruption,TruncatedSource,ResumeMismatch)


def compressed(data):return zstandard.ZstdCompressor(write_checksum=True).compress(data)
def rows(count=20):return b''.join(json.dumps({'i':i,'text':'abcdefgh'*5}).encode()+b'\n' for i in range(count))


class FixtureServer:
    def __init__(self,payload,*,ignore_range=False,changed_etag=False,bad_range=False,always_cut=False):
        self.payload=payload;self.ignore_range=ignore_range;self.changed_etag=changed_etag
        self.bad_range=bad_range;self.always_cut=always_cut;self.gets=0;self.range_requests=[]
        owner=self
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*args):pass
            def do_HEAD(self):
                self.send_response(200);self.send_header('Content-Length',str(len(owner.payload)))
                self.send_header('ETag','"fixture-v1"');self.send_header('Last-Modified','Fri, 02 Oct 2026 20:16:33 GMT')
                self.send_header('Accept-Ranges','bytes');self.end_headers()
            def do_GET(self):
                owner.gets+=1;offset=0;status=200;range_header=self.headers.get('Range')
                if range_header:
                    owner.range_requests.append(range_header)
                    if not owner.ignore_range:offset=int(range_header.split('=')[1].split('-')[0]);status=206
                self.send_response(status)
                if status==206:
                    start=offset+1 if owner.bad_range else offset
                    self.send_header('Content-Range',f'bytes {start}-{len(owner.payload)-1}/{len(owner.payload)}')
                self.send_header('Content-Length',str(len(owner.payload)-offset))
                self.send_header('ETag','"fixture-v2"' if owner.changed_etag and owner.gets>1 else '"fixture-v1"')
                self.send_header('Last-Modified','Fri, 02 Oct 2026 20:16:33 GMT');self.end_headers()
                body=owner.payload[offset:]
                cut=owner.always_cut or owner.gets==1
                if cut:body=body[:max(1,len(body)//3)]
                try:self.wfile.write(body);self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):pass
                if cut:
                    self.close_connection=True
                    try:self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.url=f'http://127.0.0.1:{self.server.server_port}/fixture.jsonl.zst'
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)


class StreamingTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/'reports/.transport-test-work'
        base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def write(self,name,data):p=self.root/name;p.write_bytes(data);return p
    def stream(self,source,**kwargs):return RecordStream(source,spool_directory=self.root/'spool',**kwargs)

    def test_plain_cursor_replay_and_digest_guard(self):
        path=self.write('small.jsonl',rows())
        with self.stream(path) as stream:
            iterator=stream.iter_records()
            prefix=[next(iterator) for _ in range(7)];cursor=copy.deepcopy(stream.cursor);identity=stream.identity
            iterator.close()
        with self.stream(path,expected_identity=identity) as stream:
            remainder=list(stream.iter_records(cursor['line'],expected_prefix_sha256=cursor['prefix_sha256']))
        self.assertEqual([x[1]['i'] for x in prefix+remainder],list(range(20)))
        with self.stream(path) as stream:
            with self.assertRaises(ResumeMismatch):list(stream.iter_records(7,expected_prefix_sha256='0'*64))
        path.write_bytes(rows()+b'{}\n')
        with self.assertRaises(SourceChanged):self.stream(path,expected_identity=identity)

    def test_multiframe_skippable_and_cross_frame_line(self):
        data=rows(30);split=data.index(b'abcdefgh')+7
        skip=struct.pack('<II',0x184D2A50,4)+struct.pack('<I',999)
        path=self.write('multi.jsonl.zst',skip+compressed(data[:split])+skip+compressed(data[split:]))
        with self.stream(path) as stream:
            actual=list(stream.iter_records());stats=dict(stream.stats)
        self.assertEqual([r['i'] for _,r in actual],list(range(30)))
        self.assertEqual(stats['verified_frames'],2);self.assertEqual(stats['skippable_frames'],2)
        self.assertEqual(stats['verified_frames_with_checksum'],2);self.assertEqual(stats['verified_frames_without_checksum'],0)
        self.assertTrue(stats['complete_source']);self.assertFalse(list((self.root/'spool').glob('*')))

    def test_checksum_absence_is_reported_and_frame_metadata_is_bounded(self):
        frame=zstandard.ZstdCompressor(write_checksum=False).compress(rows(2))
        path=self.write('no-checksum.zst',frame*6+compressed(rows(2)))
        with self.stream(path) as stream:
            self.assertEqual(len(list(stream.iter_records())),14);stats=stream.stats
        self.assertEqual(stats['verified_frames_with_checksum'],1)
        self.assertEqual(stats['verified_frames_without_checksum'],6)
        self.assertEqual(stats['integrity'],'complete_frames_without_full_checksum')
        self.assertEqual(stream.cursor['integrity'],'complete_frames_without_full_checksum')
        self.assertEqual(len(stats['first_verified_frame_parameters']),4)
        self.assertTrue(all(not p['has_checksum'] for p in stats['first_verified_frame_parameters']))

    def test_checksum_corruption_never_exposes_frame_rows(self):
        data=bytearray(compressed(rows(50)));data[-1]^=1
        path=self.write('corrupt.zst',data);seen=[]
        with self.stream(path) as stream:
            with self.assertRaises(DataCorruption):
                for row in stream.iter_records():seen.append(row)
        self.assertEqual(seen,[]);self.assertFalse(list((self.root/'spool').glob('*')))

    def test_truncated_frame_rejected(self):
        path=self.write('short.zst',compressed(rows())[:-3])
        with self.stream(path) as stream:
            with self.assertRaises(TruncatedSource):list(stream.iter_records())

    def test_partial_frame_requires_explicit_option_and_marks_integrity(self):
        payload=compressed(rows(4000));path=self.write('prefix.zst',payload)
        with self.stream(path,max_compressed_bytes=len(payload)-8,allow_partial_frames=True) as stream:
            actual=list(stream.iter_records());stats=dict(stream.stats)
        self.assertGreater(len(actual),0);self.assertLess(len(actual),4000)
        self.assertEqual([r['i'] for _,r in actual],list(range(len(actual))))
        self.assertEqual(stats['verified_frames'],0)
        self.assertEqual(stats['integrity'],'partial_frame_checksum_unavailable')

    def test_unknown_content_size_window_guard(self):
        compressor=zstandard.ZstdCompressor(write_checksum=True).compressobj()
        payload=compressor.compress(rows(4000))+compressor.flush()
        path=self.write('unknown.zst',payload)
        with self.stream(path,max_window_kib=4) as stream:
            with self.assertRaises(StreamLimit):list(stream.iter_records())
        with self.stream(path,max_window_kib=8192) as stream:
            self.assertEqual(len(list(stream.iter_records())),4000)

    def test_limits_line_frame_decompressed_and_compressed(self):
        path=self.write('many.zst',compressed(rows(100)))
        for options in [{'max_compressed_bytes':20},{'max_decompressed_bytes':100},
                        {'max_frame_bytes':100},{'max_line_bytes':10}]:
            with self.subTest(options=options),self.stream(path,**options) as stream:
                with self.assertRaises(StreamLimit):list(stream.iter_records())

    def test_resource_guard_prevents_spool_growth_and_cleans_up(self):
        path=self.write('many.zst',compressed(rows(100)))
        def quota(extra):
            if extra:raise StreamLimit('disk quota fixture')
        with self.stream(path,resource_check=quota) as stream:
            with self.assertRaises(StreamLimit):list(stream.iter_records())
        self.assertFalse(list((self.root/'spool').glob('*')))

    def test_malformed_json_quarantine_and_final_line(self):
        path=self.write('bad.jsonl',b'{"ok":1}\nnot json\n{"ok":2}')
        invalid=[]
        with self.stream(path) as stream:
            actual=list(stream.iter_records(on_invalid=lambda line,error:invalid.append((line,error))))
        self.assertEqual([r['ok'] for _,r in actual],[1,2]);self.assertEqual(invalid[0][0],2)
        with self.stream(path) as stream:
            with self.assertRaises(DataCorruption):list(stream.iter_records())

    def test_resume_beyond_eof_rejected(self):
        path=self.write('small.zst',compressed(rows()))
        with self.stream(path) as stream:
            with self.assertRaises(ResumeMismatch):list(stream.iter_records(100,expected_prefix_sha256='0'*64))
        with self.stream(path) as stream:
            with self.assertRaises(ResumeMismatch):list(stream.iter_records(1))

    def test_http_range_reconnect_preserves_decoder(self):
        with FixtureServer(compressed(rows(100))) as server:
            with self.stream(server.url,allow_loopback=True,retries=3,timeout=2) as stream:
                actual=list(stream.iter_records());stats=dict(stream.stats)
            self.assertEqual([r['i'] for _,r in actual],list(range(100)))
            self.assertTrue(server.range_requests);self.assertGreater(stats['range_reconnects'],0)

    def test_http_ignored_range_replays_compressed_prefix_once(self):
        with FixtureServer(compressed(rows(100)),ignore_range=True) as server:
            with self.stream(server.url,allow_loopback=True,retries=3,timeout=2) as stream:
                actual=list(stream.iter_records());stats=dict(stream.stats)
            self.assertEqual(len(actual),100);self.assertGreater(stats['range_ignored'],0)

    def test_http_identity_change_and_wrong_content_range_abort(self):
        for options in [{'changed_etag':True},{'bad_range':True}]:
            with self.subTest(options=options),FixtureServer(compressed(rows(100)),**options) as server:
                with self.stream(server.url,allow_loopback=True,retries=3,timeout=2) as stream:
                    with self.assertRaises(SourceChanged):list(stream.iter_records())

    def test_network_retry_budget_is_bounded(self):
        with FixtureServer(compressed(rows(100)),always_cut=True) as server:
            with self.stream(server.url,allow_loopback=True,retries=1,timeout=2) as stream:
                with self.assertRaises(StreamError):list(stream.iter_records())
            self.assertLessEqual(server.gets,2)

    def test_nonofficial_or_non_cc0_network_source_rejected(self):
        with self.assertRaises(ValueError):self.stream('https://example.com/fixture.zst')
        with self.assertRaises(ValueError):self.stream(SourceSpec('missing.jsonl',license='unknown'))


if __name__=='__main__':unittest.main()
