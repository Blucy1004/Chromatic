"""Bounded, resumable JSONL transport; no arbitrary compressed-offset decoding.

HTTP reconnects preserve the live decoder. A fresh process starts at byte zero
and verifies a decompressed-line prefix digest before returning resumed rows.
Zstandard frames are streamed into a bounded disk spool and verified before rows
are exposed. No compressed file or whole decompressed frame is retained in RAM.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Callable, Iterator
import urllib.error
import urllib.parse
import urllib.request

OFFICIAL_EVAL_URL = 'https://database.lichess.org/lichess_db_eval.jsonl.zst'
LICENSE_URL = 'https://database.lichess.org/#evals'
MIB = 1024 * 1024


class StreamError(RuntimeError): pass
class SourceChanged(StreamError): pass
class StreamLimit(StreamError): pass
class DataCorruption(StreamError): pass
class TruncatedSource(DataCorruption): pass
class ResumeMismatch(StreamError): pass


@dataclass(frozen=True)
class SourceSpec:
    location: str
    license: str = 'CC0-1.0'
    original_url: str = OFFICIAL_EVAL_URL
    version: str = 'unspecified; identity pinned by HTTP validators or file stat'
    format: str | None = None


def _identity_equal(expected: dict, actual: dict) -> None:
    for key in ('location', 'final_url', 'content_length', 'etag', 'last_modified', 'mtime_ns', 'prefix_sha256',
                'license', 'original_url', 'version'):
        if key in expected and expected[key] != actual.get(key):
            raise SourceChanged(f'source identity changed: {key}')


def _headers(response) -> dict:
    return {key.lower(): value for key, value in response.headers.items()}


def _request(url: str, *, method='GET', headers=None, timeout=20):
    request = urllib.request.Request(url, method=method, headers={
        'User-Agent': 'CHROMATIC-CC0-ShardTrainer/1.0', 'Accept-Encoding': 'identity', **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)


class _HTTPBytes:
    def __init__(self, spec, *, max_bytes, retries, timeout, expected_identity, allow_loopback, resource_check):
        parsed = urllib.parse.urlparse(spec.location)
        loopback = allow_loopback and parsed.hostname in ('127.0.0.1', 'localhost', '::1')
        if not loopback and (parsed.scheme != 'https' or parsed.hostname != 'database.lichess.org'):
            raise ValueError('network sources must be official HTTPS database.lichess.org URLs')
        self.url=spec.location; self.max_bytes=max_bytes; self.retries=retries; self.timeout=timeout
        self.resource_check=resource_check; self.offset=0; self.response=None; self.errors=0
        self.digest=hashlib.sha256()
        self.stats={'network_bytes':0, 'reconnects':0, 'range_reconnects':0, 'range_ignored':0}
        try:
            with _request(self.url, method='HEAD', timeout=timeout) as response:
                headers=_headers(response); final_url=response.url
        except urllib.error.HTTPError as exc:
            if exc.code not in (403,405,501): raise
            with _request(self.url, headers={'Range':'bytes=0-0'}, timeout=timeout) as response:
                headers=_headers(response); final_url=response.url
                self.stats['network_bytes']+=len(response.read(1))
        length=headers.get('content-length')
        if 'content-range' in headers:
            match=re.fullmatch(r'bytes \d+-\d+/(\d+)',headers['content-range'])
            if match:length=match.group(1)
        self.identity={'location':self.url,'final_url':final_url,
                       'content_length':int(length) if length else None,
                       'etag':headers.get('etag'),'last_modified':headers.get('last-modified'),
                       'accept_ranges_advertised':headers.get('accept-ranges'),
                       'license':spec.license,'original_url':spec.original_url,'version':spec.version}
        if expected_identity:
            _identity_equal(expected_identity,self.identity)
            if not self.identity['etag'] and not self.identity['last_modified']:
                raise SourceChanged('HTTP resume requires ETag or Last-Modified identity')

    def _validate(self, response):
        h=_headers(response)
        for key,header in [('etag','etag'),('last_modified','last-modified')]:
            expected=self.identity.get(key)
            if expected and h.get(header)!=expected:raise SourceChanged(f'HTTP {header} changed')
        if response.url!=self.identity['final_url']:raise SourceChanged('redirect destination changed')
        if h.get('content-encoding','identity') not in ('identity',''):
            raise StreamError('unexpected HTTP Content-Encoding; byte offsets would be ambiguous')
        return h

    def _connect(self):
        headers={}
        if self.offset:
            headers['Range']=f'bytes={self.offset}-'
            validator=self.identity.get('etag')
            if not validator or validator.startswith('W/'):validator=self.identity.get('last_modified')
            if validator:headers['If-Range']=validator
        response=_request(self.url,headers=headers,timeout=self.timeout)
        try:
            h=self._validate(response)
            if self.offset and response.status==206:
                match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',h.get('content-range',''))
                if not match or int(match[1])!=self.offset:raise SourceChanged('invalid resumed Content-Range')
                total=self.identity['content_length']
                if total is not None and int(match[3])!=total:raise SourceChanged('HTTP length changed')
                self.stats['range_reconnects']+=1
            elif self.offset and response.status==200:
                # A server may ignore Range. Discard its byte-zero prefix without
                # feeding it into the existing decoder for a second time.
                self.stats['range_ignored']+=1
                left=self.offset
                while left:
                    self.resource_check(0)
                    data=response.read(min(65536,left))
                    if not data:raise TruncatedSource('server ignored Range but ended before replay offset')
                    self.stats['network_bytes']+=len(data);left-=len(data)
            elif response.status!=200:
                raise StreamError(f'unexpected HTTP status {response.status}')
            length=h.get('content-length')
            if response.status==200 and length and self.identity['content_length'] is not None and int(length)!=self.identity['content_length']:
                raise SourceChanged('HTTP content length changed between HEAD and GET')
            self.response=response
        except BaseException:
            response.close();raise

    def read(self,size):
        if size<0:raise ValueError('unbounded transport reads are forbidden')
        if not size:return b''
        total=self.identity['content_length']
        if total is not None and self.offset==total:return b''
        if self.offset>=self.max_bytes:raise StreamLimit('compressed byte budget reached')
        size=min(size,self.max_bytes-self.offset)
        while True:
            self.resource_check(0)
            try:
                if self.response is None:self._connect()
                data=self.response.read(size)
                if not data:
                    if total is None or self.offset==total:return b''
                    raise ConnectionError('HTTP body ended before advertised Content-Length')
                self.offset+=len(data);self.stats['network_bytes']+=len(data)
                self.digest.update(data);self.stats['compressed_prefix_sha256']=self.digest.hexdigest()
                return data
            except (SourceChanged,StreamLimit,DataCorruption):raise
            except (OSError,urllib.error.URLError,http.client.HTTPException,ConnectionError) as exc:
                if isinstance(exc,http.client.IncompleteRead):self.stats['network_bytes']+=len(exc.partial)
                if self.response is not None:self.response.close();self.response=None
                # IncompleteRead.partial was not returned to the decoder, so the
                # reconnect starts at the previous confirmed compressed offset.
                self.errors+=1;self.stats['reconnects']+=1
                if self.errors>self.retries:raise StreamError(f'network retry budget exhausted: {exc}') from exc
                time.sleep(min(.1*2**(self.errors-1),1.0))

    def close(self):
        if self.response is not None:self.response.close();self.response=None


class _FileBytes:
    def __init__(self,spec,*,max_bytes,expected_identity,resource_check):
        self.path=Path(spec.location).resolve();self.file=self.path.open('rb');stat=self.path.stat()
        prefix=self.file.read(65536);self.file.seek(0)
        self.identity={'location':str(self.path),'content_length':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                       'prefix_sha256':hashlib.sha256(prefix).hexdigest(),'license':spec.license,
                       'original_url':spec.original_url,'version':spec.version}
        if expected_identity:
            try:_identity_equal(expected_identity,self.identity)
            except BaseException:self.file.close();raise
        self.max_bytes=max_bytes;self.offset=0;self.resource_check=resource_check
        self.digest=hashlib.sha256()
        self.stats={'network_bytes':0,'reconnects':0,'range_reconnects':0,'range_ignored':0}
    def read(self,size):
        if size<0:raise ValueError('unbounded transport reads are forbidden')
        if self.offset==self.identity['content_length']:return b''
        if self.offset>=self.max_bytes:raise StreamLimit('compressed/input byte budget reached')
        self.resource_check(0);data=self.file.read(min(size,self.max_bytes-self.offset));self.offset+=len(data)
        self.digest.update(data);self.stats['compressed_prefix_sha256']=self.digest.hexdigest()
        return data
    def close(self):self.file.close()


def _exact(source,count,*,empty_ok=False):
    out=bytearray()
    while len(out)<count:
        part=source.read(count-len(out))
        if not part:
            if empty_ok and not out:return None
            raise TruncatedSource('incomplete Zstandard frame/header/block at source EOF')
        out.extend(part)
    return bytes(out)


class _FrameInput(io.RawIOBase):
    """One structural Zstandard/skippable frame, without reading past its end.

    The framing follows RFC 8878: 3-byte blocks, <=128KiB block payload,
    optional 4-byte checksum. The zstd decoder verifies compressed contents.
    """
    def __init__(self,source,magic):
        self.source=source;self.pending=bytearray(magic);self.remaining=0
        self.last=False;self.checksum=False;self.done=False;self.stage='block'
        value=int.from_bytes(magic,'little')
        self.skippable=0x184D2A50<=value<=0x184D2A5F
        if self.skippable:
            size=_exact(source,4);self.pending.extend(size);self.remaining=int.from_bytes(size,'little');self.stage='skip'
        elif value==0xFD2FB528:
            descriptor=_exact(source,1);self.pending.extend(descriptor);flag=descriptor[0]
            if flag&8:raise DataCorruption('reserved Zstandard frame bit is set')
            single=bool(flag&32);self.checksum=bool(flag&4)
            fcs=(1 if single else 0,2,4,8)[flag>>6]
            rest=(0 if single else 1)+(0,1,2,4)[flag&3]+fcs
            self.pending.extend(_exact(source,rest))
        else:raise DataCorruption(f'invalid Zstandard frame magic: {magic.hex()}')
        self.header=bytes(self.pending)
    def readable(self):return True
    def read(self,size=-1):
        if size<0:raise ValueError('unbounded frame reads are forbidden')
        if not size:return b''
        while True:
            if self.pending:
                out=bytes(self.pending[:size]);del self.pending[:len(out)];return out
            if self.done:return b''
            if self.stage in ('skip','payload'):
                if self.remaining:
                    out=_exact(self.source,min(size,self.remaining));self.remaining-=len(out);return out
                if self.stage=='skip':self.done=True;continue
                self.stage='checksum' if self.last and self.checksum else 'done' if self.last else 'block'
            if self.stage=='block':
                header=_exact(self.source,3);bits=int.from_bytes(header,'little')
                self.last=bool(bits&1);kind=(bits>>1)&3;length=bits>>3
                if kind==3 or length>131072:raise DataCorruption('invalid Zstandard block header')
                self.pending.extend(header);self.remaining=1 if kind==1 else length;self.stage='payload';continue
            if self.stage=='checksum':self.pending.extend(_exact(self.source,4));self.stage='done';continue
            if self.stage=='done':self.done=True
    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)


class RecordStream:
    """Context-managed sequential input with bounded verified-frame disk spool.

    ``resource_check(extra_disk_bytes)`` is called before reads and spool writes.
    ``start_line`` is the number of already consumed physical JSONL lines.
    Prefix replay is mandatory on a new process; compressed offsets in cursors
    are diagnostics, never decoder restart points.
    """
    def __init__(self,source: str | Path | SourceSpec,*,spool_directory=None,
                 max_compressed_bytes=256*MIB,max_decompressed_bytes=2*1024*MIB,
                 max_frame_bytes=128*MIB,max_line_bytes=MIB,max_window_kib=128*1024,
                 retries=4,timeout=20,expected_identity=None,resource_check:Callable[[int],None]|None=None,
                 allow_loopback=False,allow_partial_frames=False,source_version=None,source_format=None):
        self.spec=source if isinstance(source,SourceSpec) else SourceSpec(str(source),version=source_version or 'unspecified',format=source_format)
        if self.spec.license!='CC0-1.0':raise ValueError('only explicitly CC0-1.0 sources are supported')
        if min(max_compressed_bytes,max_decompressed_bytes,max_frame_bytes,max_line_bytes,max_window_kib)<1:
            raise ValueError('stream budgets must be positive')
        self.resource_check=resource_check or (lambda extra:None)
        self.format=self.spec.format or ('zst' if self.spec.location.lower().endswith(('.zst','.zstd')) else 'jsonl')
        if self.format not in ('zst','jsonl'):raise ValueError('format must be zst or jsonl')
        self.spool_directory=Path(spool_directory) if spool_directory else None
        if self.spool_directory:self.spool_directory.mkdir(parents=True,exist_ok=True)
        parsed=urllib.parse.urlparse(self.spec.location)
        if parsed.scheme in ('http','https'):
            self.source=_HTTPBytes(self.spec,max_bytes=max_compressed_bytes,retries=retries,timeout=timeout,
                expected_identity=expected_identity,allow_loopback=allow_loopback,resource_check=self.resource_check)
        else:self.source=_FileBytes(self.spec,max_bytes=max_compressed_bytes,expected_identity=expected_identity,resource_check=self.resource_check)
        self.identity=dict(self.source.identity)
        self.max_decompressed_bytes=max_decompressed_bytes;self.max_frame_bytes=max_frame_bytes
        self.max_line_bytes=max_line_bytes;self.max_window_kib=max_window_kib
        self.allow_partial_frames=allow_partial_frames
        self.stats={'compressed_bytes':0,'decompressed_bytes':0,'lines':0,'verified_frames':0,'skippable_frames':0,
                    'verified_compressed_offset':0,'partial_frames':0,'complete_source':False,
                    'verified_frames_with_checksum':0,'verified_frames_without_checksum':0,'first_verified_frame_parameters':[],
                    'integrity':'verified_complete_zstd_frames' if self.format=='zst' else 'plain_prefix_sha256'}
        self.cursor={'line':0,'prefix_sha256':hashlib.sha256().hexdigest(),'source_identity':self.identity,
                     'compressed_offset_diagnostic':0}
        self.closed=False

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        if not self.closed:self.source.close();self.closed=True

    def _chunks(self):
        if self.format=='jsonl':
            while True:
                data=self.source.read(65536)
                if not data:self.stats['complete_source']=True;return
                self.stats['decompressed_bytes']+=len(data)
                if self.stats['decompressed_bytes']>self.max_decompressed_bytes:raise StreamLimit('decompressed byte budget reached')
                yield data
            return
        import zstandard
        while True:
            magic=_exact(self.source,4,empty_ok=True)
            if magic is None:self.stats['complete_source']=True;return
            frame=_FrameInput(self.source,magic)
            if frame.skippable:
                while frame.read(65536):pass
                self.stats['skippable_frames']+=1;continue
            parameters=zstandard.get_frame_parameters(frame.header)
            if parameters.window_size>self.max_window_kib*1024:
                raise StreamLimit('Zstandard window exceeds explicit memory budget')
            fd,path=tempfile.mkstemp(prefix='.source-frame-',suffix='.jsonl.tmp',dir=self.spool_directory)
            partial=False;frame_size=0
            try:
                with os.fdopen(fd,'wb') as spool:
                    try:
                        # Explicit header validation above is authoritative.
                        # backend_c 0.25 uses bytes here despite older KiB docs.
                        with zstandard.ZstdDecompressor(max_window_size=self.max_window_kib*1024).stream_reader(
                                frame,read_size=65536,read_across_frames=False,closefd=False) as decoder:
                            while True:
                                data=decoder.read(65536)
                                if not data:break
                                if frame_size+len(data)>self.max_frame_bytes:raise StreamLimit('verified-frame spool byte budget reached')
                                if self.stats['decompressed_bytes']+len(data)>self.max_decompressed_bytes:raise StreamLimit('decompressed byte budget reached')
                                self.resource_check(len(data));spool.write(data);frame_size+=len(data)
                                self.stats['decompressed_bytes']+=len(data)
                        if not frame.done:
                            # The parser may not have been asked for its final
                            # empty read after checksum consumption.
                            if frame.read(1):raise DataCorruption('decoder ended before frame boundary')
                    except (StreamLimit,TruncatedSource):
                        if not self.allow_partial_frames:raise
                        partial=True;self.stats['partial_frames']+=1
                        self.stats['integrity']='partial_frame_checksum_unavailable'
                    except zstandard.ZstdError as exc:raise DataCorruption('Zstandard decode/checksum failure: '+str(exc)) from exc
                    spool.flush();os.fsync(spool.fileno())
                if not partial:
                    self.stats['verified_frames']+=1;self.stats['verified_compressed_offset']=self.source.offset
                    checksum_key='verified_frames_with_checksum' if parameters.has_checksum else 'verified_frames_without_checksum'
                    self.stats[checksum_key]+=1
                    if not parameters.has_checksum:
                        self.stats['integrity']='complete_frames_without_full_checksum'
                    if len(self.stats['first_verified_frame_parameters'])<4:
                        self.stats['first_verified_frame_parameters'].append({'window_bytes':parameters.window_size,
                            'content_size':parameters.content_size,'has_checksum':parameters.has_checksum,
                            'decoded_bytes':frame_size,'compressed_end':self.source.offset})
                with open(path,'rb') as spool:
                    while data:=spool.read(65536):yield data
                if partial:return
            finally:
                try:os.unlink(path)
                except FileNotFoundError:pass

    def iter_lines(self,start_line=0,*,expected_prefix_sha256=None) -> Iterator[tuple[int,str]]:
        if start_line<0:raise ValueError('start_line must be nonnegative')
        if start_line and not expected_prefix_sha256:raise ResumeMismatch('resuming requires the saved decompressed prefix SHA-256')
        digest=hashlib.sha256();line=0;pending=bytearray();verified_resume=start_line==0
        if start_line==0 and expected_prefix_sha256 and digest.hexdigest()!=expected_prefix_sha256:raise ResumeMismatch('empty prefix digest mismatch')
        def emit(raw):
            nonlocal line,verified_resume
            line+=1;digest.update(raw)
            self.stats['lines']=line;self.stats['compressed_bytes']=self.source.offset
            self.stats.update(self.source.stats)
            self.cursor={'line':line,'prefix_sha256':digest.hexdigest(),'last_line_sha256':hashlib.sha256(raw).hexdigest(),
                         'source_identity':self.identity,'compressed_offset_diagnostic':self.source.offset,
                         'integrity':self.stats['integrity'],'verified_frames':self.stats['verified_frames']}
            if line==start_line:
                if expected_prefix_sha256 and digest.hexdigest()!=expected_prefix_sha256:raise ResumeMismatch('decompressed prefix digest changed')
                verified_resume=True
            if line<=start_line:return None
            try:return (line,raw.decode('utf-8',errors='strict').rstrip('\r\n'))
            except UnicodeDecodeError as exc:raise DataCorruption(f'invalid UTF-8 at JSONL line {line}') from exc
        try:
            for data in self._chunks():
                pending.extend(data)
                while True:
                    end=pending.find(b'\n')
                    if end<0:break
                    if end+1>self.max_line_bytes:raise StreamLimit('JSONL line byte budget reached')
                    raw=bytes(pending[:end+1]);del pending[:end+1]
                    value=emit(raw)
                    if value is not None:yield value
                if len(pending)>self.max_line_bytes:raise StreamLimit('JSONL line byte budget reached')
            if pending and self.stats['complete_source']:
                value=emit(bytes(pending))
                if value is not None:yield value
            elif pending:self.stats['incomplete_tail_bytes_discarded']=len(pending)
            if not verified_resume:raise ResumeMismatch('source ended before saved line cursor')
        finally:
            self.stats['compressed_bytes']=self.source.offset;self.stats.update(self.source.stats)

    def iter_records(self,start_line=0,*,expected_prefix_sha256=None,on_invalid=None):
        for line,text in self.iter_lines(start_line,expected_prefix_sha256=expected_prefix_sha256):
            try:
                row=json.loads(text)
                if not isinstance(row,dict):raise ValueError('JSONL row must be an object')
            except (ValueError,TypeError) as exc:
                if on_invalid is not None:on_invalid(line,str(exc));continue
                raise DataCorruption(f'invalid JSONL at line {line}: {exc}') from exc
            yield line,row


def iter_records(source,start_line=0,**kwargs):
    prefix=kwargs.pop('expected_prefix_sha256',None)
    with RecordStream(source,**kwargs) as stream:
        yield from stream.iter_records(start_line,expected_prefix_sha256=prefix)
