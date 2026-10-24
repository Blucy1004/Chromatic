# Streaming and recovery contract

`shardtrain.streaming.RecordStream` reads JSONL or JSONL/Zstandard sequentially.
It does not load a compressed file or a full decompressed frame into RAM.
Each completed Zstandard frame is decompressed into a temporary disk spool with
64KiB output chunks. Its framing and decoder checksum are verified before that
frame's records are exposed. Temporary spool writes call the resource guard.
The spool is deleted on normal completion, errors and generator close. A process
kill can leave an orphan `.source-frame-*.jsonl.tmp`; run-owner cleanup must only
remove its own orphan files after acquiring the run lock.

## Official object probe

On 2026-10-24, bounded HTTP checks found:

- URL: `https://database.lichess.org/lichess_db_eval.jsonl.zst`, no redirect.
- HEAD status 200, length 22,423,777,007 bytes, ETag `"6ac01121-5388faeef"`.
- Last-Modified: Fri, 02 Oct 2026 20:16:33 GMT. The download page separately
  identifies its export update as 2026-10-05; both values are retained.
- Ranges 0–1023 and 1024–2047 returned 206 with matching Content-Range.
- The first skippable frame stores a four-byte integer 5,684,148. At compressed
  offset 5,684,160, another skippable frame and standard Zstandard magic were
  found. These are observations, not a general assumption about all exports.
- The first standard frame has an 8MiB window, unknown declared content size,
  and a checksum. A bounded decoder/spool quota is still necessary.

Only small header/range probes were made by this module's implementation audit.
The full 22.4GB file is not required to process completed prefix frames.
`reports/http-probe.json` and `reports/frame-probe.json` preserve the responses.

## Recovery semantics

Within one process, a network disconnect reconnects at the exact compressed
offset while preserving the same live frame decoder. Range responses must start
at the requested byte and match the original object identity. If a server ignores
Range and returns 200, its already-consumed compressed prefix is discarded
without feeding duplicate bytes into the decoder. Retry count is bounded.

A new process always starts decompression at byte zero. It replays physical
JSONL lines up to the saved line number and checks their rolling SHA-256 prefix.
Resuming at a positive line number without that saved prefix digest is rejected.
The saved compressed offset is explicitly diagnostic and is never interpreted
as an independent frame start. This is correct but can reread an increasingly
large prefix after a late interruption; no claim of constant-time process resume
is made. HTTP ETag/Last-Modified/length and local file stat/header fingerprints
are checked against the saved identity before replay.

An official server advertising Range does not mean arbitrary middle bytes form
a valid independent compressed stream. Any future indexed resume optimization
must use verified frame boundaries and preserve cross-frame JSONL line state.

## Limits and incomplete input

Limits apply independently to compressed bytes, total decompressed bytes,
decompressed frame-spool bytes, line length and declared Zstandard window.
Reads are bounded; frame parsing handles skippable frames and does not read into
the next standard frame. JSONL lines can cross frame boundaries.

`allow_partial_frames=False` is the production default. An incomplete frame or
quota cutoff fails without exposing that frame's records. The explicit partial
option exposes only complete newline-terminated rows already produced and marks
`integrity=partial_frame_checksum_unavailable`; it cannot establish a verified
full frame and must not satisfy a promotion gate. Invalid compressed contents or
checksum mismatch still fail. A completed prefix frame does not establish
integrity of the unseen remainder of the full database.

The Python backend's window parameter is not trusted alone: frame-header
`window_size` is checked directly in bytes. This guards an observed backend_c
0.25 behavior where the `max_window_size` argument uses byte-sized limits despite
older documentation describing KiB.

## Interface

```python
from shardtrain.streaming import RecordStream, SourceSpec

source = SourceSpec("data/evals.jsonl.zst", license="CC0-1.0",
                    original_url="https://database.lichess.org/lichess_db_eval.jsonl.zst",
                    version="pinned export identity")
with RecordStream(source, spool_directory="data/spool",
                  max_compressed_bytes=64*1024**2,
                  max_decompressed_bytes=256*1024**2,
                  resource_check=lambda additional_disk_bytes: None,
                  expected_identity=None) as stream:
    for number, row in stream.iter_records(start_line=0):
        # Commit stream.cursor alongside a durable shard/checkpoint transaction.
        # stream.cursor includes line, prefix_sha256, source_identity and integrity.
        pass
```

`iter_lines` exposes raw UTF-8 lines so the caller can quarantine bad JSON.
`iter_records` is strict by default or accepts an `on_invalid(line,error)` callback.
The caller declares local-file provenance; a filename alone is not proof of CC0.

## References

- [Official Lichess evaluation export and CC0 declaration](https://database.lichess.org/#evals).
- [Zstandard frame format, RFC 8878](https://www.rfc-editor.org/rfc/rfc8878.html).
- [python-zstandard decompression interfaces](https://python-zstandard.readthedocs.io/en/latest/decompressor.html).
