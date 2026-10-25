# Windows x64 Release package

The package uses MSVC /MT, SSE2, an embedded UTF-8 process manifest, the unchanged
128-position champion, and unchanged experimental OFF defaults. It is an
experimental UCI distribution, not a champion/model promotion. Windows 10 1903+
or Windows 11 x64 is the intended desktop platform; native validation runs on
Windows Server 2022 x64. No Python or VC++ Redistributable is needed to run it.

`make_package.py prepare` assembles files and exact corresponding source.
`validate_package.py` executes the extracted ZIP from unrelated working
directories and paths containing spaces, an apostrophe and Korean characters.
`make_package.py finalize` adds passing native evidence and hashes, then writes
the final ZIP. The CI job also runs all 11 native CTests and audits PE imports,
x64 architecture and the actual embedded UTF-8 manifest.

The original model, deployments and experiment results are never rewritten.
Packaging output is placed in a new `packages/windows-x64/<run>/` directory.
