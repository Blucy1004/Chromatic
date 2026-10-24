param([Parameter(ValueFromRemainingArguments=$true)][string[]]$ZeroArguments)
$ErrorActionPreference = 'Stop'
$candidatePaths = @(
    (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    (Join-Path $PSScriptRoot '..\..\work\.venv\Scripts\python.exe')
)
if ($env:ZERO_PYTHON) { $candidatePaths = @($env:ZERO_PYTHON) + $candidatePaths }
$zeroPython = $candidatePaths | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $zeroPython) { throw 'Create .venv and install requirements.txt, or set ZERO_PYTHON to Python.' }
$zeroPython = (Resolve-Path -LiteralPath $zeroPython).Path
Push-Location $PSScriptRoot
try {
    & $zeroPython -m zero @ZeroArguments
    $zeroExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $zeroExitCode
