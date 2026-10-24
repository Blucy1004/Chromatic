param([string]$Python = $env:ZERO_PYTHON,[switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    foreach ($item in @('.venv\Scripts\python.exe', '..\..\work\.venv\Scripts\python.exe')) {
        $candidate = Join-Path $PSScriptRoot $item
        if (Test-Path -LiteralPath $candidate) { $Python = (Resolve-Path -LiteralPath $candidate).Path; break }
    }
}
if (-not $Python) { throw 'Pass -Python or set ZERO_PYTHON.' }
$Python = (Resolve-Path -LiteralPath $Python).Path
Push-Location $PSScriptRoot
try {
    if (-not $SkipBuild) {
        & (Join-Path $PSScriptRoot 'build.cmd')
        if ($LASTEXITCODE) { throw 'C++ build failed' }
    }
    ctest --test-dir build --output-on-failure
    if ($LASTEXITCODE) { throw 'Core tests failed' }
    & $Python tests\integration.py --engine build\project_zero.exe --report validation\integration.json
    if ($LASTEXITCODE) { throw 'Integration tests failed' }
    foreach ($item in @('reliability','arena_tests','safety')) {
        & $Python "tests\$item.py" --engine build\project_zero.exe --report "reports\release-$item.json"
        if ($LASTEXITCODE) { throw "$item tests failed" }
    }
} finally { Pop-Location }
