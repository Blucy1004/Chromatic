$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$packageRoot = $PSScriptRoot
$manifest = Get-Content -LiteralPath (Join-Path $packageRoot 'MANIFEST.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($entry in $manifest.files.PSObject.Properties) {
    $path = Join-Path $packageRoot $entry.Name
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $($entry.Name)" }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.Value) { throw "SHA-256 mismatch: $($entry.Name)" }
}
$exe = Join-Path $packageRoot 'chromatic.exe'
$output = @('zero model', 'quit') | & $exe
if ($LASTEXITCODE -ne 0) { throw 'Engine launch failed' }
$identity = $output | Where-Object { $_.StartsWith('{') } | Select-Object -First 1 | ConvertFrom-Json
if (!$identity.loaded -or $identity.sha256 -ne $manifest.model.sha256) { throw 'Bundled model was not loaded' }
Write-Host "PASS: file hashes and loaded champion $($identity.sha256)"
