$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
function Get-PackageSha256([string] $Path) {
    $digest = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        return ([System.BitConverter]::ToString($digest.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $stream.Dispose()
        $digest.Dispose()
    }
}
$packageRoot = $PSScriptRoot
$manifest = Get-Content -LiteralPath (Join-Path $packageRoot 'MANIFEST.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($entry in $manifest.files.PSObject.Properties) {
    $path = Join-Path $packageRoot $entry.Name
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $($entry.Name)" }
    $actual = Get-PackageSha256 $path
    if ($actual -ne $entry.Value) { throw "SHA-256 mismatch: $($entry.Name)" }
}
$exe = Join-Path $packageRoot 'chromatic.exe'
$start = New-Object System.Diagnostics.ProcessStartInfo
$start.FileName = $exe
$start.Arguments = '--model "' + (Join-Path $packageRoot 'default.zero') + '"'
$start.WorkingDirectory = $packageRoot
$start.UseShellExecute = $false
$start.RedirectStandardInput = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$start.StandardOutputEncoding = New-Object System.Text.UTF8Encoding($false)
$start.StandardErrorEncoding = New-Object System.Text.UTF8Encoding($false)
$process = New-Object System.Diagnostics.Process
$process.StartInfo = $start
try {
    [void]$process.Start()
    # Write ASCII command bytes directly, without a PowerShell pipeline BOM.
    $commands = [System.Text.Encoding]::ASCII.GetBytes("zero model`nquit`n")
    $process.StandardInput.BaseStream.Write($commands, 0, $commands.Length)
    $process.StandardInput.BaseStream.Flush()
    if (!$process.WaitForExit(5000)) { $process.Kill(); throw 'Engine response timeout' }
    $output = $process.StandardOutput.ReadToEnd()
    $errors = $process.StandardError.ReadToEnd()
    if ($process.ExitCode -ne 0) { throw "Engine launch failed: $errors" }
    $identity = $output.Trim() | ConvertFrom-Json
    if (!$identity.loaded -or $identity.sha256 -ne $manifest.model.sha256) { throw "Bundled model identity mismatch: $output $errors" }
}
finally { $process.Dispose() }
Write-Host "PASS: file hashes and loaded champion $($identity.sha256)"
