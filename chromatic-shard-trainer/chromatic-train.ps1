[CmdletBinding()]
param(
    [int]$Positions=100000,
    [int]$ShardSize=10000,
    [int]$Epochs=3,
    [int]$BatchSize=32,
    [double]$LearningRate=0.001,
    [string]$DataDirectory='',
    [string]$CheckpointDirectory='',
    [switch]$Resume,
    [switch]$DryRun,
    [switch]$KeepShards,
    [double]$MaxDiskGB=4,
    [double]$MaxMemoryGB=4,
    [int]$Seed=2026,
    [string]$Source='https://database.lichess.org/lichess_db_eval.jsonl.zst',
    [string]$Calibration='',
    [string]$InitialCheckpoint='',
    [int]$MaxDownloadMB=128,
    [int]$CheckpointEvery=32,
    [int]$ReplaySize=512,
    [string]$Python=''
)
$ErrorActionPreference='Stop'
$trainerRoot=[System.IO.Path]::GetDirectoryName($MyInvocation.MyCommand.Path)
if (-not $DataDirectory) { $DataDirectory=Join-Path $trainerRoot 'data' }
if (-not $CheckpointDirectory) { $CheckpointDirectory=Join-Path $trainerRoot 'runs\default' }
if (-not $Calibration) { $Calibration=Join-Path $trainerRoot 'fixtures\calibration.json' }
if (-not $InitialCheckpoint) { $InitialCheckpoint=Join-Path $trainerRoot 'champion\model.pt' }
if (-not $Python) {
    if ($env:CHROMATIC_PYTHON) { $Python=$env:CHROMATIC_PYTHON }
    elseif (Test-Path -LiteralPath (Join-Path $trainerRoot '.venv\Scripts\python.exe')) { $Python=Join-Path $trainerRoot '.venv\Scripts\python.exe' }
    elseif (Test-Path -LiteralPath (Join-Path $trainerRoot '..\..\work\.venv\Scripts\python.exe')) { $Python=(Resolve-Path -LiteralPath (Join-Path $trainerRoot '..\..\work\.venv\Scripts\python.exe')).Path }
    else { $Python=(Get-Command python -ErrorAction Stop).Source }
}
$ci=[System.Globalization.CultureInfo]::InvariantCulture
$arguments=@('-m','shardtrain.run')
$mapping=@{
    Positions=@('--positions',"$Positions"); ShardSize=@('--shard-size',"$ShardSize"); Epochs=@('--epochs',"$Epochs"); BatchSize=@('--batch-size',"$BatchSize");
    LearningRate=@('--learning-rate',$LearningRate.ToString($ci)); DataDirectory=@('--data-directory',$DataDirectory); CheckpointDirectory=@('--checkpoint-directory',$CheckpointDirectory);
    MaxDiskGB=@('--max-disk-gb',$MaxDiskGB.ToString($ci)); MaxMemoryGB=@('--max-memory-gb',$MaxMemoryGB.ToString($ci)); Seed=@('--seed',"$Seed"); Source=@('--source',$Source);
    Calibration=@('--calibration',$Calibration); InitialCheckpoint=@('--initial-checkpoint',$InitialCheckpoint); MaxDownloadMB=@('--max-download-mb',"$MaxDownloadMB");
    CheckpointEvery=@('--checkpoint-every',"$CheckpointEvery"); ReplaySize=@('--replay-size',"$ReplaySize")
}
foreach ($entry in $mapping.GetEnumerator()) {
    if (-not $Resume -or $PSBoundParameters.ContainsKey($entry.Key) -or $entry.Key -eq 'DataDirectory') { $arguments += $entry.Value }
}
if ($Resume) { $arguments+='--resume' }
if ($DryRun) { $arguments+='--dry-run' }
if ($KeepShards) { $arguments+='--keep-shards' }
elseif ($PSBoundParameters.ContainsKey('KeepShards')) { $arguments+='--no-keep-shards' }
$previousPythonPath=$env:PYTHONPATH
$previousDontWrite=$env:PYTHONDONTWRITEBYTECODE
try {
    $env:PYTHONPATH=$trainerRoot + [IO.Path]::PathSeparator + $previousPythonPath
    $env:PYTHONDONTWRITEBYTECODE='1'
    & $Python @arguments
    exit $LASTEXITCODE
} finally { $env:PYTHONPATH=$previousPythonPath; $env:PYTHONDONTWRITEBYTECODE=$previousDontWrite }
