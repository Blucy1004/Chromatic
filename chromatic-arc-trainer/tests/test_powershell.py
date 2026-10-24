"""Exercise the actual Windows PowerShell entry point and its default paths."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
POWERSHELL=shutil.which('powershell')

@pytest.mark.skipif(os.name!='nt' or not POWERSHELL,reason='Windows PowerShell 5.1 integration')
def test_windows_powershell_default_paths_dry_run(tmp_path):
    data=tmp_path/'data';candidate=tmp_path/'candidate'
    argv=[POWERSHELL,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'chromatic-train.ps1'),
          '-Python',sys.executable,'-DataDirectory',str(data),'-CheckpointDirectory',str(candidate),'-DryRun']
    result=subprocess.run(argv,cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    dry=json.loads(result.stdout)
    assert dry['status']=='dry-run-ok' and not dry['downloads_started'] and not dry['training_started']
    assert dry['configuration']['positions']==100000
    assert dry['configuration']['initial_checkpoint']==str(ROOT/'champion/model.pt')
    assert dry['configuration']['calibration']==str(ROOT/'fixtures/calibration.json')
    assert not data.exists() and not candidate.exists()

