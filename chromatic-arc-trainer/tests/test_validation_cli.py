"""CLI routing only: importing the validation tool initializes no torch/GPU."""
from pathlib import Path
import importlib.util
import inspect

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('validate_arc_cli',ROOT/'tools/validate_arc.py')
validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)


@pytest.mark.parametrize('extra,expected,profile', [([], 'full',False),(['--validation-mode','selected'],'selected',False),
    (['--validation-mode','selected','--profile-resume'],'selected',True)])
def test_recovery_cli_mode_is_forwarded_and_default_is_compatible(tmp_path,monkeypatch,extra,expected,profile):
    received={}
    original_signature=inspect.signature(validation.recovery_validation)
    def no_training(*args):
        bound=original_signature.bind(*args).arguments
        received['mode']=bound['validation_mode'];received['profile_resume']=bound['profile_resume'];received['device']=bound['device'];return {'passed':True}
    monkeypatch.setattr(validation,'recovery_validation',no_training)
    assert validation.main(['--phase','recovery','--device','cpu','--directory',str(tmp_path/'new'),
        '--output',str(tmp_path/'report.json'),*extra])==0
    assert received=={'mode':expected,'profile_resume':profile,'device':'cpu'}
    assert not (tmp_path/'new').exists() and not (tmp_path/'report.json').exists()


def test_invalid_recovery_mode_fails_before_files_are_created(tmp_path):
    with pytest.raises(ValueError,match='full or selected'):
        validation.recovery_validation(tmp_path/'new',validation_mode='unknown')
    assert not (tmp_path/'new').exists()
