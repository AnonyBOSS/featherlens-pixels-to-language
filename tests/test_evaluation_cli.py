"""Public CLI dispatch, help, and safe source-boundary errors."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('command',['export-classifier','predict','evaluate','gradcam'])
def test_delivery_commands_have_real_help(command):
    result=subprocess.run([sys.executable,'-B','-m','src.cli',command,'--help'],cwd=ROOT,
        capture_output=True,text=True,timeout=40)
    assert result.returncode==0 and 'usage:' in result.stdout


def test_predict_dispatches_to_shared_library(monkeypatch,capsys):
    from src import cli
    classifier=object()
    monkeypatch.setattr('src.evaluation.inference.load_classifier',lambda path,**kwargs:classifier)
    def predict(loaded,path,**kwargs):
        assert loaded is classifier and Path(path)==Path('bird.png') and kwargs=={'top_k':3}
        return {'class_id':7,'class_name':'bird'}
    monkeypatch.setattr('src.evaluation.classification.predict_image',predict)
    assert cli.main(['predict','--bundle','bundle.json','--image','bird.png','--top-k','3'])==0
    assert json.loads(capsys.readouterr().out)['class_id']==7


def test_evaluate_test_flag_is_required_before_loading_models(monkeypatch,capsys):
    from src import cli
    monkeypatch.setattr('src.evaluation.inference.load_classifier',lambda *a,**k:pytest.fail('model must not load'))
    assert cli.main(['evaluate','--bundle','missing.json','--prepared-dir','prepared',
        '--partition','test','--output-dir','report'])==1
    assert 'official' in capsys.readouterr().err.lower()


def test_evaluate_forwards_canonical_partition_and_options(monkeypatch,tmp_path,capsys):
    from src import cli
    classifier=object()
    monkeypatch.setattr('src.evaluation.inference.load_classifier',lambda *a,**k:classifier)
    def evaluate(loaded,path,**kwargs):
        assert loaded is classifier and Path(path)==Path('prepared')
        assert kwargs=={'partition':'val','batch_size':8,'num_workers':0,'allow_test':False,'output_dir':tmp_path/'evaluation'}
        return {'status':'completed','partition':'val','num_samples':1200,'top1_accuracy':.9}
    monkeypatch.setattr('src.evaluation.classification.evaluate_classifier',evaluate)
    assert cli.main(['evaluate','--bundle','bundle.json','--prepared-dir','prepared','--output-dir',str(tmp_path/'evaluation')])==0
    assert json.loads(capsys.readouterr().out)['num_samples']==1200


def test_delivery_runtime_preserves_measured_cuda_numerical_profile(monkeypatch):
    import torch
    from src import cli
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        cli._delivery_runtime()
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])
        torch.backends.cudnn.benchmark = previous[3]
        torch.backends.cudnn.deterministic = previous[4]
        torch.backends.cudnn.allow_tf32 = previous[5]
        torch.backends.cuda.matmul.allow_tf32 = previous[6]
