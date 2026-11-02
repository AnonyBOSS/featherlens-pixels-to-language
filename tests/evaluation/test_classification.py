"""Numerical reports and source-boundary checks without real model weights."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from src.evaluation import classification as module


def test_macro_weighted_counts_are_hand_computed_and_ties_stable():
    p = torch.tensor([[.8,.1,.1],[.1,.8,.1],[.1,.7,.2],[.1,.8,.1]])
    labels = torch.tensor([0,0,1,2]); ids = ['a','b','c','d']
    report = module.classification_report(p, labels, ['A','B','C'], ids)
    assert report['confusion_matrix'] == [[1,1,0],[0,1,0],[0,1,0]]
    assert report['top1_accuracy'] == .5 and report['top5_accuracy'] == 1
    assert report['macro_precision'] == pytest.approx(4/9)
    assert report['macro_recall'] == pytest.approx(.5)
    assert report['macro_f1'] == pytest.approx(7/18)
    assert report['weighted_precision'] == pytest.approx(7/12)
    assert report['weighted_recall'] == .5
    assert report['weighted_f1'] == pytest.approx(11/24)
    assert report['predictions']['a']['top5'] == [0,1,2]
    assert report['per_class'][2]['support'] == 1


def test_absent_classes_remain_in_macro_denominator():
    report = module.classification_report(torch.tensor([[.9,.1]]),torch.tensor([0]),['A','B'],['id'])
    assert report['macro_f1'] == .5 and report['weighted_f1'] == 1


@pytest.mark.parametrize('issue', ['duplicate','nonfinite','negative','not_normalized','bad_label','float_label','empty'])
def test_invalid_report_inputs_fail_explicitly(issue):
    p=torch.tensor([[.8,.2],[.2,.8]]); y=torch.tensor([0,1]); ids=['a','b']
    if issue=='duplicate': ids=['a','a']
    if issue=='nonfinite': p[0,0]=float('nan')
    if issue=='negative': p[0]=torch.tensor([1.1,-.1])
    if issue=='not_normalized': p[0]=torch.tensor([.2,.2])
    if issue=='bad_label': y[0]=2
    if issue=='float_label': y=y.float()
    if issue=='empty': p,y,ids=p[:0],y[:0],[]
    with pytest.raises((ValueError,FloatingPointError)):
        module.classification_report(p,y,['A','B'],ids)


def test_official_test_requires_explicit_opt_in_before_source_fetch(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'load_cub_split',lambda *a,**k: pytest.fail('source must not open'))
    with pytest.raises(ValueError,match='official'):
        module.evaluate_classifier(SimpleNamespace(),tmp_path,partition='test')


class Samples(Dataset):
    def __init__(self,ids): self.rows=[{'image_id':i,'class_id':j%2} for j,i in enumerate(ids)]
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        return {'image':torch.full((3,2,2),float(index)), 'label':index%2,'image_id':self.rows[index]['image_id']}


def fake_classifier():
    prep=SimpleNamespace(build_transform=lambda **kwargs:None)
    def predict(batches):
        indices=batches['tiny'][:,0,0,0].long()
        return torch.tensor([[.8,.2],[.1,.9],[.7,.3]])[indices]
    return SimpleNamespace(models={'tiny':object(),'base':object()},preprocessing={'tiny':prep,'base':prep},
        class_names=['A','B'],manifest_sha256='a'*64,device=torch.device('cpu'),
        component_settings={'tiny':{'cpu_precision':'float32'},'base':{'cpu_precision':'float32'}},predict_probabilities=predict)


def fixtures(monkeypatch):
    identity={'manifest_sha256':'a'*64,'class_names':['A','B']}
    monkeypatch.setattr(module,'validate_preflight_preparation',lambda path:identity)
    monkeypatch.setattr(module,'load_classification_manifest',lambda path:[])
    monkeypatch.setattr(module,'load_cub_split',lambda name:SimpleNamespace(class_names=['A','B']))
    monkeypatch.setattr(module,'CubClassificationDataset',lambda *a,**k:Samples(['a','b','c']))


def test_evaluate_covers_all_rows_and_preserves_model_alignment(monkeypatch,tmp_path):
    fixtures(monkeypatch)
    result=module.evaluate_classifier(fake_classifier(),tmp_path,batch_size=2)
    assert result['status']=='completed' and result['complete'] is True
    assert result['expected_rows']==result['num_samples']==3
    assert result['top1_accuracy']==1 and list(result['predictions'])==['a','b','c']
    assert result['partition']=='val' and result['manifest_sha256']=='a'*64


def test_manifest_mismatch_fails_before_source_fetch(monkeypatch,tmp_path):
    fixtures(monkeypatch)
    monkeypatch.setattr(module,'load_cub_split',lambda *a:pytest.fail('source must not open'))
    classifier=fake_classifier();classifier.manifest_sha256='b'*64
    with pytest.raises(ValueError,match='manifest'):
        module.evaluate_classifier(classifier,tmp_path)


def test_component_image_id_mismatch_is_not_silently_paired(monkeypatch,tmp_path):
    fixtures(monkeypatch)
    calls=[]
    def dataset(*a,**k):
        calls.append(1);return Samples(['a','b','c'] if len(calls)==1 else ['b','a','c'])
    monkeypatch.setattr(module,'CubClassificationDataset',dataset)
    with pytest.raises(ValueError,match='align|identit'):
        module.evaluate_classifier(fake_classifier(),tmp_path)
