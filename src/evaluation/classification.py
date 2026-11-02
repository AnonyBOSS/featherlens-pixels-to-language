"""Classification metrics, canonical evaluation, and local-image prediction."""
import csv
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from src.data.classification import CubClassificationDataset
from src.data.image_io import load_rgb
from src.data.preparation import load_classification_manifest
from src.data.source import load_cub_split
from src.training.runtime import validate_preflight_preparation


def classification_report(probabilities, labels, class_names, image_ids):
    """Report all declared classes; rows=true, columns=predicted, zero division=0.

    Input probabilities may be uncalibrated. No cross-entropy or calibrated
    confidence claim is inferred from an ensemble probability distribution.
    """
    if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2
            or not probabilities.is_floating_point() or not len(probabilities)):
        raise ValueError('Expected nonempty floating-point NxC probabilities')
    names, identifiers = list(class_names), list(image_ids)
    if (not names or any(not isinstance(n,str) or not n for n in names)
            or len(set(names)) != len(names) or probabilities.shape[1] != len(names)):
        raise ValueError('Ordered class names must be unique and match probabilities')
    if (not isinstance(labels,torch.Tensor) or labels.dtype != torch.int64
            or labels.shape != (len(probabilities),)):
        raise ValueError('Labels must be an int64 vector matching the samples')
    if (len(identifiers) != len(labels) or any(not isinstance(i,str) or not i for i in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError('Each sample needs one unique nonempty image_id')
    p, truth = probabilities.detach().cpu(), labels.detach().cpu()
    if not torch.isfinite(p).all():
        raise FloatingPointError('Non-finite prediction probabilities')
    if ((p < 0).any() or (p > 1).any()
            or not torch.allclose(p.sum(1),torch.ones(len(p),dtype=p.dtype),atol=1e-6,rtol=0)):
        raise ValueError('Probabilities must be nonnegative rows summing to one')
    if ((truth < 0).any() or (truth >= len(names)).any()):
        raise ValueError('Label is outside the ordered class mapping')
    prediction = p.argmax(1)
    ranked = p.argsort(dim=1,descending=True,stable=True)[:, :min(5,len(names))]
    confusion = torch.bincount(truth*len(names)+prediction,minlength=len(names)**2).reshape(len(names),len(names))
    per_class=[]
    for i,name in enumerate(names):
        tp=int(confusion[i,i]);support=int(confusion[i].sum());count=int(confusion[:,i].sum())
        per_class.append({'class_id':i,'class_name':name,'support':support,'predicted_count':count,
            'true_positive':tp,'precision':tp/count if count else 0.,'recall':tp/support if support else 0.,
            'f1':2*tp/(support+count) if support+count else 0.})
    n=len(truth)
    predictions={identifier:{'true_label':int(y),'predicted_label':int(pred),'top5':top,
        'confidence':float(p[row,pred])} for row,(identifier,y,pred,top) in
        enumerate(zip(identifiers,truth.tolist(),prediction.tolist(),ranked.tolist()))}
    return {'num_samples':n,'top1_accuracy':int(confusion.diag().sum())/n,
        'top5_accuracy':int((ranked==truth[:,None]).any(1).sum())/n,
        **{f'macro_{m}':sum(c[m] for c in per_class)/len(names) for m in ('precision','recall','f1')},
        **{f'weighted_{m}':sum(c[m]*c['support'] for c in per_class)/n for m in ('precision','recall','f1')},
        'class_names':names,'per_class':per_class,'confusion_matrix':confusion.tolist(),
        'predictions':predictions,'confidence_type':'uncalibrated_model_probability'}


def predict_image(classifier, image_path, *, top_k=5):
    """Decode one local image through the shared EXIF/RGB/white-alpha policy."""
    if type(top_k) is not int or not 1 <= top_k <= len(classifier.class_names):
        raise ValueError('top_k must be between one and the number of classes')
    with load_rgb({'path':str(image_path)}) as image:
        batches={name:recipe.build_transform(training=False)(image).unsqueeze(0)
                 for name,recipe in classifier.preprocessing.items()}
    probabilities=classifier.predict_probabilities(batches)
    if probabilities.shape != (1,len(classifier.class_names)) or not torch.isfinite(probabilities).all():
        raise ValueError('Classifier did not return finite one-image probabilities')
    order=probabilities[0].argsort(descending=True,stable=True)[:top_k].tolist()
    return {'class_id':order[0],'class_name':classifier.class_names[order[0]],
        'probability':float(probabilities[0,order[0]]),'top_k':[{'class_id':i,'class_name':classifier.class_names[i],
        'probability':float(probabilities[0,i])} for i in order],
        'confidence_type':'uncalibrated_model_probability','device':str(classifier.device)}


def evaluate_classifier(classifier, prepared_dir, *, partition='val', batch_size=8,
                        num_workers=0, allow_test=False, output_dir=None):
    """Evaluate a complete canonical partition without augmentation or filtering.

    Official test loading requires explicit opt-in. Model selection belongs to
    training/validation; the flag is for a frozen final evaluation protocol.
    """
    if partition not in ('train','val','test'):
        raise ValueError('partition must be train, val, or test')
    if type(allow_test) is not bool or partition=='test' and not allow_test:
        raise ValueError('Evaluating official test requires allow_test=True explicitly')
    if type(batch_size) is not int or batch_size<1 or type(num_workers) is not int or num_workers<0:
        raise ValueError('batch_size must be positive and num_workers nonnegative')
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError('Evaluation output directory already exists; preserve previous results')
    started=time.perf_counter()
    identity=validate_preflight_preparation(prepared_dir)
    if identity['manifest_sha256'] != classifier.manifest_sha256:
        raise ValueError('Classifier and preparation manifest hashes differ')
    if identity['class_names'] != classifier.class_names:
        raise ValueError('Classifier and preparation class mappings differ')
    manifest=load_classification_manifest(prepared_dir)
    source=load_cub_split('test' if partition=='test' else 'train')
    if list(source.class_names) != classifier.class_names:
        raise ValueError('Classifier and source class mappings differ')
    datasets={name:CubClassificationDataset(source,manifest,partition=partition,
        transform=classifier.preprocessing[name].build_transform(training=False)) for name in classifier.models}
    reference=next(iter(datasets.values())).rows
    expected=[(r['image_id'],r['class_id']) for r in reference]
    if any([(r['image_id'],r['class_id']) for r in d.rows] != expected for d in datasets.values()):
        raise ValueError('Component source identity alignment differs')
    loaders={name:DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,
        pin_memory=classifier.device.type=='cuda') for name,dataset in datasets.items()}
    chunks=[]; labels=[]; identifiers=[]
    for batches in zip(*loaders.values(),strict=True):
        first=batches[0]
        if any(b['image_id'] != first['image_id'] or not torch.equal(b['label'],first['label']) for b in batches[1:]):
            raise ValueError('Component batch identity alignment differs')
        chunks.append(classifier.predict_probabilities({name:b['image'] for name,b in zip(loaders,batches,strict=True)}))
        labels.append(first['label']);identifiers.extend(first['image_id'])
    probabilities,truth=torch.cat(chunks),torch.cat(labels)
    if list(zip(identifiers,truth.tolist())) != expected:
        raise ValueError('Evaluation coverage differs from the canonical partition')
    report=classification_report(probabilities,truth,classifier.class_names,identifiers)
    report.update(status='completed',complete=True,partition=partition,expected_rows=len(reference),
        manifest_sha256=classifier.manifest_sha256,device=str(classifier.device),batch_size=batch_size,
        num_workers=num_workers,component_settings=classifier.component_settings,augmentation=False,
        elapsed_seconds=time.perf_counter()-started,official_test_accessed=partition=='test')
    if output_dir is not None:
        save_evaluation(report,output_dir,probabilities=probabilities,labels=truth)
    return report


def save_evaluation(report, output_dir, *, probabilities=None, labels=None):
    """Save readable metrics/predictions, a confusion figure, and optional scores."""
    destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=False)
    if probabilities is not None:
        path=destination/'probabilities.pt'
        torch.save({'probabilities':probabilities,'labels':labels,'image_id':list(report['predictions']),
            'class_names':report['class_names'],'manifest_sha256':report['manifest_sha256'],
            'partition':report['partition']},path)
        with path.open('rb') as stream:report['probabilities_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    (destination/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    with (destination/'predictions.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(('image_id','true_label','predicted_label','probability'))
        writer.writerows((i,r['true_label'],r['predicted_label'],r['confidence']) for i,r in report['predictions'].items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    keys=('top1_accuracy','macro_precision','macro_recall','macro_f1')
    axes[0].bar(['Top-1','Precision','Recall','F1'],[100*report[k] for k in keys],color='#176b87')
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Score (%)');axes[0].set_title('Classification; precision/recall/F1 are macro averages')
    image=axes[1].imshow(report['confusion_matrix'],cmap='Blues',interpolation='nearest')
    axes[1].set(xlabel='Predicted class ID',ylabel='True class ID',title='Confusion matrix; original ordered class IDs')
    fig.colorbar(image,ax=axes[1],label='Images');fig.savefig(destination/'evaluation.png',dpi=160);plt.close(fig)
    return destination/'metrics.json'
