"""Small command line entry points for training and verified classifier delivery."""
import argparse
import json
from pathlib import Path
import sys


def _delivery_runtime():
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Match the measured FP32 Base encoder: cuDNN may use TF32 kernels.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


def main(argv=None):
    parser = argparse.ArgumentParser(description='FeatherLens classification')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='Train classification using a prepared canonical split')
    train.add_argument('--task', required=True, choices=['classification'])
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--prepared-dir', type=Path, required=True)
    train.add_argument('--run-dir', type=Path, required=True)
    train.add_argument('--device', default='cuda')
    initialization = train.add_mutually_exclusive_group()
    initialization.add_argument('--resume', action='store_true')
    initialization.add_argument('--initial-checkpoint', type=Path,
                                help='Initialize a fresh run from checkpoint model weights only')
    train.add_argument('--tracking', choices=['mlflow', 'json'], default='mlflow')

    export = commands.add_parser('export-classifier', help='Package original trained Tiny and scaled Base for offline inference')
    export.add_argument('--tiny-checkpoint', type=Path, required=True)
    export.add_argument('--base-weights', type=Path, required=True)
    export.add_argument('--base-head', type=Path, required=True)
    export.add_argument('--output-dir', type=Path, required=True)
    predict = commands.add_parser('predict', help='Classify one local image using a portable bundle')
    evaluate = commands.add_parser('evaluate', help='Evaluate a complete prepared partition; test requires explicit opt-in')
    cam = commands.add_parser('gradcam', help='Explain a raw component class score on one local image')
    for command in (predict,evaluate,cam):
        command.add_argument('--bundle',type=Path,required=True)
        command.add_argument('--device',default='cpu')
    for command in (predict,cam):
        command.add_argument('--image',type=Path,required=True)
    predict.add_argument('--top-k',type=int,default=5)
    evaluate.add_argument('--prepared-dir',type=Path,required=True)
    evaluate.add_argument('--partition',choices=['train','val','test'],default='val')
    evaluate.add_argument('--batch-size',type=int,default=8)
    evaluate.add_argument('--num-workers',type=int,default=0)
    evaluate.add_argument('--allow-official-test',action='store_true',help='Only for an explicitly frozen final test protocol')
    evaluate.add_argument('--output-dir',type=Path,required=True)
    cam.add_argument('--component',required=True,help='Component name in the bundle, e.g. tiny or base')
    cam.add_argument('--target-class',type=int,help='Default: the ensemble predicted class, shared across component maps')
    cam.add_argument('--output',type=Path,required=True,help='PNG overlay path; JSON metadata is saved beside it')
    args = parser.parse_args(argv)
    try:
        if args.command == 'train':
            from src.training.experiment import run_training
            config = json.loads(args.config.read_text(encoding='utf-8'))
            options = ({'initial_checkpoint': args.initial_checkpoint} if args.initial_checkpoint is not None else {})
            result = run_training(args.prepared_dir,args.run_dir,config,device=args.device,
                resume=args.resume,tracking=args.tracking,**options)
        else:
            if args.command == 'evaluate' and args.partition == 'test' and not args.allow_official_test:
                raise ValueError('Evaluating official test requires --allow-official-test explicitly')
            _delivery_runtime()
            from src.evaluation.inference import export_classifier_bundle, load_classifier
            if args.command == 'export-classifier':
                path=export_classifier_bundle(args.tiny_checkpoint,args.base_weights,args.base_head,args.output_dir)
                result={'status':'exported','bundle':str(path)}
            else:
                classifier=load_classifier(args.bundle,device=args.device)
                from src.evaluation.classification import predict_image,evaluate_classifier
                if args.command == 'predict':
                    result=predict_image(classifier,args.image,top_k=args.top_k)
                elif args.command == 'evaluate':
                    report=evaluate_classifier(classifier,args.prepared_dir,partition=args.partition,
                        batch_size=args.batch_size,num_workers=args.num_workers,
                        allow_test=args.allow_official_test,output_dir=args.output_dir)
                    result={k:v for k,v in report.items() if k not in
                        ('predictions','per_class','confusion_matrix','class_names')}
                else:
                    from src.data.image_io import load_rgb
                    from src.evaluation.gradcam import compute_gradcam,render_gradcam
                    if args.component not in classifier.models:
                        raise ValueError('Unknown component name in classifier bundle')
                    if args.output.exists() or args.output.with_suffix('.json').exists():
                        raise FileExistsError('Grad-CAM output already exists; preserve previous results')
                    prediction=predict_image(classifier,args.image)
                    target=prediction['class_id'] if args.target_class is None else args.target_class
                    with load_rgb({'path':str(args.image)}) as image:
                        recipe=classifier.preprocessing[args.component]
                        tensor=recipe.build_transform(training=False)(image).unsqueeze(0).to(classifier.device)
                        computed=compute_gradcam(classifier.models[args.component],tensor,target)
                        render_gradcam(image,computed['heatmap'],recipe.image_size,output_path=args.output)
                    result={k:v for k,v in computed.items() if k not in ('heatmap','logits')}
                    result.update(status='written',component=args.component,target_class_name=classifier.class_names[target],
                        component_prediction_name=classifier.class_names[computed['predicted_class']],
                        classifier_prediction=prediction,output=args.output.name,
                        interpretation='Raw component-score attribution; relative colors are not causal proof or ensemble contributions.')
                    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
        print(json.dumps(result,allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print('Training interrupted; resume from the latest completed checkpoint.' if args.command=='train'
              else 'Classification operation interrupted.',file=sys.stderr)
        return 130
    except Exception as error:
        prefix='Training' if args.command=='train' else 'Classification'
        print(f'{prefix} error: {type(error).__name__}: {error}',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
