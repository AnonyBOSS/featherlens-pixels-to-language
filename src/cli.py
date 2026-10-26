"""Small command line entry point for the shared classification experiment."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='FeatherLens training')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='Train classification using a prepared canonical split')
    train.add_argument('--task', required=True, choices=['classification'])
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--prepared-dir', type=Path, required=True)
    train.add_argument('--run-dir', type=Path, required=True)
    train.add_argument('--device', default='cuda')
    train.add_argument('--resume', action='store_true')
    train.add_argument('--tracking', choices=['mlflow', 'json'], default='mlflow')
    args = parser.parse_args(argv)
    from src.training.runtime import _failure
    try:
        from src.training.experiment import run_training
        config = json.loads(args.config.read_text(encoding='utf-8'))
        summary = run_training(args.prepared_dir, args.run_dir, config, device=args.device,
                               resume=args.resume, tracking=args.tracking)
        print(json.dumps(summary, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print('Training interrupted; resume from the latest completed checkpoint.', file=sys.stderr)
        return 130
    except Exception as error:
        failure = _failure('cli', error)
        print(f"Training error: {failure['type']}: {failure['message']}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
