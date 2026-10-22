"""Join saved training EDA to the shared, verified classification manifest."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from .prepared_eda import build_prepared_training_report
from .reporting import write_json_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--image-report", type=Path, default=Path("outputs/eda/reports/train-images-full.json"))
    parser.add_argument("--caption-report", type=Path, default=Path("outputs/eda/reports/train-caption-details-full.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/eda/reports/prepared-training-eda.json"))
    parser.add_argument("--figures-dir", type=Path)
    args = parser.parse_args(argv)
    image_bytes = args.image_report.read_bytes()
    caption_bytes = args.caption_report.read_bytes()
    image_report = json.loads(image_bytes)
    caption_report = json.loads(caption_bytes)
    report = build_prepared_training_report(args.prepared_dir, image_report=image_report, caption_report=caption_report)
    report["input_report_sha256"] = {"images": sha256(image_bytes).hexdigest(), "captions": sha256(caption_bytes).hexdigest()}
    write_json_report(report, args.output)
    population = report["population"]
    print(f"Saved prepared EDA: {population['derived_training_image_count']:,} training / {population['validation_image_count']:,} validation images to {args.output}", flush=True)
    if args.figures_dir is not None:
        from src.visualization.prepared_eda_plots import generate_prepared_eda_figures
        paths = generate_prepared_eda_figures(report, args.figures_dir)
        print(f"Saved {len(paths)} prepared-data figure group as PNG/PDF", flush=True)


if __name__ == "__main__":
    main()
