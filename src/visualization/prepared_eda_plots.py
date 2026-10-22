"""Class-coverage figures from saved, prepared-training EDA counts."""
from collections import Counter
import math
from pathlib import Path

from .eda_plots import _BLUE, _GOLD, _INK, _MUTED, _STYLE, _TEAL, _count_axis, _empty, _pyplot
from .figure_export import save_figure_bundle


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validated(report):
    try:
        if report["schema_version"] != "prepared-eda-1.0" or report["source"]["source_split"] != "train":
            raise ValueError("Prepared EDA figures require the training-only prepared-eda-1.0 report")
        if any(not isinstance(report["source"][key], str) or not report["source"][key].strip() for key in ("repo_id", "revision")):
            raise ValueError("Prepared EDA figures need source repository and revision provenance")
        preparation = report["preparation"]
        versions = {"policy_version": "classification-v1", "split_version": "stratified-exact-group-v1", "pixel_hash_algorithm": "exif-rgb8-white:v1"}
        if type(preparation["schema_version"]) is not int or preparation["schema_version"] != 1 or any(preparation[key] != value for key, value in versions.items()):
            raise ValueError("Unsupported preparation metadata")
        digest = preparation["manifest_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("Preparation manifest fingerprint is missing or invalid")
        _count(preparation["seed"], "seed")
        fraction = preparation["val_fraction"]
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("Preparation validation fraction must be in (0, 1)")
        if preparation["source_revisions"]["train"] != report["source"]["revision"]:
            raise ValueError("Preparation and report revisions differ")
        names, rows = report["class_names"], report["classes"]
        if not isinstance(names, list) or any(not isinstance(name, str) or not name.strip() for name in names) or len(names) != len(set(names)) or not isinstance(rows, list) or len(rows) != len(names):
            raise ValueError("Class coverage requires the complete ordered class mapping")
        totals = dict.fromkeys(("train", "val", "excluded"), 0)
        distributions = {key: [] for key in ("raw", "train", "val")}
        for class_id, (name, row) in enumerate(zip(names, rows)):
            if type(row["class_id"]) is not int or row["class_id"] != class_id or row["class_name"] != name:
                raise ValueError("Class summaries must preserve source order")
            counts = row["partition_counts"]
            if set(counts) != set(totals):
                raise ValueError("Class partition counts have unexpected fields")
            for partition in totals:
                totals[partition] += _count(counts[partition], partition)
            distributions["raw"].append(sum(counts.values()))
            for partition in ("train", "val"):
                distributions[partition].append(counts[partition])
        population = report["population"]
        for key in ("raw_training_image_count", "derived_training_image_count", "validation_image_count", "excluded_training_image_count", "unknown_class_training_image_count"):
            _count(population[key], key)
        unknown = population["unknown_class_training_image_count"]
        if totals["train"] != population["derived_training_image_count"] or totals["val"] != population["validation_image_count"] or totals["excluded"] + unknown != population["excluded_training_image_count"]:
            raise ValueError("Class coverage counts do not reconcile with the population")
        if population["raw_training_image_count"] != sum(totals.values()) + unknown or preparation["source_counts"]["train"] != population["raw_training_image_count"]:
            raise ValueError("Raw training population is inconsistent")
        expected_partitions = {"train": totals["train"], "val": totals["val"], "excluded": totals["excluded"] + unknown, "test": _count(report["official_test"]["identity_count"], "test identity count")}
        for field in ("partition_counts", "source_counts"):
            for key, value in preparation[field].items():
                _count(value, f"preparation.{field}.{key}")
        if preparation["partition_counts"] != expected_partitions or preparation["source_counts"]["test"] != expected_partitions["test"]:
            raise ValueError("Preparation counts are inconsistent")
        if report["official_test"]["identities_preserved"] is not True or report["official_test"]["image_caption_statistics_used"] is not False or report["official_test"]["preserved_test_partition_count"] != expected_partitions["test"]:
            raise ValueError("Official test assignments must be preserved")
        saved = report["class_count_distributions"]
        for table in saved.values():
            if not isinstance(table, list):
                raise ValueError("Saved class-count distributions must be lists")
            for row in table:
                _count(row["images_per_class"], "images_per_class")
                _count(row["class_count"], "class_count")
        expected = {key: [{"images_per_class": number, "class_count": count} for number, count in sorted(Counter(values).items())] for key, values in distributions.items()}
        if saved != expected:
            raise ValueError("Saved class-count distributions disagree with class coverage")
        return rows
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Missing or malformed prepared EDA plotting field: {error}") from error


def create_prepared_class_coverage_figure(report):
    rows = _validated(report)
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(15, 5.8), layout="constrained")
        try:
            for partition, label, color in (("train", "Derived training", _BLUE), ("val", "Validation", _TEAL)):
                values = [row["partition_counts"][partition] for row in rows]
                axes[0].plot(range(len(rows)), values, linewidth=1.5, marker=".", markersize=3, label=f"{label}: {sum(count > 0 for count in values)}/{len(rows)} classes", color=color)
            axes[0].set(title="Counts in source class order", xlabel="Source class ID", ylabel="Images in the class")
            axes[0].legend(fontsize=9)
            _count_axis(axes[0])
            tables = report["class_count_distributions"]
            counts = sorted({row["images_per_class"] for table in tables.values() for row in table})
            width = 0.25
            for offset, scope, label, color in ((-1, "raw", "Raw official training", _GOLD), (0, "train", "Derived training", _BLUE), (1, "val", "Validation", _TEAL)):
                frequencies = {row["images_per_class"]: row["class_count"] for row in tables[scope]}
                axes[1].bar([position + offset * width for position in range(len(counts))], [frequencies.get(count, 0) for count in counts], width=width, label=label, color=color)
            axes[1].set_xticks(range(len(counts)), labels=[str(count) for count in counts])
            axes[1].set(title="Distribution of per-class counts", xlabel="Images per class", ylabel="Classes")
            axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
            _count_axis(axes[1])
            if not rows:
                for axis in axes:
                    _empty(axis, "No source classes available")
            population = report["population"]
            figure.suptitle(f"Derived training and validation class coverage\nOfficial TRAINING source: {population['raw_training_image_count']:,} images | Derived training: {population['derived_training_image_count']:,} | Validation: {population['validation_image_count']:,}", fontsize=13, color=_INK)
            figure.supxlabel(f"One class per count-distribution observation. Raw per-class counts include known-class exclusions.\nExcluded images: {population['excluded_training_image_count']:,}; unknown-class excluded images: {population['unknown_class_training_image_count']:,}. Membership is supplied by the verified shared manifest.", fontsize=9, color=_MUTED)
            return figure
        except BaseException:
            plt.close(figure)
            raise


def generate_prepared_eda_figures(report, output_dir: str | Path) -> list[Path]:
    _validated(report)
    return save_figure_bundle((("training_prepared_class_coverage", lambda: create_prepared_class_coverage_figure(report)),), output_dir)
