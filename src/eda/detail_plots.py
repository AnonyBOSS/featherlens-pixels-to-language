"""Caption-detail and prepared-split class-coverage plots from saved reports."""

from collections import Counter
import math
from pathlib import Path
import textwrap

from .plots import _BLUE, _GOLD, _INK, _MUTED, _STYLE, _TEAL, _count_axis, _empty, _pyplot, save_figure_bundle


_CATEGORIES = (("color", "Color"), ("anatomy", "Anatomy"), ("direction", "Direction"))


def _validate_caption_details(report):
    from .captions import validate_caption_details_report

    return validate_caption_details_report(report)


def _title(figure, report, details, title):
    population = report["population"]
    if population["selection_method"] == "source_order_prefix":
        scope = "nonrandom source-order prefix"
    elif population["full_split_verified"]:
        scope = "complete split"
    else:
        scope = "complete iterator; reference count mismatch"
    coverage = details["coverage"]
    figure.suptitle(
        f"{title}\nOfficial TRAINING | {scope}\n"
        f"{coverage['image_count']:,} images | {coverage['nonempty_caption_count']:,} nonempty captions",
        fontsize=13, color=_INK,
    )


def create_caption_language_figure(report):
    """Plot caption mention percentages for the color, anatomy, and direction lexicons."""
    details = _validate_caption_details(report)
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(16, 8), layout="constrained")
        try:
            categories = {row["category"]: row for row in details["term_prevalence"]}
            from matplotlib.ticker import PercentFormatter

            for axis, (category, label), color in zip(axes, _CATEGORIES, (_BLUE, _TEAL, _GOLD)):
                prevalence = categories[category]
                selected = sorted(
                    (row for row in prevalence["terms"] if row["caption_count"] > 0),
                    key=lambda row: (-row["caption_count"], row["term"]),
                )[:10]
                positions = list(range(len(selected)))
                percentages = [row["caption_prevalence"] * 100 for row in selected]
                bars = axis.barh(positions, percentages, height=0.7, color=color)
                axis.set_yticks(positions, labels=[row["term"] for row in selected])
                axis.invert_yaxis()
                union = prevalence["caption_prevalence"]
                union_text = "undefined: no nonempty captions" if union is None else f"{union:.1%} of nonempty captions"
                axis.set(title=f"{label} terms: up to 10\nCategory mentions: {union_text}", xlabel="Nonempty captions mentioning the term")
                upper = min(100.0, max(1.0, 1.25 * max(percentages, default=0.0))) if category == "direction" else 100.0
                axis.set_xlim(0, upper)
                decimals = (2 if upper <= 1 else 1) if category == "direction" else 0
                axis.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=decimals))
                axis.set_axisbelow(True)
                axis.grid(axis="x")
                labels = (
                    [f"{percent:.3f}% (n={row['caption_count']:,})" for percent, row in zip(percentages, selected)]
                    if category == "direction" else [f"{percent:.1f}%" for percent in percentages]
                )
                axis.bar_label(bars, labels=labels, padding=3, fontsize=9, color=_INK)
                if not selected:
                    _empty(axis, "No nonempty captions available" if union is None else "No declared terms matched")
            _title(figure, report, details, "Caption language mentions")
            figure.supxlabel(
                "Each bar uses all nonempty captions as its denominator, including captions with zero analysis words.\n"
                "Terms and categories can overlap; whole-token/hyphen-component mentions do not establish visual truth or handle negation.\n"
                "The direction panel uses a finer percentage scale; bar lengths across panels are not directly comparable. Direction n = caption count.",
                fontsize=9, color=_MUTED,
            )
            return figure
        except BaseException:
            plt.close(figure)
            raise


def create_caption_diversity_figure(report):
    """Plot per-image TTR, mean pairwise Jaccard overlap, and TTR vs token count."""
    details = _validate_caption_details(report)
    ttr_rows = [row for row in details["images"] if row["type_token_ratio"] is not None]
    overlap_rows = [row for row in details["images"] if row["pairwise_jaccard_mean"] is not None]
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.8), layout="constrained")
        try:
            for axis, rows, field, title, xlabel, color in (
                (axes[0], ttr_rows, "type_token_ratio", "Per-image type/token ratio", "Unique tokens / total tokens (0–1)", _BLUE),
                (axes[1], overlap_rows, "pairwise_jaccard_mean", "Per-image mean caption overlap", "Mean pairwise token-set Jaccard (0–1)", _TEAL),
            ):
                axis.hist([row[field] for row in rows], bins=20, range=(0, 1), color=color)
                axis.set(title=f"{title}\nAvailable images: {len(rows):,}", xlabel=xlabel, ylabel="Image count", xlim=(0, 1))
                _count_axis(axis)
                if not rows:
                    _empty(axis, "No eligible per-image observations")
            axes[2].scatter([row["token_count"] for row in ttr_rows], [row["type_token_ratio"] for row in ttr_rows], color=_GOLD, s=15, alpha=0.4)
            axes[2].set(
                title=f"TTR and amount of text\nAvailable images: {len(ttr_rows):,}",
                xlabel="All analysis tokens across an image's captions",
                ylabel="Per-image type/token ratio", ylim=(0, 1),
            )
            axes[2].set_xlim(left=0)
            axes[2].grid(alpha=0.75)
            if not ttr_rows:
                _empty(axes[2], "No token-bearing images")
            _title(figure, report, details, "Caption lexical diversity and overlap")
            figure.supxlabel(
                f"Overlap averages unordered token-bearing caption pairs within each image ({details['coverage']['pair_count']:,} pairs total).\n"
                "TTR depends on text length. Repeated captions and stopwords are retained; Jaccard measures lexical overlap, not semantics.",
                fontsize=9, color=_MUTED,
            )
            return figure
        except BaseException:
            plt.close(figure)
            raise


def create_class_caption_variation_figure(report):
    """Plot per-class mean caption lengths by rank and largest/smallest five species."""
    details = _validate_caption_details(report)
    available = [row for row in details["classes"] if row["mean_caption_words"]["mean"] is not None]
    ranked = sorted(available, key=lambda row: (-row["mean_caption_words"]["mean"], row["class_id"]))
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        figure = plt.figure(figsize=(15, 8), layout="constrained")
        try:
            grid = figure.add_gridspec(2, 2, width_ratios=(1.1, 1.4))
            ranks = figure.add_subplot(grid[:, 0])
            largest = figure.add_subplot(grid[0, 1])
            smallest = figure.add_subplot(grid[1, 1])
            ranks.plot(range(1, len(ranked) + 1), [row["mean_caption_words"]["mean"] for row in ranked], marker="o", markersize=3, linewidth=1.6, color=_BLUE)
            ranks.set(
                title=f"Classes with caption means: {len(ranked):,} / {len(details['classes']):,}",
                xlabel="Class rank (largest mean first)",
                ylabel="Class mean of per-image mean words per caption",
            )
            ranks.set_ylim(bottom=0)
            ranks.grid(axis="y")
            if not ranked:
                _empty(ranks, "No class has nonempty captions")
            for axis, selected, title, color in (
                (largest, ranked[:5], "Largest class means", _BLUE),
                (smallest, sorted(available, key=lambda row: (row["mean_caption_words"]["mean"], row["class_id"]))[:5], "Smallest class means", _TEAL),
            ):
                positions = list(range(len(selected)))
                bars = axis.barh(positions, [row["mean_caption_words"]["mean"] for row in selected], color=color, height=0.65)
                axis.set_yticks(positions, labels=[textwrap.fill(row["class_name"], 27) for row in selected], fontsize=9)
                axis.invert_yaxis()
                axis.set(title=title, xlabel="Mean words per caption (images weighted equally)")
                axis.set_xlim(left=0)
                axis.grid(axis="x")
                axis.set_axisbelow(True)
                axis.bar_label(bars, labels=[f"{bar.get_width():.2f}" for bar in bars], padding=4, fontsize=9, color=_INK)
                axis.margins(x=0.16)
                if not selected:
                    _empty(axis, "No available class means")
            _title(figure, report, details, "Caption length variation across species")
            figure.supxlabel(
                "Each image contributes one mean word count; punctuation-only nonempty captions contribute zero words.\n"
                "Classes without available means are omitted. Selected examples break ties by class ID; these comparisons are descriptive.",
                fontsize=9, color=_MUTED,
            )
            return figure
        except BaseException:
            plt.close(figure)
            raise


def generate_caption_detail_figures(report, output_dir: str | Path) -> list[Path]:
    """Validate a caption-details report and save the three PNG/PDF pairs."""
    _validate_caption_details(report)
    return save_figure_bundle(
        (
            ("training_caption_language", lambda: create_caption_language_figure(report)),
            ("training_caption_diversity", lambda: create_caption_diversity_figure(report)),
            ("training_class_caption_variation", lambda: create_class_caption_variation_figure(report)),
        ),
        output_dir,
    )


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate_prepared(report):
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
    """Plot per-class train/val counts and class-size distributions."""
    rows = _validate_prepared(report)
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
    """Validate a prepared-split EDA report and save the class-coverage PNG/PDF pair."""
    _validate_prepared(report)
    return save_figure_bundle((("training_prepared_class_coverage", lambda: create_prepared_class_coverage_figure(report)),), output_dir)
