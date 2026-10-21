"""Matplotlib figures from an already saved training EDA report.

This module only arranges reported counts, distributions, and summary markers.
It never loads source records, decodes images, recomputes statistics, removes
stopwords, or chooses model preprocessing. Figure helpers return an open figure
owned by the caller; ``generate_training_figures`` saves and closes every figure.
The caller selects a Matplotlib backend (for example MPLBACKEND=Agg).
"""

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import textwrap
from typing import Any


_BLUE = "#2563a6"
_TEAL = "#138c89"
_INK = "#17324d"
_MUTED = "#526779"
_GOLD = "#c27617"
_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelcolor": _INK,
    "axes.titlecolor": _INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#b7c5cf",
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "grid.color": "#e2e9ef",
    "grid.linewidth": 0.7,
}


def _pyplot():
    # Keep importing this module independent of graphical backend availability.
    import matplotlib.pyplot as plt

    return plt


def _validate_training_report(report: Mapping[str, Any]) -> None:
    """Check saved plot inputs and reconcile counts without fitting statistics."""
    def mapping(value, path):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be an object")
        return value

    def field(value, key, path):
        value = mapping(value, path)
        if key not in value:
            raise ValueError(f"Missing required report field: {path}.{key}")
        return value[key]

    def count(value, path, *, minimum=0):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{path} must be an integer >= {minimum}")
        return value

    def number(value, path):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} must be a finite nonnegative number")
        try:
            valid = math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{path} must be a finite nonnegative number")
        return value

    def text(value, path):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} must be a nonempty string")
        return value

    def rows(value, path):
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        return value

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    mapping(report, "report")
    if report.get("schema_version") != "1.1":
        raise ValueError(
            "Plotting requires report schema_version '1.1'. Regenerate the report "
            "with the current src.analysis.explore command; legacy reports do not "
            "contain the saved plotting distributions."
        )
    source = mapping(field(report, "source", "report"), "source")
    require(field(source, "source_split", "source") == "train", "Training figures require an official training report")
    population = mapping(field(report, "population", "report"), "population")
    selection = text(field(population, "selection_method", "population"), "population.selection_method")
    require(selection in {"complete_official_training_split", "source_order_prefix"}, "Unsupported population.selection_method")
    verified = field(population, "full_split_verified", "population")
    require(isinstance(verified, bool), "population.full_split_verified must be a boolean")
    require(not verified or selection == "complete_official_training_split", "Verified full training requires complete split selection")
    dataset = mapping(field(report, "dataset", "report"), "dataset")
    images = count(field(dataset, "total_records", "dataset"), "dataset.total_records")
    analyzed = count(field(population, "records_analyzed", "population"), "population.records_analyzed")
    require(analyzed == images, "population.records_analyzed must equal dataset.total_records")
    require(field(population, "random_sample", "population") is False, "population.random_sample must be false for supported selection methods")
    if selection == "complete_official_training_split":
        require(field(population, "source_exhausted", "population") is True, "Complete training requires population.source_exhausted=true")
        require(field(population, "requested_limit", "population") is None, "Complete training population.requested_limit must be null")
        reference = count(field(population, "reference_record_count", "population"), "population.reference_record_count", minimum=1)
        matches = field(population, "matches_reference_count", "population")
        require(isinstance(matches, bool), "population.matches_reference_count must be a boolean")
        require(matches == (analyzed == reference), "population.matches_reference_count must match observed versus reference counts")
        require(verified == matches, "population.full_split_verified must match the complete-scan reference check")
    else:
        limit = count(field(population, "requested_limit", "population"), "population.requested_limit", minimum=1)
        require(analyzed <= limit, "Prefix records_analyzed must not exceed requested_limit")
    classes = rows(field(dataset, "class_distribution", "dataset"), "dataset.class_distribution")
    class_ids = set()
    class_total = 0
    for index, row in enumerate(classes):
        path = f"dataset.class_distribution[{index}]"
        identifier = count(field(row, "class_id", path), f"{path}.class_id")
        require(identifier not in class_ids, "dataset.class_distribution class IDs must be unique")
        class_ids.add(identifier)
        text(field(row, "class_name", path), f"{path}.class_name")
        class_total += count(field(row, "count", path), f"{path}.count")
    require(class_total == images, "Class counts must sum to dataset.total_records")

    captions = mapping(field(report, "captions", "report"), "captions")
    caption_images = count(field(captions, "image_count", "captions"), "captions.image_count")
    require(caption_images == images, "captions.image_count must equal dataset.total_records")
    nonempty = count(field(captions, "nonempty_caption_count", "captions"), "captions.nonempty_caption_count")
    count(field(captions, "expected_captions_per_image", "captions"), "captions.expected_captions_per_image")
    tokens = count(field(captions, "total_tokens", "captions"), "captions.total_tokens")

    def distribution(name, value_key, frequency_key, total):
        values = rows(field(captions, name, "captions"), f"captions.{name}")
        seen = set()
        actual_total = 0
        weighted = 0
        for index, row in enumerate(values):
            path = f"captions.{name}[{index}]"
            value = count(field(row, value_key, path), f"{path}.{value_key}")
            frequency = count(field(row, frequency_key, path), f"{path}.{frequency_key}", minimum=1)
            require(value not in seen, f"captions.{name} bins must be unique")
            seen.add(value)
            actual_total += frequency
            weighted += value * frequency
        require(actual_total == total, f"captions.{name} frequencies must sum to {total}")
        return weighted, (min(seen), max(seen)) if seen else None

    words, word_bounds = distribution("word_length_distribution", "word_count", "caption_count", nonempty)
    require(words == tokens, "Weighted word-length counts must equal captions.total_tokens")
    _, character_bounds = distribution("character_length_distribution", "character_count", "caption_count", nonempty)
    coverage, _ = distribution("per_image_nonempty_count_distribution", "caption_count", "image_count", images)
    require(coverage == nonempty, "Weighted per-image caption counts must equal captions.nonempty_caption_count")
    for name, bounds in (("word_lengths", word_bounds), ("character_lengths", character_bounds)):
        summary = mapping(field(captions, name, "captions"), f"captions.{name}")
        for key in ("median", "p95"):
            value = field(summary, key, f"captions.{name}")
            if nonempty:
                number(value, f"captions.{name}.{key}")
            else:
                require(value is None, f"captions.{name}.{key} must be null when no captions exist")
        if nonempty:
            require(bounds[0] <= summary["median"] <= summary["p95"] <= bounds[1], f"captions.{name} median/p95 must lie within occupied histogram bounds and be ordered")

    frequencies = mapping(field(captions, "token_frequencies", "captions"), "captions.token_frequencies")
    for token, frequency in frequencies.items():
        text(token, "captions.token_frequencies key")
        count(frequency, f"captions.token_frequencies[{token!r}]", minimum=1)
    vocabulary = count(field(captions, "vocabulary_size", "captions"), "captions.vocabulary_size")
    hapax = count(field(captions, "hapax_word_types", "captions"), "captions.hapax_word_types")
    require(len(frequencies) == vocabulary, "Token frequency keys must match captions.vocabulary_size")
    require(sum(frequencies.values()) == tokens, "Token frequencies must sum to captions.total_tokens")
    require(sum(value == 1 for value in frequencies.values()) == hapax, "Singleton token frequencies must match captions.hapax_word_types")
    share = field(captions, "hapax_token_share", "captions")
    if tokens:
        number(share, "captions.hapax_token_share")
        require(math.isclose(share, hapax / tokens, rel_tol=1e-12, abs_tol=1e-12), "captions.hapax_token_share must use all tokens as its denominator")
    else:
        require(share is None, "captions.hapax_token_share must be null when no tokens exist")

    repetition = mapping(field(captions, "repetition", "captions"), "captions.repetition")
    unique = count(field(repetition, "unique_nonempty_texts", "captions.repetition"), "captions.repetition.unique_nonempty_texts")
    extra = count(field(repetition, "global_extra_occurrences", "captions.repetition"), "captions.repetition.global_extra_occurrences")
    require(unique <= nonempty and (unique > 0) == (nonempty > 0) and extra == nonempty - unique, "Global repeated-text counts must reconcile to nonempty captions")
    scope_extras = 0
    for scope in ("within_images", "across_images"):
        path = f"captions.repetition.{scope}"
        value = mapping(field(repetition, scope, "captions.repetition"), path)
        distinct = count(field(value, "repeated_distinct_texts", path), f"{path}.repeated_distinct_texts")
        occurrences = count(field(value, "extra_occurrences", path), f"{path}.extra_occurrences")
        require(distinct <= unique and distinct <= occurrences and (distinct > 0) == (occurrences > 0), f"{path} repeated-text and extra counts are inconsistent")
        if scope == "across_images":
            require(occurrences <= distinct * max(0, images - 1), f"{path} extras exceed possible distinct-image appearances")
        scope_extras += occurrences
    require(scope_extras == extra, "Within-image and across-image extras must sum to global_extra_occurrences")
    across = repetition["across_images"]
    examples = rows(field(across, "examples", "captions.repetition.across_images"), "captions.repetition.across_images.examples")
    require(len(examples) <= across["repeated_distinct_texts"], "Across-image examples cannot exceed repeated distinct texts")
    seen_texts = set()
    example_across_extras = 0
    example_within_extras = 0
    for index, example in enumerate(examples):
        path = f"captions.repetition.across_images.examples[{index}]"
        caption = text(field(example, "text", path), f"{path}.text")
        require(caption not in seen_texts, "Across-image example texts must be unique")
        seen_texts.add(caption)
        image_count = count(field(example, "image_count", path), f"{path}.image_count", minimum=2)
        occurrences = count(field(example, "occurrence_count", path), f"{path}.occurrence_count")
        require(image_count <= images and image_count <= occurrences <= nonempty, f"{path} image and occurrence counts are inconsistent")
        example_across_extras += image_count - 1
        example_within_extras += occurrences - image_count
    require(example_across_extras <= across["extra_occurrences"], "Across-image example extras must not exceed the scope total")
    require(example_within_extras <= repetition["within_images"]["extra_occurrences"], "Within-image extras in shown examples must not exceed the scope total")


def _context(report: Mapping[str, Any]) -> str:
    if report["source"]["source_split"] != "train":
        raise ValueError("Training figures require an official training report")
    population = report["population"]
    selection = population["selection_method"]
    if population.get("full_split_verified"):
        if selection != "complete_official_training_split":
            raise ValueError("Verified full training requires complete split selection")
        scope = "complete split"
    elif selection == "source_order_prefix":
        scope = "nonrandom source-order prefix"
    else:
        scope = "selected records; full split unverified"
    images = report["dataset"]["total_records"]
    captions = report["captions"]["nonempty_caption_count"]
    return (
        f"Official TRAINING | {scope} | {images:,} images | "
        f"{captions:,} nonempty captions"
    )


def _title(figure, report: Mapping[str, Any], title: str) -> None:
    # Let constrained layout reserve space for the complete two-line heading.
    figure.suptitle(f"{title}\n{_context(report)}", fontsize=15, color=_INK)


def _empty(axis, message: str) -> None:
    axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center", color=_MUTED)


def _count_axis(axis, *, horizontal: bool = False) -> None:
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    count_axis = axis.xaxis if horizontal else axis.yaxis
    count_axis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    count_axis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.set_axisbelow(True)
    axis.grid(axis="x" if horizontal else "y")
    if horizontal:
        axis.set_xlim(left=0)
    else:
        axis.set_ylim(bottom=0)


def _bar_labels(axis, bars) -> None:
    axis.bar_label(bars, labels=[f"{bar.get_width():,.0f}" for bar in bars], padding=4, color=_INK, fontsize=9)
    axis.margins(x=0.18)


def create_class_distribution_figure(report: Mapping[str, Any]):
    """Plot all class counts by rank and readable largest/smallest examples."""
    _context(report)
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        figure = plt.figure(figsize=(14, 8), layout="constrained")
        try:
            grid = figure.add_gridspec(2, 2, width_ratios=(1.15, 1.4))
            rank_axis = figure.add_subplot(grid[:, 0])
            top_axis = figure.add_subplot(grid[0, 1])
            bottom_axis = figure.add_subplot(grid[1, 1])
            distribution = report["dataset"]["class_distribution"]
            ranked = sorted(distribution, key=lambda row: (-row["count"], row["class_id"]))
            rank_axis.plot(
                range(1, len(ranked) + 1), [row["count"] for row in ranked],
                color=_BLUE, linewidth=1.8, marker="o", markersize=3,
            )
            rank_axis.set(title=f"Every expected class ({len(ranked):,})", xlabel="Class rank (highest count first)", ylabel="Image count")
            _count_axis(rank_axis)
            rank_axis.set_ylim(top=max(1, max((row["count"] for row in ranked), default=0) * 1.15))
            if not ranked:
                _empty(rank_axis, "No class counts reported")
            for axis, rows, title, color in (
                (top_axis, ranked[:5], "Most represented classes", _BLUE),
                (bottom_axis, sorted(distribution, key=lambda row: (row["count"], row["class_id"]))[:5], "Least represented classes", _TEAL),
            ):
                positions = range(len(rows))
                bars = axis.barh(positions, [row["count"] for row in rows], color=color, height=0.65)
                axis.set_yticks(list(positions), labels=[textwrap.fill(row["class_name"], 27) for row in rows], fontsize=9)
                axis.invert_yaxis()
                axis.set(title=title, xlabel="Image count")
                _count_axis(axis, horizontal=True)
                _bar_labels(axis, bars)
            _title(figure, report, "Species representation")
            figure.supxlabel(
                "Counts before quality exclusions; selected examples break ties by source class ID.",
                fontsize=9, color=_MUTED,
            )
            return figure
        except BaseException:
            plt.close(figure)
            raise


def create_caption_length_figure(report: Mapping[str, Any]):
    """Draw precomputed discrete length and per-image coverage distributions."""
    _context(report)
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
        try:
            captions = report["captions"]
            for axis, distribution, value_field, summary, title, xlabel, color in (
                (axes[0], captions["word_length_distribution"], "word_count", captions["word_lengths"], "Caption word length", "Words per nonempty caption", _BLUE),
                (axes[1], captions["character_length_distribution"], "character_count", captions["character_lengths"], "Caption character length", "Unicode code points per nonempty caption", _TEAL),
            ):
                rows = sorted(distribution, key=lambda row: row[value_field])
                axis.bar([row[value_field] for row in rows], [row["caption_count"] for row in rows], width=0.9, color=color)
                for field, label, marker_color, linestyle in (
                    ("median", "Median", _INK, "--"), ("p95", "95th percentile", _GOLD, ":"),
                ):
                    value = summary[field]
                    if value is not None:
                        axis.axvline(value, color=marker_color, linestyle=linestyle, linewidth=1.6, label=f"{label}: {value:g}")
                if rows:
                    axis.legend(frameon=False, fontsize=8, loc="upper right")
                else:
                    _empty(axis, "No nonempty captions reported")
                axis.set(title=title, xlabel=xlabel, ylabel="Nonempty caption count")
                _count_axis(axis)
            rows = sorted(captions["per_image_nonempty_count_distribution"], key=lambda row: row["caption_count"])
            expected = captions["expected_captions_per_image"]
            if len(rows) == 1:
                row = rows[0]
                axes[2].set_axis_off()
                axes[2].set_title("Caption coverage")
                axes[2].text(0.5, 0.72, f"{row['caption_count']:,}", transform=axes[2].transAxes, ha="center", fontsize=44, color=_BLUE)
                axes[2].text(0.5, 0.59, "nonempty captions per image", transform=axes[2].transAxes, ha="center", fontsize=12, color=_INK)
                axes[2].text(0.5, 0.46, f"All {row['image_count']:,} analyzed images", transform=axes[2].transAxes, ha="center", fontsize=11, color=_INK)
                matches = row["caption_count"] == expected
                axes[2].text(0.5, 0.35, "Matches documented expectation" if matches else f"Documented expectation: {expected:g}", transform=axes[2].transAxes, ha="center", fontsize=10, color=_TEAL if matches else _GOLD)
                axes[2].text(0.5, 0.15, "Distinct descriptions may be fewer.\nSee repetition analysis.", transform=axes[2].transAxes, ha="center", fontsize=10, color=_MUTED)
            else:
                from matplotlib.ticker import MaxNLocator

                axes[2].bar([row["caption_count"] for row in rows], [row["image_count"] for row in rows], width=0.75, color=_BLUE)
                axes[2].axvline(expected, color=_GOLD, linestyle=":", linewidth=1.6, label=f"Documented expectation: {expected:g}")
                axes[2].legend(frameon=False, fontsize=8, loc="upper right")
                axes[2].set(title="Caption coverage per image", xlabel="Nonempty captions per image", ylabel="Image count")
                axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
                _count_axis(axes[2])
                if not rows:
                    _empty(axes[2], "No image coverage reported")
            _title(figure, report, "Caption lengths and coverage")
            return figure
        except BaseException:
            plt.close(figure)
            raise


def create_vocabulary_figure(report: Mapping[str, Any]):
    """Display reported token frequencies, retaining stopwords and rare words."""
    _context(report)
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(14, 8), layout="constrained")
        try:
            captions = report["captions"]
            ranked = sorted(captions["token_frequencies"].items(), key=lambda item: (-item[1], item[0]))
            selected = ranked[:20]
            positions = range(len(selected))
            bars = axes[0].barh(positions, [count for _, count in selected], color=_BLUE, height=0.72)
            axes[0].set_yticks(list(positions), labels=[token for token, _ in selected])
            axes[0].invert_yaxis()
            axes[0].set(title="Top 20 words (stopwords retained)", xlabel="Token occurrences")
            _count_axis(axes[0], horizontal=True)
            _bar_labels(axes[0], bars)
            if ranked:
                axes[1].loglog(range(1, len(ranked) + 1), [count for _, count in ranked], color=_TEAL, linewidth=1.5)
                axes[1].grid(which="major", alpha=0.8)
            else:
                _empty(axes[0], "No word tokens reported")
                _empty(axes[1], "No word tokens reported")
            axes[1].set(title="Vocabulary rank and frequency", xlabel="Word rank (log scale)", ylabel="Token occurrences (log scale)")
            share = captions["hapax_token_share"]
            share_text = "undefined (no tokens)" if share is None else f"{share:.2%} of all tokens"
            annotation = (
                f"{captions['vocabulary_size']:,} distinct word types\n"
                f"{captions['total_tokens']:,} token occurrences\n\n"
                f"{captions['hapax_word_types']:,} word types occur once\n"
                f"Their token share: {share_text}"
            )
            axes[1].text(0.98, 0.97, annotation, transform=axes[1].transAxes, ha="right", va="top", fontsize=10, color=_INK, bbox={"facecolor": "white", "edgecolor": "#d8e2ea", "alpha": 0.95, "pad": 9})
            _title(figure, report, "Caption vocabulary")
            return figure
        except BaseException:
            plt.close(figure)
            raise


def create_caption_repetition_figure(report: Mapping[str, Any]):
    """Show overlapping distinct-text scopes and texts shared across images."""
    _context(report)
    plt = _pyplot()
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": (1, 2.6)}, layout="constrained")
        try:
            repetition = report["captions"]["repetition"]
            values = [repetition["within_images"]["repeated_distinct_texts"], repetition["across_images"]["repeated_distinct_texts"]]
            bars = axes[0].barh(range(2), values, color=[_BLUE, _TEAL], height=0.55)
            axes[0].set_yticks(range(2), labels=["Within at least one image", "Across at least two images"])
            axes[0].invert_yaxis()
            axes[0].set(title="Distinct repeated texts by scope (scopes overlap)", xlabel="Distinct clean caption texts")
            _count_axis(axes[0], horizontal=True)
            _bar_labels(axes[0], bars)
            examples = sorted(repetition["across_images"]["examples"], key=lambda row: (-row["image_count"], row["text"]))[:5]
            positions = range(len(examples))
            bars = axes[1].barh(positions, [row["image_count"] for row in examples], color=_TEAL, height=0.6)
            axes[1].set_yticks(list(positions), labels=[textwrap.fill(row["text"], 66) for row in examples], fontsize=9)
            axes[1].invert_yaxis()
            axes[1].set(title="Clean caption texts shared by the most images (up to 5)", xlabel="Distinct images containing this exact text")
            _count_axis(axes[1], horizontal=True)
            _bar_labels(axes[1], bars)
            if not examples:
                _empty(axes[1], "No shared clean caption texts reported")
            _title(figure, report, "Caption text repetition")
            figure.supxlabel(
                f"{repetition['unique_nonempty_texts']:,} unique nonempty clean texts; "
                f"{repetition['global_extra_occurrences']:,} additional caption occurrences globally.\n"
                "Case-sensitive text comparison; scopes can overlap. Repeated text does not establish duplicate images or leakage.",
                fontsize=9, color=_MUTED,
            )
            return figure
        except BaseException:
            plt.close(figure)
            raise


def generate_training_figures(report: Mapping[str, Any], output_dir: str | Path) -> list[Path]:
    """Save four PNG figures plus vector PDFs, returning the PNG paths.

    Titles distinguish verified complete training from selected records. Test
    reports and incompatible schemas are rejected before filesystem changes.
    All eight files render in a temporary directory before publication. Existing
    target files have backups and are restored on publication failure; unrelated
    files are preserved. This is not a power-loss-atomic directory transaction.
    """
    _validate_training_report(report)
    destination = Path(output_dir)
    if destination.exists() and not destination.is_dir():
        raise ValueError("output_dir must be a directory")
    staging_parent = destination.absolute().parent
    while not staging_parent.exists():
        staging_parent = staging_parent.parent
    if not staging_parent.is_dir():
        raise ValueError("The existing output parent must be a directory")
    staging = Path(tempfile.mkdtemp(prefix=".eda-figures-", dir=staging_parent)).resolve()
    if staging.parent != staging_parent.resolve():
        raise RuntimeError("Unexpected temporary figure directory location")
    rendered = staging / "rendered"
    previous = staging / "previous"
    figures = (
        ("training_class_distribution", create_class_distribution_figure),
        ("training_caption_lengths", create_caption_length_figure),
        ("training_caption_vocabulary", create_vocabulary_figure),
        ("training_caption_repetition", create_caption_repetition_figure),
    )
    names = [f"{name}.{extension}" for name, _ in figures for extension in ("png", "pdf")]
    retain_backups = False
    try:
        rendered.mkdir()
        previous.mkdir()
        plt = _pyplot()
        for name, create in figures:
            before = set(plt.get_fignums())
            try:
                figure = create(report)
                figure.savefig(rendered / f"{name}.png", dpi=200, bbox_inches="tight")
                figure.savefig(rendered / f"{name}.pdf", bbox_inches="tight")
            finally:
                for identifier in set(plt.get_fignums()) - before:
                    plt.close(identifier)
        if any(not (rendered / name).is_file() or (rendered / name).stat().st_size == 0 for name in names):
            raise OSError("All eight figure files must render successfully before publication")

        # Preserve each existing target before replacing any member of the bundle.
        existing = set()
        for name in names:
            target = destination / name
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError(f"Figure target must be a regular file: {target}")
            if target.exists():
                shutil.copy2(target, previous / name)
                existing.add(name)
        created_directories = []
        parent = destination
        while not parent.exists():
            created_directories.append(parent)
            parent = parent.parent
        published = []
        try:
            destination.mkdir(parents=True, exist_ok=True)
            for name in names:
                os.replace(rendered / name, destination / name)
                published.append(name)
        except BaseException as publication_error:
            rollback_errors = []
            for name in reversed(published):
                try:
                    if name in existing:
                        os.replace(previous / name, destination / name)
                    else:
                        (destination / name).unlink()
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            for directory in created_directories:
                try:
                    directory.rmdir()  # Only remove empty directories created here.
                except OSError:
                    pass
            if rollback_errors:
                retain_backups = True
                raise RuntimeError(
                    f"Figure publication failed and rollback was incomplete; prior files remain in {previous}"
                ) from publication_error
            raise
        return [destination / f"{name}.png" for name, _ in figures]
    finally:
        if not retain_backups:
            # This exact, verified temporary directory was created by this call.
            shutil.rmtree(staging)


def main(argv: list[str] | None = None) -> None:
    """Regenerate figures from a JSON report without opening a data source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda/figures"))
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    paths = generate_training_figures(report, args.output_dir)
    print(f"Saved {len(paths)} training figures (PNG and PDF) to {args.output_dir}")


if __name__ == "__main__":
    main()
