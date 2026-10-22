"""Caption-detail figures from a validated saved report, without source access.

Plots use the recorded term prevalence and per-image/class measurements. They
do not tokenize text, match lexicons, compute Jaccard, or refit any policy. Term
mentions are lexical evidence; TTR is length-sensitive; token-set overlap does
not measure semantic agreement. Each available per-image value has equal weight.
"""

from pathlib import Path
import textwrap

from .eda_plots import _BLUE, _GOLD, _INK, _MUTED, _STYLE, _TEAL, _count_axis, _empty, _pyplot
from .figure_export import save_figure_bundle


_CATEGORIES = (("color", "Color"), ("anatomy", "Anatomy"), ("direction", "Direction"))


def _validated(report):
    # The analysis entry point imports plotting lazily; keep the dependency one-way.
    from src.analysis.explore_caption_details import validate_caption_details_report

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
    """Show saved caption percentages, with one caption counted once per term."""
    details = _validated(report)
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
    """Plot saved per-image TTR, mean lexical overlap, and token-count dependence."""
    details = _validated(report)
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
    """Compare reported class means of per-image mean caption word counts."""
    details = _validated(report)
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
    """Validate first, then publish the three PNG/PDF groups as one bundle."""
    _validated(report)
    return save_figure_bundle(
        (
            ("training_caption_language", lambda: create_caption_language_figure(report)),
            ("training_caption_diversity", lambda: create_caption_diversity_figure(report)),
            ("training_class_caption_variation", lambda: create_class_caption_variation_figure(report)),
        ),
        output_dir,
    )
