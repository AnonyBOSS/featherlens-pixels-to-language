"""Training-only caption language and within-image diversity analysis."""

import argparse
from collections.abc import Mapping
from dataclasses import asdict
from itertools import islice
import math
from pathlib import Path

from src.data.schemas import SourceSpec
from .caption_details import LEXICON_VERSION, MATCHING_RULE, TERM_LEXICON, summarize_caption_details
from .explore import _runtime_versions
from .reporting import summarize_numbers, write_json_report


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _same(actual, expected, name):
    """Reconcile saved aggregates, allowing only float reduction roundoff."""
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{name} has inconsistent fields")
        for key, value in expected.items():
            _same(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"{name} disagrees with saved image observations")
    elif actual != expected or (isinstance(expected, int) and isinstance(actual, bool)):
        raise ValueError(f"{name} disagrees with saved image observations")


def _check_prevalence(item, captions, images, coverage, name):
    _same(item["caption_count"], captions, name + ".caption_count")
    _same(item["image_count"], images, name + ".image_count")
    for key, numerator, denominator in (
        ("caption_prevalence", captions, coverage["nonempty_caption_count"]),
        ("image_prevalence_among_captioned", images, coverage["images_with_nonempty_captions"]),
        ("image_prevalence_all", images, coverage["image_count"]),
    ):
        _same(item[key], numerator / denominator if denominator else None, name + "." + key)


def validate_caption_details_report(report):
    """Validate saved plotting inputs before creating any figures or files."""
    try:
        if report["schema_version"] != "caption-details-1.0":
            raise ValueError("Caption detail figures require schema 'caption-details-1.0'")
        source = report["source"]
        if source["source_split"] != "train":
            raise ValueError("Caption details accept only the official training source")
        if any(not isinstance(source[key], str) or not source[key].strip() for key in ("repo_id", "revision")):
            raise ValueError("Source repository and revision must be recorded")
        spec = SourceSpec(source["repo_id"], source["source_split"], source["revision"])
        names = report["class_names"]
        if not isinstance(names, list) or any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("Class names must be an ordered list of unique nonempty names")
        details = report["caption_details"]
        lexicon = details["lexicon"]
        if lexicon["version"] != LEXICON_VERSION or lexicon["matching_rule"] != MATCHING_RULE:
            raise ValueError("Unsupported caption lexicon version or matching rule")
        categories = lexicon["categories"]
        if categories != TERM_LEXICON:
            raise ValueError("Saved lexicon terms must match the declared lexicon version")
        if not isinstance(categories, Mapping) or set(categories) != {"color", "anatomy", "direction"}:
            raise ValueError("Caption lexicon categories are incomplete")
        for category, terms in categories.items():
            if not isinstance(terms, Mapping) or not terms:
                raise ValueError("Each lexicon category needs declared terms")
            for term, variants in terms.items():
                if not isinstance(term, str) or not term or not isinstance(variants, list) or not variants or any(not isinstance(value, str) or not value for value in variants):
                    raise ValueError("Lexicon terms require explicit nonempty variants")
        rows = details["images"]
        if not isinstance(rows, list):
            raise ValueError("Saved image observations must be a list")
        attempted = len(rows)
        population = report["population"]
        if _count(population["records_analyzed"], "records_analyzed") != attempted:
            raise ValueError("Population count disagrees with saved image observations")
        size = _count(population["indexed_source_size"], "indexed_source_size")
        reference = _count(population["reference_record_count"], "reference_record_count")
        if reference < 1 or population["random_sample"] is not False:
            raise ValueError("Caption population needs a positive reference and nonrandom scope")
        matches = attempted == reference
        if not isinstance(population["matches_reference_count"], bool) or population["matches_reference_count"] != matches:
            raise ValueError("Reference count flag is inconsistent")
        selection = population["selection_method"]
        if selection == "complete_official_training_split":
            if population["requested_limit"] is not None or population["source_exhausted"] is not True or attempted != size:
                raise ValueError("Complete analysis requires all indexed source rows")
            verified = matches
        elif selection == "source_order_prefix":
            limit = _count(population["requested_limit"], "requested_limit")
            if limit < 1 or attempted != min(size, limit) or population["source_exhausted"] is not False:
                raise ValueError("Prefix analysis requires exact source-order coverage")
            verified = False
        else:
            raise ValueError("Unsupported caption selection method")
        if not isinstance(population["full_split_verified"], bool) or population["full_split_verified"] != verified:
            raise ValueError("Full-split verification flag is inconsistent")

        coverage = dict.fromkeys(("image_count", "caption_line_count", "nonempty_caption_count", "token_bearing_caption_count", "images_with_nonempty_captions", "images_with_token_bearing_captions", "images_with_pairwise_overlap", "pair_count"), 0)
        coverage["image_count"] = attempted
        source_rows, identities = set(), set()
        category_caption_counts = dict.fromkeys(categories, 0)
        category_image_counts = dict.fromkeys(categories, 0)
        term_caption_counts = {category: dict.fromkeys(terms, 0) for category, terms in categories.items()}
        term_image_counts = {category: dict.fromkeys(terms, 0) for category, terms in categories.items()}
        grouped = [[] for _ in names]
        by_id = {}
        for row in rows:
            position = _count(row["source_row"], "source_row")
            identity = row["image_id"]
            if position in source_rows or identity in identities:
                raise ValueError("Duplicate image identity or source row")
            if identity != spec.image_id(position) or (row["source_repo"], row["source_revision"], row["source_split"]) != (spec.repo_id, spec.revision, "train"):
                raise ValueError("Image provenance must match the canonical training source")
            source_rows.add(position)
            identities.add(identity)
            by_id[identity] = row
            class_id = _count(row["class_id"], "class_id")
            if class_id >= len(names) or row["class_name"] != names[class_id]:
                raise ValueError("Image class metadata differs from the ordered class mapping")
            grouped[class_id].append(row)
            for key in ("caption_line_count", "nonempty_caption_count", "token_bearing_caption_count", "token_count", "vocabulary_size", "pair_count"):
                _count(row[key], key)
            nonempty, words = row["nonempty_caption_count"], row["token_bearing_caption_count"]
            tokens, vocabulary = row["token_count"], row["vocabulary_size"]
            if not row["caption_line_count"] >= nonempty >= words or not tokens >= words or not vocabulary <= tokens or (tokens > 0) != (vocabulary > 0) or (tokens > 0) != (words > 0):
                raise ValueError("Caption and token denominators are inconsistent")
            _same(row["mean_caption_words"], tokens / nonempty if nonempty else None, "mean_caption_words")
            _same(row["type_token_ratio"], vocabulary / tokens if tokens else None, "type_token_ratio")
            _same(row["pair_count"], words * (words - 1) // 2, "pair_count")
            if row["pair_count"]:
                _fraction(row["pairwise_jaccard_mean"], "pairwise_jaccard_mean")
            elif row["pairwise_jaccard_mean"] is not None:
                raise ValueError("Images without caption pairs must have null overlap")
            for key in ("caption_line_count", "nonempty_caption_count", "token_bearing_caption_count", "pair_count"):
                coverage[key] += row[key]
            coverage["images_with_nonempty_captions"] += bool(nonempty)
            coverage["images_with_token_bearing_captions"] += bool(words)
            coverage["images_with_pairwise_overlap"] += bool(row["pair_count"])
            if any(not isinstance(row[key], Mapping) or set(row[key]) != set(categories) for key in ("term_caption_counts", "category_caption_counts", "term_hits")):
                raise ValueError("Per-image term categories are inconsistent")
            expected_hits = []
            for category, terms in categories.items():
                counts = row["term_caption_counts"][category]
                if not isinstance(counts, Mapping) or not set(counts) <= set(terms):
                    raise ValueError("Unknown per-image lexicon term")
                for term, count in counts.items():
                    if not 0 < _count(count, term) <= nonempty:
                        raise ValueError("Term caption counts exceed their denominator")
                    term_caption_counts[category][term] += count
                    term_image_counts[category][term] += 1
                if row["term_hits"][category] != sorted(counts):
                    raise ValueError("Term hit labels disagree with their counts")
                count = _count(row["category_caption_counts"][category], category)
                if not max(counts.values(), default=0) <= count <= min(nonempty, sum(counts.values())):
                    raise ValueError("Category counts must represent unions of term mentions")
                category_caption_counts[category] += count
                category_image_counts[category] += bool(count)
                if count:
                    expected_hits.append(category)
            if sorted(row["category_hits"]) != sorted(expected_hits) or len(row["category_hits"]) != len(expected_hits):
                raise ValueError("Category hit labels disagree with their counts")
        if source_rows != set(range(attempted)):
            raise ValueError("Saved image rows must cover the exact source-order range")
        _same(details["coverage"], coverage, "coverage")
        metrics = ("mean_caption_words", "type_token_ratio", "pairwise_jaccard_mean", "token_count", "vocabulary_size")
        _same(details["overall"], {key: summarize_numbers(row[key] for row in rows if row[key] is not None) for key in metrics}, "overall")
        classes = details["classes"]
        if not isinstance(classes, list) or len(classes) != len(names):
            raise ValueError("Class summaries must retain the complete ordered mapping")
        for class_id, (name, group, item) in enumerate(zip(names, grouped, classes)):
            expected = {"class_id": class_id, "class_name": name, "image_count": len(group),
                        "images_with_nonempty_captions": sum(bool(row["nonempty_caption_count"]) for row in group),
                        "nonempty_caption_count": sum(row["nonempty_caption_count"] for row in group),
                        "category_image_counts": {category: sum(category in row["category_hits"] for row in group) for category in categories}}
            expected.update({key: summarize_numbers(row[key] for row in group if row[key] is not None) for key in metrics[:3]})
            _same(item, expected, "classes")
        prevalence = details["term_prevalence"]
        if not isinstance(prevalence, list) or len(prevalence) != len(categories) or {item["category"] for item in prevalence} != set(categories):
            raise ValueError("Category prevalence coverage is inconsistent")
        for item in prevalence:
            category = item["category"]
            _check_prevalence(item, category_caption_counts[category], category_image_counts[category], coverage, category)
            terms = item["terms"]
            if not isinstance(terms, list) or len(terms) != len(categories[category]) or {term["term"] for term in terms} != set(categories[category]):
                raise ValueError("Term prevalence coverage is inconsistent")
            for term in terms:
                label = term["term"]
                _check_prevalence(term, term_caption_counts[category][label], term_image_counts[category][label], coverage, label)
                if not isinstance(term["examples"], list) or len(term["examples"]) > 5:
                    raise ValueError("Term examples must be bounded")
                for example in term["examples"]:
                    if example["image_id"] not in by_id or label not in by_id[example["image_id"]]["term_hits"][category] or not isinstance(example["text"], str) or not example["text"].strip():
                        raise ValueError("Term examples need matching image references and text")
                    _count(example["caption_index"], "caption_index")
        for examples in details["overlap_examples"].values():
            if not isinstance(examples, list) or len(examples) > 5:
                raise ValueError("Overlap examples must be bounded")
            for pair in examples:
                if pair["image_id"] not in by_id or by_id[pair["image_id"]]["pair_count"] < 1:
                    raise ValueError("Overlap examples need a measured image with caption pairs")
                _fraction(pair["jaccard"], "example jaccard")
                if len(pair["caption_indices"]) != 2 or len(set(pair["caption_indices"])) != 2 or len(pair["texts"]) != 2:
                    raise ValueError("Overlap examples need two distinct caption references")
                for index in pair["caption_indices"]:
                    _count(index, "caption_index")
                if any(not isinstance(text, str) or not text.strip() for text in pair["texts"]):
                    raise ValueError("Overlap example text must be nonempty")
        return details
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Missing or malformed caption detail field: {error}") from error


def build_caption_details_report(source, *, limit=None, expected_records=5994):
    """Read only metadata/captions through the shared indexed training source."""
    if source.spec.source_split != "train":
        raise ValueError("Caption details accept only the official train split")
    if limit is not None and (_count(limit, "limit") < 1):
        raise ValueError("limit must be a positive integer")
    if _count(expected_records, "expected_records") < 1:
        raise ValueError("expected_records must be a positive integer")
    try:
        size = len(source.dataset)
    except (TypeError, AttributeError) as error:
        raise ValueError("Caption details require the shared indexed source") from error
    iterator = source.iter_records()
    details = summarize_caption_details(iterator if limit is None else islice(iterator, limit), class_names=source.class_names)
    count = details["coverage"]["image_count"]
    full, matches = limit is None, count == expected_records
    report = {
        "schema_version": "caption-details-1.0", "source": asdict(source.spec),
        "class_names": list(source.class_names), "runtime": _runtime_versions(),
        "population": {"selection_method": "complete_official_training_split" if full else "source_order_prefix",
                       "requested_limit": limit, "records_analyzed": count, "indexed_source_size": size,
                       "source_exhausted": full, "reference_record_count": expected_records,
                       "matches_reference_count": matches, "full_split_verified": full and matches, "random_sample": False},
        "methods": {"unit": "image-level observations; related captions are not independent images",
                    "terms": "declared lexical mentions, not verified visual attributes; categories can overlap",
                    "diversity": "pooled per-image type/token ratio; depends on token count",
                    "overlap": "mean unordered caption-pair token-set Jaccard per image; stopwords and repeats retained",
                    "class_comparison": "equal image weighting; descriptive summaries without significance tests"},
        "caption_details": details,
        "limitations": ["Lexicon coverage is explicit and incomplete; context, negation, and species names can affect mentions.",
                        "Word-set overlap ignores word order and meaning; it is not semantic accuracy or a quality score.",
                        "TTR depends on text length; no source cleaning, augmentation policy, or model vocabulary is selected.",
                        "No official test source is opened for this analysis; derived-partition summaries require verified manifests."],
    }
    validate_caption_details_report(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-records", type=int, default=5994)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.expected_records < 1:
        parser.error("--expected-records must be positive")
    from src.data.source import load_cub_split
    source = load_cub_split("train", streaming=False)
    report = build_caption_details_report(source, limit=args.limit, expected_records=args.expected_records)
    name = "train-caption-details-full.json" if args.limit is None else f"train-caption-details-prefix-{args.limit}.json"
    output = args.output or Path("outputs/eda/reports") / name
    write_json_report(report, output)
    print(f"Saved caption details for {report['population']['records_analyzed']:,} training images to {output}", flush=True)
    if args.figures_dir is not None:
        from src.visualization.caption_detail_plots import generate_caption_detail_figures
        paths = generate_caption_detail_figures(report, args.figures_dir)
        print(f"Saved {len(paths)} caption figure groups as PNG/PDF", flush=True)


if __name__ == "__main__":
    main()
