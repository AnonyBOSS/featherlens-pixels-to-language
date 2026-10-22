"""Render a complete PNG/PDF bundle before replacing any existing figures."""

from collections.abc import Callable, Sequence
import os
from pathlib import Path
import re
import shutil
import tempfile


def save_figure_bundle(figures: Sequence[tuple[str, Callable]], output_dir: str | Path) -> list[Path]:
    """Publish named PNG/PDF pairs, restoring prior files on caught failures.

    Callers validate their report before calling. Only the named targets are
    replaced; unrelated files survive. A failed rollback retains its backups
    and reports their path. Process termination and concurrent writers are not
    covered by this multi-file rollback.
    """
    import matplotlib.pyplot as plt

    figures = tuple(figures)
    stems = [stem for stem, _ in figures]
    if any(not isinstance(stem, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", stem) for stem in stems):
        raise ValueError("Figure stems must be simple lowercase filenames")
    if not stems or len(stems) != len(set(stems)):
        raise ValueError("Figure stems must be nonempty and unique")
    if any(not callable(create) for _, create in figures):
        raise ValueError("Each figure needs a callable builder")
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
    rendered, previous = staging / "rendered", staging / "previous"
    names = [f"{stem}.{extension}" for stem in stems for extension in ("png", "pdf")]
    retain_backups = False
    try:
        rendered.mkdir()
        previous.mkdir()
        for stem, create in figures:
            before = set(plt.get_fignums())
            try:
                figure = create()
                figure.savefig(rendered / f"{stem}.png", dpi=200, bbox_inches="tight")
                figure.savefig(rendered / f"{stem}.pdf", bbox_inches="tight")
            finally:
                for identifier in set(plt.get_fignums()) - before:
                    plt.close(identifier)
        if any(not (rendered / name).is_file() or (rendered / name).stat().st_size == 0 for name in names):
            raise OSError("All figure files must render successfully before publication")
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
                    directory.rmdir()
                except OSError:
                    pass
            if rollback_errors:
                retain_backups = True
                raise RuntimeError(
                    f"Figure publication failed and rollback was incomplete; prior files remain in {previous}"
                ) from publication_error
            raise
        return [destination / f"{stem}.png" for stem in stems]
    finally:
        if not retain_backups:
            # This exact temporary directory was created and verified above.
            shutil.rmtree(staging)
