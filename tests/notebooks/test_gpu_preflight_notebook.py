"""Offline checks of executable notebook setup and the actual shared API."""

import ast
import inspect
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/pretrained/01_gpu_preflight.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code():
    return ["".join(cell["source"]) for cell in _notebook()["cells"] if cell["cell_type"] == "code"]


def _function(name):
    for source in _code():
        tree = ast.parse(source)
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if nodes:
            namespace = {}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"Missing executable helper {name}")


def test_notebook_is_clean_and_all_code_compiles():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    headings = ["".join(cell["source"]).splitlines()[0] for cell in notebook["cells"]
                if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("## ")]
    assert len(headings) == 12
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")


def test_repository_bootstrap_precedes_project_imports():
    nodes = ast.parse("\n\n".join(_code())).body
    project_import = next(index for index, node in enumerate(nodes)
                          if isinstance(node, ast.ImportFrom) and node.module.startswith("src."))
    checkout = next(index for index, node in enumerate(nodes)
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == "checkout_repository")
    path_setup = next(index for index, node in enumerate(nodes)
                      if isinstance(node, ast.If) and any(isinstance(child, ast.Call)
                      and isinstance(child.func, ast.Attribute) and child.func.attr == "insert"
                      and ast.unparse(child.func.value) == "sys.path" for child in ast.walk(node)))
    assert checkout < path_setup < project_import


@pytest.mark.parametrize("commit", ["", "main", "HEAD", "a" * 39, "a" * 41])
def test_repository_setup_rejects_unpinned_source_before_subprocess(tmp_path, commit):
    with pytest.raises(ValueError, match="40-character"):
        _function("checkout_repository")(tmp_path / "repo", "https://github.com/example/repo.git", commit)


@pytest.mark.parametrize("url", ["https://user:secret@github.com/x/y.git", "https://github.com/x/y?token=secret", "ssh://git@github.com/x/y"])
def test_repository_setup_rejects_credential_urls_before_subprocess(tmp_path, url):
    with pytest.raises(ValueError, match="credential-free HTTPS"):
        _function("checkout_repository")(tmp_path / "repo", url, "a" * 40)


def test_manual_checkout_records_real_pinned_git_provenance(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Notebook test", "-c", "user.email=test@example.invalid",
                    "commit", "--allow-empty", "-m", "test fixture"], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert _function("checkout_repository")(repo, "https://github.com/example/repo.git", commit) == commit
    with pytest.raises(ValueError, match="published commit"):
        _function("checkout_repository")(repo, "https://github.com/example/repo.git", "a" * 40)


def test_metadata_zip_extracts_only_known_entries_and_never_overwrites(tmp_path):
    import zipfile

    archive = tmp_path / "handoff.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("classification-v1/preparation.json", "{}")
        handle.writestr("classification-v1/classification_manifest.jsonl", "{}\n")
        handle.writestr("../escape.txt", "do not extract")
        handle.writestr("other/secret.txt", "do not extract")
    destination = tmp_path / "prepared"
    _function("extract_prepared_metadata")(archive, destination)
    assert sorted(path.name for path in destination.iterdir()) == ["classification_manifest.jsonl", "preparation.json"]
    assert not (tmp_path / "escape.txt").exists()
    with pytest.raises(FileExistsError):
        _function("extract_prepared_metadata")(archive, destination)


def test_metadata_zip_rejects_ambiguous_known_files_without_writes(tmp_path):
    import zipfile

    archive = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("one/preparation.json", "{}")
        handle.writestr("two/preparation.json", "{}")
        handle.writestr("classification_manifest.jsonl", "{}\n")
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="exactly one"):
        _function("extract_prepared_metadata")(archive, destination)
    assert not destination.exists()


def test_notebook_imports_and_arguments_bind_to_actual_shared_api(tmp_path, monkeypatch):
    # Imports/configuration are real and CPU-safe; neither source nor weights is fetched.
    import datasets
    import torch

    def network_forbidden(*args, **kwargs):
        raise AssertionError("Notebook validation must not download source or weights")

    monkeypatch.setattr(datasets, "load_dataset", network_forbidden)
    monkeypatch.setattr(torch.hub, "download_url_to_file", network_forbidden)
    from src.models.pretrained import ModelConfig
    from src.training.runtime import PreflightConfig, collect_runtime, run_gpu_preflight
    from src.training.weights import resolve_weight_info

    namespace = {"PREPARED_DIR": tmp_path / "prepared", "OUTPUT_DIR": tmp_path / "output", "PERSISTENT_DIR": None,
                 "OFFLINE": True, "BATCH_SIZE": 2, "NUM_WORKERS": 0,
                 "PreflightConfig": PreflightConfig}
    config_nodes = [node for source in _code() for node in ast.parse(source).body
                    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "config" for target in node.targets)]
    assert len(config_nodes) == 1
    exec(compile(ast.Module(body=config_nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    config = namespace["config"]
    inspect.signature(run_gpu_preflight).bind(config)
    assert config.batch_size == 2 and config.num_workers == 0 and config.offline
    assert tuple(config.architectures) == ("convnext_tiny", "efficientnet_b3", "regnet_y_3_2gf")
    for architecture in config.architectures:
        assert resolve_weight_info(ModelConfig(architecture=architecture))["weight_id"]
    runtime = collect_runtime(ROOT)
    assert isinstance(runtime, dict)
