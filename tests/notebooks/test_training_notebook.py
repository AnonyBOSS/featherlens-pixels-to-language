"""Execute the cloud notebook boundaries without network, pixels or GPU work."""
import ast
import inspect
import json
from pathlib import Path
import subprocess
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/pretrained/02_train_convnext.ipynb"


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _execute(cell_id, namespace):
    cell = next(cell for cell in _cells() if cell["id"] == cell_id)
    exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)


def _function(name):
    nodes = [node for cell in _cells() if cell["cell_type"] == "code"
             for node in ast.parse("".join(cell["source"])).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(nodes) == 1
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace[name]


def test_clean_notebook_cells_compile():
    cells = _cells()
    assert len({cell["id"] for cell in cells}) == len(cells)
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")


@pytest.mark.parametrize("commit", ["", "main", "HEAD", "a" * 39, "z" * 40])
def test_bootstrap_requires_published_sha_before_any_clone(tmp_path, commit):
    with pytest.raises(ValueError, match="40-character"):
        _function("checkout_repository")(tmp_path / "repo", "https://github.com/example/repo.git", commit)
    assert not (tmp_path / "repo").exists()


@pytest.mark.parametrize("url", ["https://user:secret@github.com/x/y.git", "https://github.com/x/y?token=secret", "ssh://git@github.com/x/y"])
def test_bootstrap_refuses_credential_bearing_urls(tmp_path, url):
    with pytest.raises(ValueError, match="credential-free HTTPS"):
        _function("checkout_repository")(tmp_path / "repo", url, "a" * 40)


def test_existing_git_checkout_must_match_and_be_clean(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("published\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Notebook test", "-c",
                    "user.email=test@example.invalid", "commit", "-m", "fixture"], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    checkout = _function("checkout_repository")
    assert checkout(repo, "https://github.com/example/repo.git", commit) == commit
    with pytest.raises(ValueError, match="published commit"):
        checkout(repo, "https://github.com/example/repo.git", "a" * 40)
    tracked.write_text("edited\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean published checkout"):
        checkout(repo, "https://github.com/example/repo.git", commit)


def test_metadata_zip_preserves_bytes_and_excludes_arbitrary_paths(tmp_path):
    archive = tmp_path / "handoff.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("nested/preparation.json", b'{"seed":42}\r\n')
        handle.writestr("nested/classification_manifest.jsonl", b'{}\n')
        handle.writestr("../escape.txt", b"excluded")
    destination = tmp_path / "prepared"
    extract = _function("extract_prepared_metadata")
    extract(archive, destination)
    assert (destination / "preparation.json").read_bytes() == b'{"seed":42}\r\n'
    assert (destination / "classification_manifest.jsonl").read_bytes() == b'{}\n'
    assert len(list(destination.iterdir())) == 2 and not (tmp_path / "escape.txt").exists()
    with pytest.raises(FileExistsError):
        extract(archive, destination)


def test_ambiguous_metadata_zip_fails_before_writing(tmp_path):
    archive = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name in ("one/preparation.json", "two/preparation.json", "classification_manifest.jsonl"):
            handle.writestr(name, "{}")
    with pytest.raises(ValueError, match="exactly one"):
        _function("extract_prepared_metadata")(archive, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("install,conflict", [(False, False), (True, False), (True, True)])
def test_dependency_setup_installs_only_missing_tracking_and_surfaces_conflicts(monkeypatch, capsys, install, conflict):
    import importlib.metadata
    import importlib.util
    import sys
    import src.training.runtime  # Import before the deliberately simulated missing dependency.
    import mlflow

    original_find = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "mlflow" else original_find(name))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1 if conflict and command[-1] == "check" else 0,
                                           stdout="fixture package conflict" if conflict else "compatible", stderr="")

    monkeypatch.setattr(subprocess, "run", run)  # Never invoke a network package install in a test.
    cell = next(cell for cell in _cells() if cell["id"] == "training-dependencies")
    tree = ast.parse("".join(cell["source"]))
    tree.body[0].value = ast.Constant(install)
    ast.fix_missing_locations(tree)
    namespace = {"importlib": importlib, "sys": sys, "subprocess": subprocess, "os": __import__("os"), "CLOUD": "colab"}
    if not install or conflict:
        with pytest.raises(RuntimeError, match="Review missing packages" if not install else "pip check found conflicts"):
            exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    else:
        exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    if install:
        assert commands == [[sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", "mlflow-skinny==3.17.2"],
                            [sys.executable, "-m", "pip", "check"]]
    else:
        assert commands == []
    if conflict:
        assert "fixture package conflict" in capsys.readouterr().out


KAGGLE_CONFLICTS = "\n".join([
    "bigframes 2.48.0 requires google-cloud-bigquery-storage, which is not installed.",
    "google-colab 1.0.0 has requirement jupyter-server==2.20.0, but you have jupyter-server 2.12.5.",
    "google-colab 1.0.0 has requirement pandas==2.2.3, but you have pandas 2.3.3.",
    "google-colab 1.0.0 has requirement requests==2.32.4, but you have requests 2.34.2.",
    "dopamine-rl 4.1.2 has requirement gym<=0.25.2, but you have gym 0.26.2.",
    "gradio 6.26.0 has requirement pillow<13, but you have pillow 13.0.0.",
    "moviepy 1.0.3 has requirement decorator<5, but you have decorator 5.3.1.",
])


@pytest.mark.parametrize("cloud,stdout,stderr,code,allowed", [
    ("kaggle", KAGGLE_CONFLICTS, "", 1, True),
    ("colab", KAGGLE_CONFLICTS, "", 1, False),
    ("kaggle", KAGGLE_CONFLICTS + "\ndatasets 5.1.0 has requirement numpy<2, but you have numpy 2.5.4.", "", 1, False),
    ("kaggle", KAGGLE_CONFLICTS + "\nunknown 1.0 requires missing-package, which is not installed.", "", 1, False),
    ("kaggle", "gradio 6.26.0 has requirement numpy<2, but you have numpy 2.5.4.", "", 1, False),
    ("kaggle", "bigframes broken output", "", 1, False),
    ("kaggle", "", "", 1, False),
    ("kaggle", KAGGLE_CONFLICTS, "pip process warning", 1, False),
    ("kaggle", KAGGLE_CONFLICTS, "", 2, False),
    ("kaggle", "No broken requirements found.", "pip process warning", 0, False),
])
def test_dependency_check_only_warns_for_reviewed_kaggle_pairs(monkeypatch, capsys, cloud, stdout, stderr, code, allowed):
    import importlib.metadata
    import importlib.util
    import os
    import sys

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", run)
    namespace = {"importlib": importlib, "os": os, "sys": sys, "subprocess": subprocess, "CLOUD": cloud}
    if allowed:
        _execute("training-dependencies", namespace)
        assert callable(namespace["validate_preflight_preparation"])
    else:
        with pytest.raises(RuntimeError, match="pip check"):
            _execute("training-dependencies", namespace)
    assert commands == [[sys.executable, "-m", "pip", "check"]]
    output = capsys.readouterr().out
    assert stdout in output and stderr in output
    if allowed:
        assert "WARNING" in output and "Kaggle" in output


def test_kaggle_warning_does_not_hide_required_import_failure(monkeypatch):
    import importlib.metadata
    import importlib.util
    import os
    import sys

    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 1, stdout=KAGGLE_CONFLICTS, stderr=""))
    original_import = importlib.import_module

    def import_module(name):
        if name == "mlflow":
            assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"
            assert os.environ["MLFLOW_DISABLE_AGENT_HINT"] == "1"
            assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"
            raise ImportError("fixture: installed MLflow is broken")
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    with pytest.raises(ImportError, match="installed MLflow is broken"):
        _execute("training-dependencies", {"importlib": importlib, "os": os, "sys": sys,
                                            "subprocess": subprocess, "CLOUD": "kaggle"})


@pytest.mark.parametrize("cloud,root", [("colab", "/content/drive/MyDrive/featherlens/experiments"),
                                         ("kaggle", "/kaggle/working/experiments")])
def test_acceptance_config_and_stable_run_path_use_real_api(cloud, root, monkeypatch):
    import datasets
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("Notebook checks must not fetch pixels or weights")

    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    monkeypatch.setattr(torch.hub, "download_url_to_file", forbidden)
    namespace = {"CLOUD": cloud, "REPO_DIR": ROOT, "Path": Path, "json": json}
    _execute("training-config", namespace)
    config = namespace["config"]
    assert config.architecture == "convnext_tiny"
    assert [(phase.mode, phase.epochs, phase.lr) for phase in config.phases] == [("frozen", 2, 0.001)]
    assert (config.batch_size, config.accumulation_steps, config.seed, config.num_workers) == (8, 2, 42, 0)
    assert config.amp == "float16" and config.freeze_bn_stats and config.weight_decay == 0.0001
    assert config.image_size == 224
    assert namespace["RUN_DIR"].as_posix() == root + "/convnext-frozen-acceptance-v1"
    assert namespace["RESUME"] is False
    # Re-execution does not invent a new experiment directory.
    _execute("training-config", namespace)
    assert namespace["RUN_DIR"].as_posix() == root + "/convnext-frozen-acceptance-v1"
    config = namespace["config"]
    from src.training.experiment import run_training
    call = next(node for cell in _cells() if cell["cell_type"] == "code"
                for node in ast.walk(ast.parse("".join(cell["source"])))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_training")
    namespace.update(PREPARED_DIR=ROOT / "unused", RESUME=True, DEVICE="cuda", TRACKING="mlflow")
    args = [eval(compile(ast.Expression(node), str(NOTEBOOK), "eval"), namespace) for node in call.args]
    kwargs = {item.arg: eval(compile(ast.Expression(item.value), str(NOTEBOOK), "eval"), namespace)
              for item in call.keywords}
    bound = inspect.signature(run_training).bind(*args, **kwargs)
    assert bound.arguments["config"] is config and bound.arguments["resume"] is True
    assert bound.arguments["device"] == "cuda" and bound.arguments["tracking"] == "mlflow"


def test_results_and_export_read_actual_artifacts_without_weights(tmp_path, capsys):
    summary = {"status": "completed", "epochs_completed": 2, "optimizer_updates": 599,
               "skipped_updates": 1, "best_epoch": 1, "best_metrics": {"top1_accuracy": 0.1},
               "latest_metrics": {"top1_accuracy": 0.09}, "elapsed_seconds": 12.5,
               "resolved_precision": "float16", "checkpoint_reload": {"status": "forward_verified"},
               "checkpoints": {"best": {"path": "best.pt", "sha256": "test"},
                               "latest": {"path": "latest.pt", "sha256": "test"}},
               "tracking": {"location": "mlruns", "mode": "mlflow", "run_id": "fixture-id", "status": "finished"},
               "failures": [], "runtime": {"fixture": True}}
    (tmp_path / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "evidence.zip").write_bytes(b"fixture evidence")
    namespace = {"RUN_DIR": tmp_path, "json": json, "Path": Path, "CLOUD": "kaggle"}
    _execute("training-results", namespace)
    _execute("training-export", namespace)
    output = capsys.readouterr().out
    assert '599' in output and 'fixture-id' in output and 'forward_verified' in output
    printed_paths = [ast.literal_eval(line) for line in output.splitlines()
                     if line.startswith(("{'checkpoint':", "{'tracking_path':"))]
    assert printed_paths[0]["path"] == str(tmp_path / "best.pt")
    assert printed_paths[2]["tracking_path"] == str(tmp_path / "mlruns")
    assert namespace["return_files"] == [tmp_path / "run_summary.json", tmp_path / "evidence.zip"]
