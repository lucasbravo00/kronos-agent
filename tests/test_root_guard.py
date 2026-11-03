"""The test suite's guard of the real data, weights and report roots (``root_guard``, D-055 (6)).

Every default root is the test's own, in this process and in the processes it starts, and holds
no weights to load. Writes into each real root are refused before anything is written: through
the project's own write paths, through the native writers the session wraps (safetensors,
``torch.save``) and through the command line on stand-ins, ``data bingx`` on the fake exchange
and ``finetune timing`` on the measurement stand-ins (the command of the incident), which, left
on the default roots, write only under the test's own. The opt-ins read the real
roots and still cannot write them. The listing sees every change but Finder's, and inner pytest
sessions over temporary stand-in roots show the session side: a clean session passes and writes
its start listing; one whose child process wrote into a root, or that attempted a write outside
any test, fails although its tests pass; a write that the code swallowed still fails its test.

Every refusal here aims at a new probe name (``.root-guard-probe-<random>``) under a real root,
so that a guard that failed could only add a probe, never change a real file; a probe found after
its test is removed and fails the test (and the session, through the listing).
"""

from __future__ import annotations

import importlib
import io
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import finetune_fakes as ff
import numpy as np
import pytest
import root_guard
import safetensors.numpy
import safetensors.torch
import step2_fakes as fakes
import test_finetune_timing as timing_tests
import torch

from kronos_agent import cli, paths
from kronos_agent.cli import data as data_cli
from kronos_agent.data import fsutil
from kronos_agent.data.forecast_store import fan_path
from kronos_agent.data.store import StoreId, job_lock, read_meta
from kronos_agent.data.timeutil import HOUR_MS, MINUTE_MS, to_ms
from kronos_agent.data.windows import window_path
from kronos_agent.finetune import export_csv, registry, timing, train
from kronos_agent.kronos import weights
from kronos_agent.models import spec_for
from kronos_agent.report import figures, writer

TESTS_DIR = Path(__file__).resolve().parent
PROBE_PREFIX = ".root-guard-probe-"
SMALL = spec_for("small")
DECISION = datetime(2025, 3, 3, 12, tzinfo=UTC)
CHECKPOINT = "ks-btc-0123abcd-wf-2024-07"
CHECKPOINT_END = to_ms(datetime(2024, 6, 9, 16, tzinfo=UTC))
APPROVED_TIMING = ["finetune", "timing", "--approved-by-owner", "1.5"]  # the owner's figure, D-055
SETTLED_HOURS = 60

world = timing_tests.world  # the measurement stand-ins (FakeWorld) as a fixture of this module


@pytest.fixture
def real(request) -> root_guard.RealRoots:
    """The real roots of this session."""
    return request.config.pluginmanager.get_plugin(root_guard.PLUGIN_NAME).real


@pytest.fixture
def probe(real) -> Iterator[Callable[[str | Path], Path]]:
    """``probe(name)``: a new path right under the real root ``name``; ``probe(directory)``: one
    in ``directory`` (inside a real root, or reaching one through a view). Nothing is created;
    a probe that exists after the test (a write the guard let through) is removed, and the test
    fails."""
    made: list[Path] = []

    def new(where: str | Path) -> Path:
        parent = Path(real.code[where]) if isinstance(where, str) else Path(where)
        path = parent / f"{PROBE_PREFIX}{uuid.uuid4().hex}"
        made.append(path)
        return path

    yield new
    leaked = [path for path in made if os.path.lexists(path)]
    with root_guard.GUARD.disarmed():
        for path in leaked:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    assert not leaked, f"the guard let writes through: {leaked}"


def refused_by_the_guard(attempt: Callable[[], object], target: Path) -> list[str]:
    """Run ``attempt``, which the guard must stop (:class:`root_guard.RealRootWriteError`) with
    nothing written at ``target``; return its refusals, taken so that they fail no test."""
    with root_guard.GUARD.collect() as refused, pytest.raises(root_guard.RealRootWriteError):
        attempt()
    assert refused, "no write was refused"
    assert all(target.name in line for line in refused), refused
    assert not os.path.lexists(target)
    return refused


# == every default root is the test's own ==========================================================


def test_every_default_root_is_the_tests_own_and_holds_no_weights(isolated_roots):
    own = {name: getattr(isolated_roots, name).resolve() for name in root_guard.ROOT_ENV}
    assert root_guard.GUARD.armed and isolated_roots.views == frozenset()
    assert paths.data_dir() == own["data"]
    assert paths.weights_dir() == own["weights"]
    assert paths.reports_dir() == own["reports"]
    derived = {
        "data": [
            cli.data_root(),
            StoreId("bingx", "1h").csv_path(),
            window_path("bingx", DECISION),
            fan_path("bingx", "0" * 16, "forecast", DECISION),
            export_csv.fold_export_dir("0ee0d123", "synthetic"),
        ],
        "weights": [
            weights.repo_dir(SMALL.repo_id),
            registry.finetuned_dir(),
            registry.runs_dir(),
            registry.scratch_dir(),
        ],
        "reports": [
            cli.report_directory(),
            writer._report_dir(None),
            figures.figures_dir(),
            timing._report_dir(None),
        ],
    }
    for name, found in derived.items():
        for path in found:
            assert path.is_relative_to(own[name]), (name, path)
            assert root_guard.GUARD.locate(path) is None, path
    # a test that forgets its stand-ins cannot start a real measurement: no model to load
    assert not weights.weights_available("small")
    with pytest.raises(weights.WeightsError, match="not verified"):
        weights.load_models("small")


def test_a_tests_own_monkeypatch_undo_keeps_its_roots(isolated_roots, monkeypatch):
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(isolated_roots.data / "elsewhere"))
    monkeypatch.undo()
    assert paths.data_dir() == isolated_roots.data.resolve()


def test_the_processes_a_test_starts_inherit_its_roots(isolated_roots):
    code = (
        "from kronos_agent import paths\n"
        "print(paths.data_dir()); print(paths.weights_dir()); print(paths.reports_dir())\n"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        str(getattr(isolated_roots, name).resolve()) for name in ("data", "weights", "reports")
    ]
    # the training child's allow-list passes the data and weights roots (it writes no report)
    environment = train.child_environment(train.Compute())
    assert environment[paths.DATA_DIR_ENV] == str(isolated_roots.data)
    assert environment[paths.WEIGHTS_DIR_ENV] == str(isolated_roots.weights)


def test_this_session_lists_guards_and_isolates_the_repositorys_roots(request, real):
    plugin = request.config.pluginmanager.get_plugin(root_guard.PLUGIN_NAME)
    assert {name: real.protected[name] for name in root_guard.ROOT_ENV} == {
        name: paths.REPO_ROOT / name for name in root_guard.ROOT_ENV
    }
    assert root_guard.GUARD.armed
    assert plugin.before is not None and set(root_guard.ROOT_ENV) <= set(plugin.before)
    session = plugin.factory.base / root_guard.SESSION_ROOTS  # what wider fixtures resolve
    assert (session / "weights").is_dir()  # a view of the shipped weights, for loaded_small
    assert not (session / "weights" / registry.FINETUNED_DIRNAME).is_symlink()


# == writes into each real root are refused ========================================================


def _durable_write(target: Path, tmp_path: Path) -> None:
    fsutil.atomic_write_bytes(target / "store.csv", b"open_ms\n")


def _publish_once(target: Path, tmp_path: Path) -> None:
    fsutil.publish_new(target / "fan.npz", b"fan")


def _job_lock(target: Path, tmp_path: Path) -> None:
    with job_lock("raw/bingx/.lock", root=target):
        pass


def _run_directory(target: Path, tmp_path: Path) -> None:
    train.new_run_dir("guard-probe", root=target / registry.FINETUNED_DIRNAME)


def _checkpoint(target: Path, tmp_path: Path) -> None:
    payload = ff.checkpoint_payload(CHECKPOINT, data_end_ms=CHECKPOINT_END)
    source = ff.stand_in_source(tmp_path / "sources", payload)
    registry.install_checkpoint(
        source,
        payload=payload,
        check=lambda directory: [],
        root=target / registry.FINETUNED_DIRNAME,
    )


def _study_config(target: Path, tmp_path: Path) -> None:
    writer.record_config("guard-probe", {"probe": True}, directory=target)


WRITE_PATHS = [
    ("data", _durable_write),
    ("data", _publish_once),
    ("data", _job_lock),
    ("weights", _run_directory),
    ("weights", _checkpoint),
    ("reports", _study_config),
]


@pytest.mark.parametrize(
    "name, attempt",
    WRITE_PATHS,
    ids=[f"{name}-{attempt.__name__.strip('_')}" for name, attempt in WRITE_PATHS],
)
def test_the_projects_write_paths_cannot_write_a_real_root(name, attempt, probe, tmp_path):
    target = probe(name)
    refused_by_the_guard(lambda: attempt(target, tmp_path), target)


def _safetensors_file(target: Path) -> None:
    from safetensors.torch import save_file  # bound now: the session's wrapped writer

    save_file({"w": torch.zeros(2)}, str(target))


def _safetensors_model(target: Path) -> None:
    safetensors.torch.save_model(torch.nn.Linear(1, 1), str(target))


def _safetensors_numpy(target: Path) -> None:
    safetensors.numpy.save_file({"w": np.zeros(2)}, filename=target)


def _torch_save(target: Path) -> None:
    torch.save(torch.zeros(2), target)


def _torch_serialization_save(target: Path) -> None:
    torch.serialization.save(torch.zeros(2), str(target))


NATIVE_WRITES = [
    _safetensors_file,
    _safetensors_model,
    _safetensors_numpy,
    _torch_save,
    _torch_serialization_save,
]


@pytest.mark.parametrize(
    "attempt", NATIVE_WRITES, ids=[attempt.__name__.strip("_") for attempt in NATIVE_WRITES]
)
def test_native_writers_cannot_write_a_real_root(attempt, probe, tmp_path):
    """Native code writes without Python's open, so no audit event fires: the session wraps the
    writers (``root_guard.NATIVE_WRITERS``) and each refuses a real-root target before writing;
    the same writers still write the test's own files."""
    target = probe("weights")
    (refused,) = refused_by_the_guard(lambda: attempt(target), target)
    assert refused.split(" ")[0] in {
        f"{module}.{name}" for module, name, _, _ in root_guard.NATIVE_WRITERS
    }
    own = tmp_path / "own.bin"
    attempt(own)
    assert own.is_file() and own.stat().st_size > 0


def test_the_native_writers_are_wrapped_for_the_session():
    for module, name, _, _ in root_guard.NATIVE_WRITERS:
        function = getattr(importlib.import_module(module), name)
        assert getattr(function, root_guard.WRAPPED_ATTRIBUTE, None) is not None, (module, name)
    assert root_guard.GUARD.wrap_native_writers() == []  # once only
    buffer = io.BytesIO()  # a file object: opened by Python, seen by the audit hook
    torch.save(torch.zeros(2), buffer)
    assert buffer.getbuffer().nbytes > 0


@pytest.fixture
def exchange(monkeypatch) -> fakes.FakeBingX:
    """The fake BingX real market behind ``data bingx``: 60 settled hourly candles."""
    server_ms = fakes.utc_ms(2025, 6, 1, 7) + 2 * MINUTE_MS  # 7:02, the 6:00 candle settled
    start = server_ms - 2 * MINUTE_MS - SETTLED_HOURS * HOUR_MS
    candles = fakes.synthetic_candles(start, SETTLED_HOURS + 1, seed=9)  # the last one is open
    fake = fakes.FakeBingX(server_ms, {"1h": candles}, history_start_ms={"1h": start})
    monkeypatch.setattr(data_cli, "_bingx_client", lambda: fakes.bingx_client(fake)[0])
    return fake


def test_the_data_command_on_stand_ins_writes_only_under_the_tests_data_root(
    exchange, isolated_roots, real
):
    before = root_guard.list_roots(real.protected)
    assert cli.main(["data", "bingx", "--interval", "1h"]) == 0
    store = StoreId("bingx", "1h")
    assert store.csv_path().is_relative_to(isolated_roots.data.resolve())
    assert read_meta(store)["rows"] == SETTLED_HOURS
    assert root_guard.list_roots(real.protected) == before


def test_the_data_command_cannot_write_the_real_data_root(exchange, probe, monkeypatch):
    target = probe("data")
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(target))
    refused_by_the_guard(lambda: cli.main(["data", "bingx", "--interval", "1h"]), target)


def test_the_measurement_command_on_stand_ins_writes_only_under_the_tests_roots(
    world, isolated_roots, real, monkeypatch
):
    """The command of the incident (D-055 (6)), approved and on its stand-ins but with no root of
    its own: its plan, report and scratch land under the test's roots, no real root changes."""
    monkeypatch.setattr(timing, "real_steps", world.steps)
    before = root_guard.list_roots(real.protected)
    assert cli.main(APPROVED_TIMING) == 0
    directory = isolated_roots.reports / "finetune"
    report = timing.load_timing(directory=directory)
    assert report["status"] == "completed"
    assert (directory / f"{timing.TIMING_STEM}.config.json").is_file()
    assert registry.scratch_dir().is_relative_to(isolated_roots.weights.resolve())
    assert (registry.scratch_dir() / report["scratch"]).is_dir()
    assert root_guard.list_roots(real.protected) == before


@pytest.mark.parametrize("name", ["reports", "weights"])
def test_the_measurement_command_cannot_write_the_real_reports_or_weights_root(
    name, world, probe, monkeypatch
):
    monkeypatch.setattr(timing, "real_steps", world.steps)
    target = probe(name)
    monkeypatch.setenv(root_guard.ROOT_ENV[name], str(target))
    refused_by_the_guard(lambda: cli.main(APPROVED_TIMING), target)
    assert world.events == []  # refused before any measurement


# == the opt-ins read the real roots and cannot write them =========================================


@pytest.mark.weights
def test_the_weights_marker_reads_the_shipped_weights_and_cannot_write_them(
    isolated_roots, real, probe
):
    view = isolated_roots.weights
    assert paths.weights_dir() == view.resolve() and isolated_roots.views == {"weights"}
    shipped = sorted(
        entry.name
        for entry in Path(real.code["weights"]).iterdir()
        if entry.name not in root_guard.IGNORED_NAMES | {registry.FINETUNED_DIRNAME}
    )
    assert sorted(entry.name for entry in view.iterdir()) == shipped
    assert all((view / name).is_symlink() for name in shipped)
    assert weights.verify_model("small").ok  # the shipped weights, read through the view
    assert not registry.finetuned_dir().exists()  # the test's own fine-tune root
    assert root_guard.GUARD.locate(registry.finetuned_dir()) is None
    target = probe(weights.repo_dir(SMALL.repo_id))  # inside the real Kronos-small
    refused_by_the_guard(lambda: fsutil.atomic_write_bytes(target, b"x"), target)
    native = probe(weights.repo_dir(SMALL.repo_id))  # a native writer, through the view's link
    refused_by_the_guard(lambda: _safetensors_file(native), native)


@pytest.mark.real_roots("data", "reports")
def test_views_of_the_real_data_and_reports_roots_read_them_and_cannot_write_them(
    isolated_roots, real, probe
):
    assert isolated_roots.views == {"data", "reports"}
    linked: list[Path] = []
    for name in ("data", "reports"):
        view = getattr(isolated_roots, name)
        assert os.environ[root_guard.ROOT_ENV[name]] == str(view)
        source = Path(real.code[name])
        expected = sorted(
            entry.name
            for entry in (source.iterdir() if source.is_dir() else ())
            if entry.name not in root_guard.IGNORED_NAMES
        )
        assert sorted(entry.name for entry in view.iterdir()) == expected
        for entry in view.iterdir():
            assert entry.is_symlink() and entry.resolve() == (source / entry.name).resolve()
            if entry.is_dir():
                linked.append(entry)
    own = paths.data_dir() / f"{PROBE_PREFIX}{uuid.uuid4().hex}"  # new at the top: the test's
    fsutil.atomic_write_bytes(own / "export.csv", b"x")
    assert own.resolve().is_relative_to(isolated_roots.data.resolve())
    assert root_guard.GUARD.locate(own / "export.csv") is None
    if not linked:
        pytest.skip("no directory under the real data or reports root to write through")
    target = probe(linked[0])  # through a link, into the real root
    refused_by_the_guard(lambda: fsutil.atomic_write_bytes(target / "x.json", b"{}"), target)


@pytest.mark.real_roots("weights")
def test_a_view_of_the_real_weights_root_links_its_fine_tune_root_too(isolated_roots, real, probe):
    source = Path(real.code["weights"])
    expected = sorted(
        entry.name for entry in source.iterdir() if entry.name not in root_guard.IGNORED_NAMES
    )
    assert sorted(entry.name for entry in isolated_roots.weights.iterdir()) == expected
    if registry.FINETUNED_DIRNAME not in expected:
        pytest.skip("no fine-tune root under the real weights root")
    assert registry.finetuned_dir().is_symlink()
    target = probe(registry.finetuned_dir())
    refused_by_the_guard(lambda: fsutil.ensure_directory(target / ".runs"), target)


def test_the_real_roots_marker_takes_known_roots_only():
    mark = pytest.mark.real_roots
    assert root_guard.view_names([mark("data").mark, mark("reports", "data").mark]) == {
        "data",
        "reports",
    }
    for bad in (mark().mark, mark("logs").mark, mark("data", root="data").mark):
        with pytest.raises(ValueError, match="real_roots takes"):
            root_guard.view_names([bad])


# == the listing ===================================================================================

PAST_NS = 1_700_000_000 * 10**9


def stand_in_roots(tmp_path: Path) -> dict[str, Path]:
    """Three stand-in roots: ``data`` with two candle files, ``reports`` with a report and a link
    to ``data/raw``, ``weights`` absent; every mtime set in the past, so that any write shows."""
    roots = {name: tmp_path / name for name in ("data", "weights", "reports")}
    (roots["data"] / "raw").mkdir(parents=True)
    (roots["data"] / "raw" / "a.csv").write_text("a")
    (roots["data"] / "raw" / "b.csv").write_text("b")
    (roots["reports"] / "finetune").mkdir(parents=True)
    (roots["reports"] / "finetune" / "t.json").write_text("{}")
    (roots["reports"] / "link").symlink_to(roots["data"] / "raw", target_is_directory=True)
    for current, dirnames, filenames in os.walk(tmp_path):
        for name in [*dirnames, *filenames]:
            os.utime(Path(current) / name, ns=(PAST_NS, PAST_NS), follow_symlinks=False)
    for root in (roots["data"], roots["reports"]):
        os.utime(root, ns=(PAST_NS, PAST_NS))
    return roots


def test_the_listing_sees_every_change_but_finders(tmp_path):
    roots = stand_in_roots(tmp_path)
    before = root_guard.list_roots(roots)
    assert before["weights"] == root_guard.Entry("absent")
    assert before["data/raw/a.csv"] == root_guard.Entry("file", 1, PAST_NS)
    assert before["reports/link"].kind == "link" and "reports/link/a.csv" not in before
    assert root_guard.listing_changes(before, root_guard.list_roots(roots)) == []

    (roots["data"] / "raw" / "a.csv").write_text("aa")  # size
    os.utime(roots["data"] / "raw" / "b.csv", ns=(PAST_NS, PAST_NS + 1))  # mtime only
    (roots["data"] / "raw" / "new.csv").write_text("n")  # added
    (roots["data"] / "raw" / ".DS_Store").write_bytes(b"\0")  # Finder: explains raw's mtime
    (roots["data"] / "transient.tmp").write_text("t")  # written and removed: data's mtime
    (roots["data"] / "transient.tmp").unlink()
    (roots["reports"] / "finetune" / "t.json").unlink()  # removed (and finetune's mtime)
    roots["weights"].mkdir()  # a root that appears
    changes = root_guard.listing_changes(before, root_guard.list_roots(roots))
    assert [line.split(" (")[0] for line in changes] == [
        "changed  data",
        "changed  data/raw/a.csv",
        "changed  data/raw/b.csv",
        "added    data/raw/new.csv",
        "changed  reports/finetune",
        "removed  reports/finetune/t.json",
        "added    weights",
    ]
    assert "file, 1 bytes" in changes[1] and "file, 2 bytes" in changes[1]


def test_the_listing_file_has_one_line_per_entry(tmp_path):
    roots = stand_in_roots(tmp_path)
    text = root_guard.render_listing(
        root_guard.list_roots(roots), roots, listed_at=datetime(2026, 11, 3, 4, tzinfo=UTC)
    )
    lines = text.splitlines()
    assert lines[0] == (
        "# real roots of the kronos-agent test suite, listed at session start 2026-11-03T04:00:00Z"
    )
    body = [line.split("\t") for line in lines if not line.startswith("#")]
    assert [fields[0] for fields in body] == [
        "data",
        "data/raw",
        "data/raw/a.csv",
        "data/raw/b.csv",
        "reports",
        "reports/finetune",
        "reports/finetune/t.json",
        "reports/link",
        "weights",
    ]
    assert body[2] == ["data/raw/a.csv", "file", "1", str(PAST_NS)]
    assert body[1] == ["data/raw", "dir", "-", str(PAST_NS)]
    assert body[7][:2] == ["reports/link", "link"] and body[7][4] == str(roots["data"] / "raw")
    assert body[8] == ["weights", "absent", "-", "-"]


def test_a_listing_file_under_a_real_root_is_refused(tmp_path):
    roots = {"data": tmp_path / "data"}
    assert root_guard.listing_path(None, roots) is None
    assert (
        root_guard.listing_path(str(tmp_path / "baseline.txt"), roots)
        == (tmp_path / "baseline.txt").resolve()
    )
    with pytest.raises(pytest.UsageError, match="outside the real roots"):
        root_guard.listing_path(str(tmp_path / "data" / "baseline.txt"), roots)


# == the session side, in inner pytest sessions ====================================================

INNER_CONFTEST = '''
import os
from pathlib import Path

import pytest
import root_guard

FAKE = Path(os.environ["ROOT_GUARD_INNER_ROOTS"])
ROOTS = {name: FAKE / name for name in ("data", "weights", "reports")}


def pytest_configure(config):
    config.pluginmanager.register(
        root_guard.RootGuardPlugin(
            root_guard.RealRoots(protected=ROOTS, code=ROOTS),
            guard=root_guard.GUARD,
            listing=Path(os.environ["ROOT_GUARD_INNER_LISTING"]),
        ),
        root_guard.PLUGIN_NAME,
    )


@pytest.fixture(autouse=True)
def account():
    """The accounting of conftest.isolated_roots."""
    with root_guard.GUARD.collect() as refused:
        yield
    root_guard.fail_on_refusals(refused)
'''
INNER_PREAMBLE = """
import os
import subprocess
import sys
from pathlib import Path

import root_guard

FAKE = Path(os.environ["ROOT_GUARD_INNER_ROOTS"])
"""
CLEAN_TESTS = """
def test_reads_only():
    assert (FAKE / "data" / "kept.csv").read_text() == "kept"
"""
CHILD_TESTS = """
try:  # at collection: outside any test
    os.remove(FAKE / "weights" / "model.bin")
except root_guard.RealRootWriteError:
    pass


def test_a_child_process_writes_under_a_root():
    code = (
        "import pathlib, sys; root = pathlib.Path(sys.argv[1]); "
        "(root / 'data' / 'added.csv').write_text('new'); "
        "(root / 'reports' / 'kept.md').write_text('changed')"
    )
    subprocess.run([sys.executable, "-c", code, str(FAKE)], check=True)
"""
SWALLOWED_TESTS = """
def test_a_write_the_code_swallowed():
    try:
        (FAKE / "reports" / "new.md").write_text("x")
    except PermissionError:
        pass
"""


def run_inner_session(tmp_path: Path, tests: str) -> tuple[subprocess.CompletedProcess, Path]:
    """Run ``tests`` in a pytest session of its own over three stand-in roots under
    ``tmp_path / "roots"``, guarded by :class:`root_guard.RootGuardPlugin`; returns the finished
    process and the start listing's file."""
    fake = tmp_path / "roots"
    for name in ("data", "weights", "reports"):
        (fake / name).mkdir(parents=True)
    (fake / "data" / "kept.csv").write_text("kept")
    (fake / "reports" / "kept.md").write_text("kept")
    (fake / "weights" / "model.bin").write_bytes(b"\0" * 8)
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "pytest.ini").write_text("[pytest]\n")
    (inner / "conftest.py").write_text(INNER_CONFTEST)
    (inner / "test_inner.py").write_text(INNER_PREAMBLE + tests)
    listing = tmp_path / "listing.txt"
    environment = {
        **os.environ,
        "PYTHONPATH": str(TESTS_DIR),
        "PYTHONDONTWRITEBYTECODE": "1",
        "ROOT_GUARD_INNER_ROOTS": str(fake),
        "ROOT_GUARD_INNER_LISTING": str(listing),
    }
    environment.pop(root_guard.LISTING_ENV, None)
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(inner)],
        cwd=inner,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return completed, listing


SESSION_TITLE = "test root guard: the session wrote, or tried to write, under a real root"


def test_a_clean_session_passes_and_writes_its_start_listing(tmp_path):
    completed, listing = run_inner_session(tmp_path, CLEAN_TESTS)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert SESSION_TITLE not in completed.stdout
    lines = listing.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# real roots of the kronos-agent test suite")
    assert any(line.startswith("data/kept.csv\tfile\t4\t") for line in lines)


def test_a_session_whose_child_process_wrote_under_a_root_fails_although_its_tests_pass(
    tmp_path,
):
    completed, _ = run_inner_session(tmp_path, CHILD_TESTS)
    out = completed.stdout
    assert completed.returncode == 1, out + completed.stderr
    assert "1 passed" in out and "failed" not in out and "error" not in out
    assert out.index("1 passed") < out.index(SESSION_TITLE)  # printed last
    section = out[out.index(SESSION_TITLE) :]
    assert "added    data/added.csv" in section
    assert "changed  reports/kept.md" in section
    assert "refused  os.remove weights/model.bin (not taken by a test)" in section
    assert (tmp_path / "roots" / "weights" / "model.bin").read_bytes() == b"\0" * 8


def test_a_write_the_code_swallowed_still_fails_its_test(tmp_path):
    completed, _ = run_inner_session(tmp_path, SWALLOWED_TESTS)
    out = completed.stdout
    assert completed.returncode == 1, out + completed.stderr
    assert "1 passed, 1 error" in out
    assert "ERROR at teardown of test_a_write_the_code_swallowed" in out
    assert "each refused before it happened:\n  open reports/new.md\n" in out
    assert SESSION_TITLE not in out  # refused before it happened: no real root changed
    assert not (tmp_path / "roots" / "reports" / "new.md").exists()
