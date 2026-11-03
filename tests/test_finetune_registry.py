"""The fine-tune registry and the checkpoint loader (``finetune.registry``, ``kronos.weights``;
``docs/RETRAINING.md`` 3.6; test T21).

Installing appends a hash-chained canonical-JSON line; reading verifies every line and refuses a
torn, non-canonical, edited, removed, reordered or duplicated one, never repairing it. The
registry's last line has no line after it to carry its hash: it is protected only by the line
sha256 recorded where a checkpoint is used (``CheckpointRef.line_sha256``). Installs refuse a
registered id, a leftover directory, an invalid payload, files that differ from the payload or
fail the loadability check (the staging directory is removed) and a busy lock;
``registry_status`` reports what a fine-tune root holds. ``load_models(checkpoint=...)`` refuses
an entry of another model or tokenizer and any changed byte, size or ``config.json``.

The registry tests use stand-in checkpoints (``finetune_fakes.install_stand_in``) in temporary
fine-tune roots. The tests marked ``weights`` read the verified weights through symlinks in a
temporary weights root (``KRONOS_AGENT_WEIGHTS_DIR``, with ``KRONOS_AGENT_DATA_DIR`` also
temporary): the shipped Kronos-small goes through the child's output path
(``runner._save_selected``), the real loadability check, the registry and
``load_models(checkpoint=...)``, and its fan must be the shipped model's, bit for bit.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import finetune_fakes as ff
import pytest

from kronos_agent.data.forecast_store import config_id, forecast_config
from kronos_agent.data.fsutil import FILE_MODE, LockBusyError, exclusive_lock
from kronos_agent.data.timeutil import HOUR_MS, to_ms
from kronos_agent.finetune import registry, runner
from kronos_agent.finetune.registry import (
    CheckpointExists,
    CheckpointNotFound,
    CheckpointRef,
    RegistryChainError,
    RegistryError,
)
from kronos_agent.hashing import canonical_json, sha256_bytes, sha256_file
from kronos_agent.kronos import batch, weights
from kronos_agent.kronos.decision import forecast_window
from kronos_agent.kronos.vendored import import_kronos
from kronos_agent.kronos.weights import WeightsError, load_registry, repos_for_model
from kronos_agent.models import spec_for
from kronos_agent.params import load_params
from kronos_agent.paths import weights_dir

FIRST = "ks-btc-0123abcd-wf-2024-07"
SECOND = "ks-btc-0123abcd-wf-2024-08"
THIRD = "ks-btc-0123abcd-wf-2024-09"
FIRST_END = to_ms(datetime(2024, 6, 9, 16, tzinfo=UTC))
SECOND_END = to_ms(datetime(2024, 7, 10, 16, tzinfo=UTC))
THIRD_END = to_ms(datetime(2024, 8, 10, 16, tzinfo=UTC))
SMALL = spec_for("small")
PARAMS = load_params()
SMALL_PARAMS = replace(PARAMS, kronos=replace(PARAMS.kronos, trajectories=4))


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "ft"


@pytest.fixture
def two(root):
    """Two stand-in checkpoints registered in ``root``, in order."""
    first = ff.install_stand_in(root, FIRST, data_end_ms=FIRST_END)
    second = ff.install_stand_in(root, SECOND, data_end_ms=SECOND_END)
    return first, second


def registry_bytes(root: Path) -> bytes:
    return registry.registry_path(root).read_bytes()


def relined(entry, previous: bytes | None, **changes) -> bytes:
    """``entry``'s document with ``changes``, chained after the line ``previous``."""
    document = {**dict(entry.document), **changes}
    document["prev_sha256"] = (
        registry.GENESIS_SHA256 if previous is None else sha256_bytes(previous)
    )
    return canonical_json(document)


def listing(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


# -- installing and reading ----------------------------------------------------------------------


def test_install_appends_hash_chained_canonical_lines(root, two):
    first, second = two
    assert registry_bytes(root) == first.line + b"\n" + second.line + b"\n"
    assert first.document["prev_sha256"] == registry.GENESIS_SHA256
    assert second.document["prev_sha256"] == first.line_sha256 == sha256_bytes(first.line)
    assert registry.registry_head(root) == second.line_sha256
    assert [entry.line for entry in registry.read_registry(root)] == [first.line, second.line]
    _, tokenizer_repo = repos_for_model("small")
    for entry, end in ((first, FIRST_END), (second, SECOND_END)):
        assert canonical_json(dict(entry.document)) == entry.line
        assert tuple(sorted(entry.document)) == registry.ENTRY_KEYS
        assert len(registry.PAYLOAD_KEYS) == 33 and "activation_checkpointing" in entry.document
        assert entry.document["schema_version"] == registry.REGISTRY_SCHEMA_VERSION
        assert (entry.kind, entry.model, entry.data_end_ms) == ("fold", "small", end)
        assert entry.state_dict_numel == SMALL.state_dict_numel
        directory = registry.checkpoint_dir(CheckpointRef(entry.checkpoint_id, root))
        assert listing(directory) == list(registry.CHECKPOINT_FILES)  # README.md is not copied
        assert stat.S_IMODE(directory.stat().st_mode) == registry.DIRECTORY_MODE
        assert [item.name for item in entry.files] == list(registry.CHECKPOINT_FILES)
        for item in entry.files:
            path = directory / item.name
            assert (item.size, item.sha256) == (path.stat().st_size, sha256_file(path))
            assert stat.S_IMODE(path.stat().st_mode) == FILE_MODE
        assert entry.weights_sha256() == {
            **{f"finetuned/{entry.checkpoint_id}/{item.name}": item.sha256 for item in entry.files},
            **{
                f"{tokenizer_repo.repo_id}/{item.name}": item.sha256
                for item in tokenizer_repo.files
            },
        }
        reference = CheckpointRef(entry.checkpoint_id, root, line_sha256=entry.line_sha256)
        assert registry.find_entry(reference).line == entry.line
        assert registry.checkpoint_problems(reference) == []
    assert registry.registry_status(root) == {
        "entries": 2,
        "head": second.line_sha256,
        "chain_ok": True,
        "problem": None,
        "unregistered_dirs": [],
        "staging_leftovers": [],
    }


def test_an_absent_registry_is_empty_and_an_unknown_id_is_not_found(root):
    assert registry.read_registry(root) == ()
    assert registry.registry_head(root) == registry.GENESIS_SHA256
    assert registry.registry_status(root)["entries"] == 0
    assert not root.exists()  # reading creates nothing
    ff.install_stand_in(root, FIRST, data_end_ms=FIRST_END)
    with pytest.raises(CheckpointNotFound, match=SECOND):
        registry.find_entry(CheckpointRef(SECOND, root))


def test_checkpoint_references_and_ids_are_validated(tmp_path):
    longest = "ks-" + "a" * 61
    assert CheckpointRef(longest).checkpoint_id == longest
    for bad in ("KS-btc-x", "ks-", "ks-btc--x", "ks-btc-x-", "ks_btc", longest + "a", 7, None):
        with pytest.raises(ValueError, match="a checkpoint id matches"):
            CheckpointRef(bad)
    with pytest.raises(TypeError, match="root must be a path"):
        CheckpointRef(FIRST, root=3)
    assert CheckpointRef(FIRST, root=str(tmp_path)).root == tmp_path
    assert CheckpointRef(FIRST).resolved_root() == registry.finetuned_dir()
    assert CheckpointRef(FIRST, line_sha256="0" * 64).line_sha256 == "0" * 64
    for bad in ("A" * 64, "0" * 63, b"0" * 64, 0):
        with pytest.raises(ValueError, match="line_sha256"):
            CheckpointRef(FIRST, line_sha256=bad)
    assert registry.checkpoint_id_for("0123abcd", "wf-2025-06-r1") == (
        "ks-btc-0123abcd-wf-2025-06-r1"
    )
    for recipe8, label in (
        ("0123ABCD", "wf-2025-06"),
        ("0123abc", "wf-2025-06"),
        ("0123abcd", "WF-2025-06"),
        ("0123abcd", "wf_2025"),
        ("0123abcd", ""),
        ("0123abcd", "a" * 50),  # the id would pass 64 characters
    ):
        with pytest.raises(ValueError):
            registry.checkpoint_id_for(recipe8, label)


# -- a broken registry is refused, never repaired ------------------------------------------------


def torn(first, second) -> bytes:
    return first.line + b"\n" + second.line


def non_canonical(first, second) -> bytes:
    return first.line.replace(b'"model":"small"', b'"model": "small"') + b"\n"


def not_json(first, second) -> bytes:
    return b"{not json\n" + second.line + b"\n"


def edited_first(first, second) -> bytes:
    edited = relined(first, None, platform="another-platform")
    return edited + b"\n" + second.line + b"\n"


def removed_first(first, second) -> bytes:
    return second.line + b"\n"


def reordered(first, second) -> bytes:
    return second.line + b"\n" + first.line + b"\n"


def registered_twice(first, second) -> bytes:
    return first.line + b"\n" + second.line + b"\n" + relined(first, second.line) + b"\n"


def unknown_schema(first, second) -> bytes:
    return first.line + b"\n" + relined(second, first.line, schema_version=2) + b"\n"


def missing_key(first, second) -> bytes:
    document = {**dict(second.document)}
    del document["platform"]
    return first.line + b"\n" + canonical_json(document) + b"\n"


def invalid_value(first, second) -> bytes:
    return first.line + b"\n" + relined(second, first.line, threads=0) + b"\n"


BROKEN_CHAIN = "prev_sha256 is not the sha256 of the previous line"


@pytest.mark.parametrize(
    "mutation, message, verified",
    [
        (torn, "registry line 2: no LF at the end of the file (a torn write)", 1),
        (non_canonical, "registry line 1: not canonical JSON", 0),
        (not_json, "registry line 1: not JSON", 0),
        (edited_first, f"registry line 2: {BROKEN_CHAIN}", 1),
        (removed_first, f"registry line 1: {BROKEN_CHAIN}", 0),
        (reordered, f"registry line 1: {BROKEN_CHAIN}", 0),
        (registered_twice, f"registry line 3: checkpoint {FIRST} is registered twice", 2),
        (unknown_schema, "registry line 2: unknown schema_version 2", 1),
        (missing_key, "registry line 2: misses ['platform']", 1),
        (invalid_value, "registry line 2: threads must be a positive integer", 1),
    ],
)
def test_a_broken_registry_is_refused_and_never_repaired(root, two, mutation, message, verified):
    data = mutation(*two)
    registry.registry_path(root).write_bytes(data)
    with pytest.raises(RegistryChainError, match=re.escape(message)):
        registry.read_registry(root)
    for checkpoint_id in (FIRST, SECOND):  # a valid entry too: the whole registry is refused
        with pytest.raises(RegistryChainError, match=re.escape(message)):
            registry.find_entry(CheckpointRef(checkpoint_id, root))
    with pytest.raises(RegistryChainError, match=re.escape(message)):
        ff.install_stand_in(root, THIRD, data_end_ms=THIRD_END)
    assert registry_bytes(root) == data  # never repaired
    # no staging directory, no new checkpoint directory
    assert listing(root) == [registry.LOCK_NAME, FIRST, SECOND, registry.REGISTRY_NAME]
    status = registry.registry_status(root)
    assert status["chain_ok"] is False and message in status["problem"]
    assert status["entries"] == verified


def test_the_last_line_is_protected_only_by_its_recorded_line(root, two):
    """The chain cannot see an edit that keeps the last line a valid entry, nor its removal: the
    line sha256 recorded where the checkpoint is used does. An earlier data end would let the
    checkpoint forecast hours it was trained on, with the same files and config_id."""
    first, second = two
    anchored = CheckpointRef(SECOND, root, line_sha256=second.line_sha256)
    edited = ff.edit_last_line(root, **ff.earlier_data_end(second, 24))
    assert registry_bytes(root) == first.line + b"\n" + edited + b"\n"
    assert registry.read_registry(root)[-1].data_end_ms == SECOND_END - 24 * HOUR_MS
    assert registry.find_entry(CheckpointRef(SECOND, root)).line == edited  # not anchored
    assert registry.registry_status(root)["chain_ok"] is True
    with pytest.raises(RegistryChainError, match="changed after it was recorded"):
        registry.find_entry(anchored)
    with pytest.raises(RegistryChainError, match="changed after it was recorded"):
        registry.checkpoint_problems(anchored, registry.read_registry(root)[-1])
    with pytest.raises(RegistryChainError, match="changed after it was recorded"):
        batch.registered_weights_sha256("small", checkpoint=anchored)
    registry.registry_path(root).write_bytes(first.line + b"\n")  # the last line removed
    assert registry.registry_status(root)["chain_ok"] is True
    with pytest.raises(CheckpointNotFound, match=SECOND):
        registry.find_entry(anchored)
    unchanged = CheckpointRef(FIRST, root, line_sha256=first.line_sha256)
    assert registry.find_entry(unchanged).line == first.line


# -- installing: refusals ------------------------------------------------------------------------


def test_install_refuses_a_registered_id_and_a_leftover_directory(root, tmp_path, two):
    before = registry_bytes(root)
    with pytest.raises(CheckpointExists, match=f"checkpoint {FIRST} is registered already"):
        ff.install_stand_in(root, FIRST, data_end_ms=FIRST_END)
    (root / THIRD).mkdir()  # a kill between an install's rename and its append leaves this
    with pytest.raises(CheckpointExists, match="exists without a registry entry"):
        ff.install_stand_in(root, THIRD, data_end_ms=THIRD_END)
    fourth = "ks-btc-0123abcd-wf-2024-10"
    (root / fourth).symlink_to(tmp_path / "nowhere")
    with pytest.raises(CheckpointExists, match="exists without a registry entry"):
        ff.install_stand_in(root, fourth, data_end_ms=THIRD_END + 744 * HOUR_MS)
    assert registry_bytes(root) == before
    leftover = root / f"{registry.STAGING_PREFIX}{THIRD}-abc123"
    leftover.mkdir()
    status = registry.registry_status(root)
    assert status["unregistered_dirs"] == [THIRD] and status["staging_leftovers"] == [leftover.name]
    assert (status["entries"], status["chain_ok"]) == (2, True)
    with pytest.raises(CheckpointNotFound):  # nothing loads an unregistered directory
        registry.find_entry(CheckpointRef(THIRD, root))


def test_a_failed_install_leaves_no_staging_directory(root, tmp_path):
    payload = ff.checkpoint_payload(FIRST, data_end_ms=FIRST_END)
    source = ff.stand_in_source(tmp_path / "sources", payload)

    def install(check, **changes):
        return registry.install_checkpoint(
            source, payload={**payload, **changes}, check=check, root=root
        )

    with pytest.raises(RegistryError, match=f"checkpoint {FIRST} is not loadable: bad tensor"):
        install(lambda directory: ["bad tensor"])
    assert listing(root) == [registry.LOCK_NAME]

    class Interrupted(BaseException):  # a signal, say: not an Exception
        pass

    def interrupted(directory):
        raise Interrupted

    with pytest.raises(Interrupted):
        install(interrupted)
    assert listing(root) == [registry.LOCK_NAME]
    other = {**payload["expected_config"], "n_layers": 9}
    with pytest.raises(RegistryError, match="differs from the payload's expected_config"):
        install(lambda directory: [], expected_config=other)
    assert listing(root) == [registry.LOCK_NAME]
    seen = []

    def look(directory):
        seen.append((directory.name, listing(directory)))
        return []

    entry = install(look)
    ((name, files),) = seen
    assert name.startswith(f"{registry.STAGING_PREFIX}{FIRST}-")
    assert files == list(registry.CHECKPOINT_FILES)  # what the check sees is what is installed
    assert listing(root) == [registry.LOCK_NAME, FIRST, registry.REGISTRY_NAME]
    assert entry.files[1].sha256 == sha256_bytes(ff.STAND_IN_WEIGHTS)
    (source / "model.safetensors").unlink()
    with pytest.raises(RegistryError, match=f"{SECOND} to install has no model.safetensors"):
        install(lambda directory: [], **ff.checkpoint_payload(SECOND, data_end_ms=SECOND_END))
    assert listing(root) == [registry.LOCK_NAME, FIRST, registry.REGISTRY_NAME]


INNER_START = FIRST_END - 1344 * HOUR_MS


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"platform": None}, "platform must be a non-empty string"),
        ({"extra": 1}, r"unknown keys \['extra'\]"),
        ({"kind": "other"}, "kind must be one of"),
        ({"kind": "identity"}, "an identity entry has null"),
        ({"fold": None}, r"a fold entry needs \['fold'\]"),
        ({"inner_validation_range_ms": [INNER_START, FIRST_END - HOUR_MS]}, "must end at data_end"),
        ({"test_start_ms": FIRST_END - HOUR_MS}, "test months start at or after data_end_ms"),
        ({"data_end": "2024-06-09T17:00Z"}, "data_end must be '2024-06-09T16:00Z'"),
        ({"data_end_ms": FIRST_END + 1}, "on an exact hour"),
        ({"expected_config": {"n_layers": float("nan")}}, "finite numbers"),
        ({"activation_checkpointing": "no"}, "activation_checkpointing must be a boolean"),
        ({"train_csv_sha256": "x"}, "train_csv_sha256 must be a sha256 hex or null"),
        ({"checkpoint_id": "KS-btc"}, "a checkpoint id matches"),
    ],
)
def test_an_invalid_payload_is_refused_before_anything_is_written(root, tmp_path, changes, message):
    payload = ff.checkpoint_payload(FIRST, data_end_ms=FIRST_END)
    source = ff.stand_in_source(tmp_path / "sources", payload)
    with pytest.raises(RegistryError, match=message):
        registry.install_checkpoint(
            source, payload={**payload, **changes}, check=lambda directory: [], root=root
        )
    assert not root.exists()


def test_a_missing_payload_key_is_refused(root, tmp_path):
    payload = ff.checkpoint_payload(FIRST, data_end_ms=FIRST_END)
    source = ff.stand_in_source(tmp_path / "sources", payload)
    del payload["activation_checkpointing"]
    with pytest.raises(RegistryError, match=r"misses \['activation_checkpointing'\]"):
        registry.install_checkpoint(source, payload=payload, check=lambda d: [], root=root)
    assert not root.exists()


def test_a_busy_lock_refuses_an_install(root):
    with exclusive_lock(root / registry.LOCK_NAME), pytest.raises(LockBusyError):
        ff.install_stand_in(root, FIRST, data_end_ms=FIRST_END)
    assert listing(root) == [registry.LOCK_NAME]
    assert registry.read_registry(root) == ()


def test_checkpoint_problems_name_every_changed_file(root, two):
    first, _ = two
    reference = CheckpointRef(FIRST, root)
    directory = registry.checkpoint_dir(reference)
    label = f"finetuned/{FIRST}"
    weights_file = directory / "model.safetensors"
    original = weights_file.read_bytes()
    weights_file.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert registry.checkpoint_problems(reference) == [
        f"sha256 mismatch: {label}/model.safetensors"
    ]
    weights_file.write_bytes(original[:-1])
    assert registry.checkpoint_problems(reference) == [
        f"size mismatch: {label}/model.safetensors ({len(original) - 1} != {len(original)})"
    ]
    weights_file.write_bytes(original)
    config_file = directory / "config.json"
    config = json.loads(config_file.read_text(encoding="ascii"))
    config_file.write_text(json.dumps({**config, "n_layers": 9}), encoding="ascii")
    problems = registry.checkpoint_problems(reference)
    assert problems[-1] == f"config content mismatch: {label}/config.json"
    assert problems[0].startswith(("sha256 mismatch", "size mismatch"))
    config_file.unlink()
    assert registry.checkpoint_problems(reference) == [f"missing: {label}/config.json"]
    with pytest.raises(ValueError, match="is not the one of"):
        registry.checkpoint_problems(CheckpointRef(SECOND, root), first)


# -- the loader ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes, message",
    [
        (
            {
                "tokenizer": {
                    "repo_id": SMALL.tokenizer_repo_id,
                    "sha256": {"config.json": "a" * 64, "model.safetensors": "b" * 64},
                }
            },
            "it was trained with another tokenizer than NeoQuasar/Kronos-Tokenizer-base",
        ),
        ({"state_dict_numel": 1}, "its state_dict has 1 elements"),
        ({"model": "base"}, "it is a 'base' model"),
    ],
)
def test_the_loader_refuses_an_entry_of_another_model_or_tokenizer(
    root, monkeypatch, changes, message
):
    """Refused from the entry alone, before any file is read or anything loaded."""
    ff.install_stand_in(root, FIRST, data_end_ms=FIRST_END, **changes)

    def never(*args, **kwargs):
        raise AssertionError("nothing may be loaded for a refused entry")

    monkeypatch.setattr(weights, "import_kronos", never)
    monkeypatch.setattr(weights, "verify_repo", never)
    with pytest.raises(WeightsError, match=f"checkpoint {FIRST} is not a Kronos-small") as info:
        weights.load_models("small", checkpoint=CheckpointRef(FIRST, root))
    assert message in str(info.value)
    if changes.get("model") == "base":
        with pytest.raises(WeightsError, match="is a 'base' model, not 'small'"):
            batch.registered_weights_sha256("small", checkpoint=CheckpointRef(FIRST, root))


@pytest.fixture
def staged_weights(tmp_path, monkeypatch) -> Path:
    """A temporary weights root holding the verified shipped repositories as symlinks (read
    only), and a temporary data root."""
    real = weights_dir()
    staged = tmp_path / "weights"
    for repo_id in (SMALL.repo_id, SMALL.tokenizer_repo_id):
        (staged / repo_id).parent.mkdir(parents=True, exist_ok=True)
        (staged / repo_id).symlink_to(real / repo_id, target_is_directory=True)
    monkeypatch.setenv("KRONOS_AGENT_WEIGHTS_DIR", str(staged))
    monkeypatch.setenv("KRONOS_AGENT_DATA_DIR", str(tmp_path / "data"))
    return staged


@pytest.mark.weights
def test_the_childs_output_path_registers_and_loads_the_shipped_weights(
    loaded_small, staged_weights, tmp_path
):
    """The shipped Kronos-small saved as the child saves a selected state, checked by the real
    loadability check, registered and loaded by id: its bytes are the shipped ones, its fans the
    shipped model's bit for bit, and its config_id its own."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = SimpleNamespace(  # what _save_selected reads of the child's run specification
        predictor=SimpleNamespace(directory=weights_dir() / SMALL.repo_id), run_dir=run_dir
    )
    record = runner._save_selected(import_kronos(), spec, runner._snapshot(loaded_small.model))
    assert (record["dir"], record["source"]) == ("selected", "nll4_tf")
    selected = run_dir / "selected"
    shipped = weights_dir() / SMALL.repo_id / "model.safetensors"
    assert (selected / "model.safetensors").read_bytes() == shipped.read_bytes()
    config = json.loads((selected / "config.json").read_text(encoding="utf-8"))
    assert config == load_registry()[SMALL.repo_id].expected_config
    assert weights.checkpoint_dir_problems(selected) == []
    root = tmp_path / "ft"
    payload = ff.checkpoint_payload(FIRST, data_end_ms=FIRST_END)
    entry = registry.install_checkpoint(
        selected, payload=payload, check=weights.checkpoint_dir_problems, root=root
    )
    reference = CheckpointRef(FIRST, root, line_sha256=entry.line_sha256)
    loaded = weights.load_models("small", checkpoint=reference)
    assert (loaded.checkpoint, loaded.data_end_ms) == (FIRST, FIRST_END)
    assert loaded.weights_sha256 == entry.weights_sha256() != loaded_small.weights_sha256
    assert loaded.extra["registry_line_sha256"] == entry.line_sha256
    assert loaded.spec == loaded_small.spec and weights.inference_problems(loaded) == []
    kronos = PARAMS.kronos
    checkpoint_config = config_id(forecast_config(loaded, kronos))
    assert checkpoint_config != config_id(forecast_config(loaded_small, kronos))
    assert checkpoint_config == config_id(batch.registry_config(kronos, checkpoint=reference))
    window = batch.timing_window(SMALL_PARAMS)
    shipped_fan, checkpoint_fan = (
        forecast_window(models, window, SMALL_PARAMS, stream="timing", period_role="timing")
        for models in (loaded_small, loaded)
    )
    assert shipped_fan.paths.tobytes() == checkpoint_fan.paths.tobytes()


@pytest.mark.weights
def test_the_loader_refuses_a_changed_byte_size_or_config(staged_weights, root, monkeypatch):
    """Every file of the checkpoint is verified against its entry before anything is loaded; the
    shipped tokenizer verifies, so each refusal names only what changed."""
    ff.install_stand_in(root, FIRST, data_end_ms=FIRST_END)
    reference = CheckpointRef(FIRST, root)
    directory = registry.checkpoint_dir(reference)
    label = f"finetuned/{FIRST}"

    def never(*args, **kwargs):
        raise AssertionError("nothing may be loaded from a checkpoint that does not verify")

    monkeypatch.setattr(weights, "import_kronos", never)

    def refused() -> str:
        """The problems of the refusal: the checkpoint's only (the tokenizer verifies)."""
        with pytest.raises(
            WeightsError, match=f"weights of checkpoint {FIRST} not verified"
        ) as info:
            weights.load_models("small", checkpoint=reference)
        return str(info.value).split("not verified: ", 1)[1]

    weights_file = directory / "model.safetensors"
    original = weights_file.read_bytes()
    weights_file.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    assert refused() == f"sha256 mismatch: {label}/model.safetensors"
    weights_file.write_bytes(original[:-4])
    assert refused() == (
        f"size mismatch: {label}/model.safetensors ({len(original) - 4} != {len(original)})"
    )
    weights_file.write_bytes(original)
    config_file = directory / "config.json"
    text = config_file.read_text(encoding="ascii")
    assert '"n_layers": 8' in text
    config_file.write_text(text.replace('"n_layers": 8', '"n_layers": 9'), encoding="ascii")
    assert refused() == f"sha256 mismatch: {label}/config.json"
    os.remove(config_file)
    assert refused() == f"missing: {label}/config.json"
