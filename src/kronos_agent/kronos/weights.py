"""Pre-registered Kronos weights: registry, verified download, offline strict loading.

``manifests/weights.json`` is written BEFORE any download with the sizes and sha256 values
published by the Hugging Face API (LFS object ids). Downloads are plain HTTPS GETs to
``huggingface.co/<repo>/resolve/<revision>/<file>`` whose redirects are followed only within
``huggingface.co`` / ``*.hf.co``; every file is hashed while streaming into a ``.part`` file that is
renamed into place only if size and sha256 match, and deleted otherwise or when the transfer
breaks off. Loading never contacts the Hub: local directories, ``strict=True``, ``.eval()`` on
both models, CPU. The predictor is built with :data:`PREDICTOR_CLIP`, and
:func:`inference_problems` tells whether loaded models still forecast as loaded (the forecast
configuration and the decision layer refuse models that do not).

A fine-tuned checkpoint (step 6, ``docs/RETRAINING.md`` 3.6) is loaded by the same
:func:`load_models` with ``checkpoint=``: its entry in the fine-tune registry
(:mod:`kronos_agent.finetune.registry`) names its files and hashes, which this module verifies
itself before loading (this file is hashed into the forecast ``config_id``; the registry module is
not). Its ``weights_sha256`` holds the checkpoint's own hashes (``finetuned/<id>/<file>``), so
its fans never share a configuration with the shipped model's, and ``LoadedModels.data_end_ms``
carries the end of its training data for the model-period rule of the decision layer.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ..hashing import sha256_file
from ..manifests import ManifestMismatch, VerifyReport
from ..models import MODEL_SPECS, ModelSpec, spec_for
from ..paths import manifests_dir, redact_home, weights_dir
from ..runtime import assert_hf_offline, require_configured
from .vendored import import_kronos

if TYPE_CHECKING:
    from ..finetune.registry import CheckpointRef, RegistryEntry

REGISTRY_NAME = "weights.json"
RESOLVE_URL = "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
POLICY_ALLOWED = "download_allowed"
POLICY_GATED = "needs_owner_approval"
# a fine-tuned checkpoint described as a repository of the local fine-tune registry
POLICY_LOCAL_CHECKPOINT = "local_checkpoint"
CHECKPOINT_REPO_PREFIX = "finetuned"
MAX_REDIRECTS = 5
REFERENCE_DEVICE = "cpu"
# KronosPredictor clamps the normalised context to [-clip, clip] (upstream's default, SPEC section
# 3); every forecast configuration records it (kronos_agent.data.forecast_store)
PREDICTOR_CLIP = 5


class Gate4Error(PermissionError):
    """Downloading this repository needs the owner's explicit approval (SPEC section 17, gate 4)."""


class WeightsError(RuntimeError):
    """Weights are absent, corrupted or inconsistent with the registry."""


@dataclass(frozen=True)
class WeightFile:
    name: str
    size: int
    sha256: str | None


@dataclass(frozen=True)
class WeightRepo:
    repo_id: str
    role: str
    revision: str
    policy: str
    files: tuple[WeightFile, ...]
    state_dict_numel: int | None = None
    expected_config: dict[str, Any] | None = None

    def file(self, name: str) -> WeightFile:
        for item in self.files:
            if item.name == name:
                return item
        raise KeyError(name)


@dataclass
class LoadedModels:
    """What :func:`load_models` returns. ``checkpoint`` is the fine-tuned checkpoint's id and
    ``data_end_ms`` the end of its training data (both ``None`` for the shipped weights)."""

    spec: ModelSpec
    tokenizer: Any
    model: Any
    predictor: Any
    weights_sha256: dict[str, str]
    load_seconds: float
    device: str = REFERENCE_DEVICE
    extra: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
    data_end_ms: int | None = None


def registry_path() -> Path:
    return manifests_dir() / REGISTRY_NAME


def load_registry(path: Path | None = None) -> dict[str, WeightRepo]:
    raw = json.loads((path or registry_path()).read_text(encoding="utf-8"))
    repos: dict[str, WeightRepo] = {}
    for item in raw["repos"]:
        repos[item["repo_id"]] = WeightRepo(
            repo_id=item["repo_id"],
            role=item["role"],
            revision=item["revision"],
            policy=item["policy"],
            files=tuple(
                WeightFile(f["name"], int(f["size"]), f.get("sha256")) for f in item["files"]
            ),
            state_dict_numel=item.get("state_dict_numel"),
            expected_config=item.get("expected_config"),
        )
    return repos


def repo_dir(repo_id: str, base: Path | None = None) -> Path:
    return (base or weights_dir()) / repo_id


def verify_repo(repo: WeightRepo, base: Path | None = None) -> VerifyReport:
    """Size and sha256 (or expected config content) of every registered file."""
    return verify_directory(repo_dir(repo.repo_id, base), repo)


def verify_directory(directory: Path, repo: WeightRepo) -> VerifyReport:
    """:func:`verify_repo` on an explicit directory; problems are labelled
    ``<repo_id>/<file>``, never with the directory's path."""
    report = VerifyReport()
    directory = Path(directory)
    for item in repo.files:
        path = directory / item.name
        if not path.is_file():
            report.problems.append(f"missing: {repo.repo_id}/{item.name}")
            continue
        report.checked += 1
        size = path.stat().st_size
        if size != item.size:
            report.problems.append(
                f"size mismatch: {repo.repo_id}/{item.name} ({size} != {item.size})"
            )
            continue
        if item.sha256 is not None:
            if sha256_file(path) != item.sha256:
                report.problems.append(f"sha256 mismatch: {repo.repo_id}/{item.name}")
        elif item.name == "config.json" and repo.expected_config is not None:
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                content = None
            if content != repo.expected_config:
                report.problems.append(f"config content mismatch: {repo.repo_id}/{item.name}")
    return report


def repos_for_model(
    name: str, registry: dict[str, WeightRepo] | None = None
) -> tuple[WeightRepo, WeightRepo]:
    registry = registry or load_registry()
    spec = spec_for(name)
    return registry[spec.repo_id], registry[spec.tokenizer_repo_id]


def verify_model(name: str, base: Path | None = None) -> VerifyReport:
    report = VerifyReport()
    for repo in repos_for_model(name):
        partial = verify_repo(repo, base)
        report.checked += partial.checked
        report.problems.extend(partial.problems)
    return report


def weights_available(name: str = "small", base: Path | None = None) -> bool:
    try:
        return verify_model(name, base).ok
    except (OSError, KeyError, ValueError):
        return False


def _host_allowed(host: str | None) -> bool:
    if not host:
        return False
    host = host.lower()
    return host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co"))


def _download_file(
    client: httpx.Client, url: httpx.URL, destination: Path, expected: WeightFile
) -> str:
    """Stream ``url`` into ``destination`` (via a ``.part`` file); return the sha256."""
    import hashlib

    part = destination.with_name(destination.name + ".part")
    for _hop in range(MAX_REDIRECTS + 1):
        try:
            with client.stream("GET", url) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        raise WeightsError(f"redirect without location for {expected.name}")
                    url = url.join(location)
                    # raw_host: url.host IDNA-decodes and raises for an invalid A-label
                    host = url.raw_host.decode("ascii")
                    if url.scheme != "https" or not _host_allowed(host):
                        raise WeightsError(f"refusing redirect to {host!r} for {expected.name}")
                    continue
                if response.status_code != 200:
                    raise WeightsError(f"HTTP {response.status_code} for {expected.name}")
                digest = hashlib.sha256()
                size = 0
                try:
                    with open(part, "wb") as handle:
                        for chunk in response.iter_bytes():
                            handle.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                            if size > expected.size:
                                break
                except BaseException:  # a dropped connection or Ctrl-C mid-stream: no partial file
                    part.unlink(missing_ok=True)
                    raise
                break
        except (UnicodeError, httpx.InvalidURL):
            # httpx builds the never-followed redirect request itself, before this code runs: it
            # reads the Location host (``url.host``, idna's UnicodeError for an invalid A-label)
            # and rebuilds a host-less absolute Location such as ``https:x`` (httpx.InvalidURL,
            # which is not an httpx.HTTPError)
            raise WeightsError(f"unparseable redirect location for {expected.name}") from None
    else:
        raise WeightsError(f"too many redirects for {expected.name}")
    if size != expected.size:
        part.unlink(missing_ok=True)
        raise WeightsError(f"size mismatch for {expected.name}: {size} != {expected.size}")
    actual = digest.hexdigest()
    if expected.sha256 is not None and actual != expected.sha256:
        part.unlink(missing_ok=True)
        raise WeightsError(f"sha256 mismatch for {expected.name}")
    os.replace(part, destination)
    return actual


def download_repo(
    repo: WeightRepo,
    *,
    approved_by_owner: bool = False,
    base: Path | None = None,
    transport: httpx.BaseTransport | None = None,
    log=print,
) -> list[Path]:
    """Download every registered file of ``repo`` that is not already present and verified."""
    if repo.policy != POLICY_ALLOWED and not approved_by_owner:
        raise Gate4Error(
            f"{repo.repo_id} needs the owner's approval before download (SPEC section 17, gate 4); "
            "rerun with --approved-by-owner once granted"
        )
    directory = repo_dir(repo.repo_id, base)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with httpx.Client(
        follow_redirects=False,
        trust_env=False,
        timeout=httpx.Timeout(120.0, connect=15.0),
        transport=transport,
        headers={"User-Agent": "kronos-agent/0.1 (weights fetch)"},
    ) as client:
        for item in repo.files:
            destination = directory / item.name
            if (
                destination.is_file()
                and verify_repo(
                    WeightRepo(
                        repo.repo_id,
                        repo.role,
                        repo.revision,
                        repo.policy,
                        (item,),
                        None,
                        repo.expected_config,
                    ),
                    base,
                ).ok
            ):
                log(f"present and verified: {repo.repo_id}/{item.name}")
                continue
            url = httpx.URL(
                RESOLVE_URL.format(repo_id=repo.repo_id, revision=repo.revision, filename=item.name)
            )
            log(f"downloading {repo.repo_id}/{item.name} ({item.size:,} bytes)")
            digest = _download_file(client, url, destination, item)
            if (
                item.sha256 is None
                and item.name == "config.json"
                and repo.expected_config is not None
            ):
                try:
                    content = json.loads(destination.read_text(encoding="utf-8"))
                except ValueError:
                    content = None
                if content != repo.expected_config:
                    destination.unlink(missing_ok=True)
                    raise WeightsError(f"config content mismatch for {repo.repo_id}")
            log(f"verified {repo.repo_id}/{item.name} sha256={digest}")
            written.append(destination)
    return written


def download_model(
    name: str, *, approved_by_owner: bool = False, base: Path | None = None, log=print
) -> list[Path]:
    written: list[Path] = []
    for repo in repos_for_model(name):
        written.extend(download_repo(repo, approved_by_owner=approved_by_owner, base=base, log=log))
    return written


def _state_dict_numel(module: Any) -> int:
    return int(sum(tensor.numel() for tensor in module.state_dict().values()))


def _check_config(directory: Path, repo: WeightRepo) -> None:
    if repo.expected_config is None:
        return
    content = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if content != repo.expected_config:
        raise WeightsError(f"{repo.repo_id}/config.json differs from the registered configuration")


def checkpoint_repo(entry: RegistryEntry) -> WeightRepo:
    """A registered fine-tuned checkpoint as a :class:`WeightRepo` (``finetuned/<id>``): its
    files with their sizes and sha256, the element count and configuration it must have."""
    return WeightRepo(
        repo_id=f"{CHECKPOINT_REPO_PREFIX}/{entry.checkpoint_id}",
        role="model",
        revision=entry.checkpoint_id,
        policy=POLICY_LOCAL_CHECKPOINT,
        files=tuple(WeightFile(item.name, item.size, item.sha256) for item in entry.files),
        state_dict_numel=entry.state_dict_numel,
        expected_config=dict(entry.expected_config),
    )


def checkpoint_dir_problems(directory: Path, name: str = "small") -> list[str]:
    """Why the Hugging Face-format ``directory`` is not a loadable ``name`` model (empty: it is).

    ``config.json`` missing, unparseable or other than the registered ``expected_config`` of the
    model's repository; ``model.safetensors`` missing; ``Kronos.from_pretrained(...,
    strict=True)`` failing (its message, the home directory redacted); a ``state_dict`` element
    count other than the model's. The model is loaded on the CPU and dropped. Raises only when
    the runtime is not configured or the vendored tree does not verify. The fine-tune registry
    runs it on a staged checkpoint before installing it.
    """
    require_configured()
    assert_hf_offline()
    spec = spec_for(name)
    model_repo = load_registry()[spec.repo_id]
    directory = Path(directory)
    problems: list[str] = []
    config_path = directory / "config.json"
    if not config_path.is_file():
        problems.append("config.json is missing")
    else:
        try:
            content = json.loads(config_path.read_text(encoding="utf-8"))
        except ValueError:
            problems.append("config.json is not valid JSON")
        else:
            if content != model_repo.expected_config:
                problems.append(
                    f"config.json differs from the registered configuration of {spec.repo_id}"
                )
    if not (directory / "model.safetensors").is_file():
        problems.append("model.safetensors is missing")
    if problems:
        return problems
    api = import_kronos()
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # the Hub mixin's "Loading weights" line
            loaded = api.Kronos.from_pretrained(str(directory), strict=True)
    except Exception as exc:  # noqa: BLE001 - any failure to load is the problem reported
        return [f"Kronos.from_pretrained(strict=True) fails: {redact_home(str(exc))}"]
    numel = _state_dict_numel(loaded)
    if numel != spec.state_dict_numel:
        problems.append(
            f"the state_dict has {numel} elements, Kronos-{name} has {spec.state_dict_numel}"
        )
    return problems


def _verified_checkpoint(
    name: str,
    ref: CheckpointRef,
    model_repo: WeightRepo,
    tokenizer_repo: WeightRepo,
    base: Path | None,
) -> tuple[RegistryEntry, WeightRepo, Path]:
    """The registry entry, repository and directory of the checkpoint ``ref``.

    Refused (:class:`WeightsError`): an entry that is not a ``name`` model with the registered
    configuration and element count, or that was trained with another tokenizer than the
    registered one; a tokenizer or checkpoint file whose size or sha256 does not verify. The
    registry's own errors (a broken chain, an unknown id) propagate.
    """
    from ..finetune import registry

    entry = registry.find_entry(ref)
    spec = spec_for(name)
    problems: list[str] = []
    if entry.model != name:
        problems.append(f"it is a {entry.model!r} model")
    if dict(entry.expected_config) != model_repo.expected_config:
        problems.append(f"its configuration is not the registered one of {model_repo.repo_id}")
    if entry.state_dict_numel != spec.state_dict_numel:
        problems.append(f"its state_dict has {entry.state_dict_numel} elements")
    registered = {item.name: item.sha256 for item in tokenizer_repo.files}
    if entry.tokenizer_repo_id != tokenizer_repo.repo_id or dict(entry.tokenizer_sha256) != (
        registered
    ):
        problems.append(f"it was trained with another tokenizer than {tokenizer_repo.repo_id}")
    if problems:
        raise WeightsError(
            f"checkpoint {entry.checkpoint_id} is not a Kronos-{name} checkpoint to load: "
            + "; ".join(problems)
        )
    repo = checkpoint_repo(entry)
    directory = registry.checkpoint_dir(ref)
    report = verify_repo(tokenizer_repo, base)
    report.problems.extend(verify_directory(directory, repo).problems)
    if not report.ok:
        raise WeightsError(
            f"weights of checkpoint {entry.checkpoint_id} not verified: "
            f"{ManifestMismatch(report.problems)}"
        )
    return entry, repo, directory


def load_models(
    name: str = "small",
    *,
    base: Path | None = None,
    device: str = REFERENCE_DEVICE,
    allow_non_reference_device: bool = False,
    checkpoint: CheckpointRef | None = None,
) -> LoadedModels:
    """Verify, then load tokenizer + model from local directories, in eval mode without
    gradients, on CPU, with a predictor clipping at :data:`PREDICTOR_CLIP`.

    ``checkpoint`` loads a registered fine-tuned checkpoint of ``name`` instead of the shipped
    model (``base`` still locates the shipped tokenizer; the checkpoint lives in its fine-tune
    root): its registry entry must describe a ``name`` model trained with the registered
    tokenizer, the tokenizer and checkpoint files must verify (size and sha256), and the model
    is loaded with ``strict=True`` and the same checks as the shipped one. Its
    ``weights_sha256`` are the entry's (:meth:`RegistryEntry.weights_sha256
    <kronos_agent.finetune.registry.RegistryEntry.weights_sha256>`), ``checkpoint`` and
    ``data_end_ms`` name it, and ``extra`` records the sha256 of its registry line; ``spec``
    stays the ``name`` spec.
    """
    require_configured()
    assert_hf_offline()
    if device != REFERENCE_DEVICE and not allow_non_reference_device:
        raise WeightsError(
            "comparable inference is CPU only; pass allow_non_reference_device=True for benchmarks"
        )
    spec = spec_for(name)
    model_repo, tokenizer_repo = repos_for_model(name)
    entry: RegistryEntry | None = None
    if checkpoint is None:
        report = verify_model(name, base)
        if not report.ok:
            raise WeightsError(
                f"weights for {name!r} not verified: {ManifestMismatch(report.problems)}"
            )
        model_dir = repo_dir(model_repo.repo_id, base)
    else:
        entry, model_repo, model_dir = _verified_checkpoint(
            name, checkpoint, model_repo, tokenizer_repo, base
        )
    api = import_kronos()
    import torch

    started = time.perf_counter()
    tokenizer_dir = repo_dir(tokenizer_repo.repo_id, base)
    _check_config(tokenizer_dir, tokenizer_repo)
    _check_config(model_dir, model_repo)
    tokenizer = api.KronosTokenizer.from_pretrained(str(tokenizer_dir), strict=True)
    model = api.Kronos.from_pretrained(str(model_dir), strict=True)
    for module in (tokenizer, model):
        module.eval()
        module.requires_grad_(False)
        module.to(device)
    tokenizer_numel = _state_dict_numel(tokenizer)
    model_numel = _state_dict_numel(model)
    if tokenizer_numel != spec.tokenizer_state_dict_numel:
        raise WeightsError(
            f"tokenizer state_dict has {tokenizer_numel} elements, expected {spec.tokenizer_state_dict_numel}"
        )
    if model_numel != spec.state_dict_numel:
        raise WeightsError(
            f"model state_dict has {model_numel} elements, expected {spec.state_dict_numel}"
        )
    predictor = api.KronosPredictor(
        model, tokenizer, device=device, max_context=spec.max_context, clip=PREDICTOR_CLIP
    )
    extra: dict[str, Any] = {"torch_version": torch.__version__}
    if entry is None:
        weights_sha256 = {
            f"{repo.repo_id}/{item.name}": item.sha256
            for repo in (model_repo, tokenizer_repo)
            for item in repo.files
            if item.sha256 is not None
        }
    else:
        weights_sha256 = entry.weights_sha256()
        extra["registry_line_sha256"] = entry.line_sha256
    return LoadedModels(
        spec=spec,
        tokenizer=tokenizer,
        model=model,
        predictor=predictor,
        weights_sha256=weights_sha256,
        load_seconds=time.perf_counter() - started,
        device=device,
        extra=extra,
        checkpoint=None if entry is None else entry.checkpoint_id,
        data_end_ms=None if entry is None else entry.data_end_ms,
    )


def _modules_of(models: LoadedModels) -> list[tuple[str, Any]]:
    """The tokenizer and model a fan runs, as ``models`` and its predictor hold them: each object
    once, absent and ``None`` ones skipped."""
    predictor = getattr(models, "predictor", None)
    found: list[tuple[str, Any]] = []
    for owner, name, label in (
        (models, "tokenizer", "the tokenizer"),
        (models, "model", "the model"),
        (predictor, "tokenizer", "the predictor's tokenizer"),
        (predictor, "model", "the predictor's model"),
    ):
        module = getattr(owner, name, None)
        if module is not None and all(module is not seen for _, seen in found):
            found.append((label, module))
    return found


def inference_problems(models: LoadedModels) -> list[str]:
    """Why a fan of ``models`` would differ from one of the models :func:`load_models` returns.

    The predictor must clip at :data:`PREDICTOR_CLIP`, read ``models.spec.max_context`` candles
    and run on ``models.device``; every torch module a fan runs (the tokenizer and the model, as
    ``models`` and as its predictor hold them) must be in eval mode, every submodule included
    (training mode turns dropout on and changes the cross-attention mask), with no parameter
    requiring gradients. Models without any module run no Kronos code (the registry description
    of :mod:`kronos_agent.kronos.batch`, the stand-in predictors of the tests): a clip, context
    length or device their predictor does not have is not asked for. An empty list: nothing to
    refuse.
    """
    problems: list[str] = []
    predictor = getattr(models, "predictor", None)
    modules = _modules_of(models)
    built: dict[str, Any] = {
        "clip": PREDICTOR_CLIP,
        "max_context": models.spec.max_context,
        "device": str(models.device),
    }
    for name, expected in built.items():
        actual = getattr(predictor, name, None)
        if actual is None and not modules:
            continue
        if name == "device" and actual is not None:
            actual = str(actual)  # a torch.device names the same device as its string
        if actual != expected:
            problems.append(
                f"the predictor's {name} is {actual!r}, load_models builds it with {expected!r}"
            )
    if modules:
        require_configured()
        import torch

        for label, module in modules:
            if not isinstance(module, torch.nn.Module):
                problems.append(f"{label} is not a torch module ({type(module).__name__})")
                continue
            if any(submodule.training for submodule in module.modules()):
                problems.append(f"{label} is in training mode")
            if any(parameter.requires_grad for parameter in module.parameters()):
                problems.append(f"{label} has parameters with gradients enabled")
    return problems


def registry_lines(registry: dict[str, WeightRepo] | None = None) -> Iterable[str]:
    """GNU sha256sum lines (relative to the repository root) for every file with a known sha256."""
    registry = registry or load_registry()
    for repo in sorted(registry.values(), key=lambda r: r.repo_id):
        for item in repo.files:
            if item.sha256 is not None:
                yield f"{item.sha256}  weights/{repo.repo_id}/{item.name}"


__all__ = [
    "CHECKPOINT_REPO_PREFIX",
    "MODEL_SPECS",
    "POLICY_LOCAL_CHECKPOINT",
    "PREDICTOR_CLIP",
    "Gate4Error",
    "LoadedModels",
    "WeightFile",
    "WeightRepo",
    "WeightsError",
    "checkpoint_dir_problems",
    "checkpoint_repo",
    "download_model",
    "download_repo",
    "inference_problems",
    "load_models",
    "load_registry",
    "registry_lines",
    "repo_dir",
    "verify_directory",
    "verify_model",
    "verify_repo",
    "weights_available",
]
