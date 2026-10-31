"""Pre-registered Kronos weights: registry, verified download, offline strict loading.

``manifests/weights.json`` is written BEFORE any download with the sizes and sha256 values
published by the Hugging Face API (LFS object ids). Downloads are plain HTTPS GETs to
``huggingface.co/<repo>/resolve/<revision>/<file>`` whose redirects are followed only within
``huggingface.co`` / ``*.hf.co``; every file is hashed while streaming into a ``.part`` file that is
renamed into place only if size and sha256 match, and deleted otherwise or when the transfer
breaks off. Loading never contacts the Hub: local directories, ``strict=True``, ``.eval()`` on
both models, CPU.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..hashing import sha256_file
from ..manifests import ManifestMismatch, VerifyReport
from ..models import MODEL_SPECS, ModelSpec, spec_for
from ..paths import manifests_dir, weights_dir
from ..runtime import assert_hf_offline, require_configured
from .vendored import import_kronos

REGISTRY_NAME = "weights.json"
RESOLVE_URL = "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
POLICY_ALLOWED = "download_allowed"
POLICY_GATED = "needs_owner_approval"
MAX_REDIRECTS = 5
REFERENCE_DEVICE = "cpu"


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
    spec: ModelSpec
    tokenizer: Any
    model: Any
    predictor: Any
    weights_sha256: dict[str, str]
    load_seconds: float
    device: str = REFERENCE_DEVICE
    extra: dict[str, Any] = field(default_factory=dict)


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
    report = VerifyReport()
    directory = repo_dir(repo.repo_id, base)
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


def load_models(
    name: str = "small",
    *,
    base: Path | None = None,
    device: str = REFERENCE_DEVICE,
    allow_non_reference_device: bool = False,
) -> LoadedModels:
    """Verify, then load tokenizer + model from local directories, in eval mode, on CPU."""
    require_configured()
    assert_hf_offline()
    if device != REFERENCE_DEVICE and not allow_non_reference_device:
        raise WeightsError(
            "comparable inference is CPU only; pass allow_non_reference_device=True for benchmarks"
        )
    spec = spec_for(name)
    model_repo, tokenizer_repo = repos_for_model(name)
    report = verify_model(name, base)
    if not report.ok:
        raise WeightsError(
            f"weights for {name!r} not verified: {ManifestMismatch(report.problems)}"
        )
    api = import_kronos()
    import torch

    started = time.perf_counter()
    tokenizer_dir = repo_dir(tokenizer_repo.repo_id, base)
    model_dir = repo_dir(model_repo.repo_id, base)
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
        model, tokenizer, device=device, max_context=spec.max_context, clip=5
    )
    weights_sha256 = {
        f"{repo.repo_id}/{item.name}": item.sha256
        for repo in (model_repo, tokenizer_repo)
        for item in repo.files
        if item.sha256 is not None
    }
    return LoadedModels(
        spec=spec,
        tokenizer=tokenizer,
        model=model,
        predictor=predictor,
        weights_sha256=weights_sha256,
        load_seconds=time.perf_counter() - started,
        device=device,
        extra={"torch_version": torch.__version__},
    )


def registry_lines(registry: dict[str, WeightRepo] | None = None) -> Iterable[str]:
    """GNU sha256sum lines (relative to the repository root) for every file with a known sha256."""
    registry = registry or load_registry()
    for repo in sorted(registry.values(), key=lambda r: r.repo_id):
        for item in repo.files:
            if item.sha256 is not None:
                yield f"{item.sha256}  weights/{repo.repo_id}/{item.name}"


__all__ = [
    "MODEL_SPECS",
    "Gate4Error",
    "LoadedModels",
    "WeightFile",
    "WeightRepo",
    "WeightsError",
    "download_model",
    "download_repo",
    "load_models",
    "load_registry",
    "registry_lines",
    "repo_dir",
    "verify_model",
    "verify_repo",
    "weights_available",
]
