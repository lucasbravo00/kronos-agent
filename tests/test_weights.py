"""Verified local weights: corruption detection, strict offline loading, pairing, gate 4 and the
verified download path (served by an ``httpx.MockTransport``; no network anywhere)."""

from __future__ import annotations

import json
import os
import shutil
import socket
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from kronos_agent.hashing import sha256_bytes
from kronos_agent.kronos import weights as weights_module
from kronos_agent.kronos.weights import (
    POLICY_GATED,
    Gate4Error,
    WeightFile,
    WeightsError,
    download_repo,
    load_models,
    load_registry,
    repo_dir,
    verify_repo,
)
from kronos_agent.models import spec_for
from kronos_agent.paths import weights_dir

SMALL = spec_for("small")

FAKE_REPO = "Example/Fake-Model"
FAKE_REVISION = "0123456789abcdef0123456789abcdef01234567"
CONFIG_BYTES = b'{"d_model": 8}\n'
WEIGHT_BYTES = bytes(range(256)) * 16
CDN_URL = "https://cdn-lfs.hf.co/repos/fake/model.safetensors"


@pytest.fixture
def staged_weights(tmp_path: Path) -> Path:
    """Temporary weights root: Kronos-small symlinked (untouched), tokenizer copied for mutation."""
    base = tmp_path / "weights"
    (base / "NeoQuasar").mkdir(parents=True)
    os.symlink(weights_dir() / SMALL.repo_id, base / SMALL.repo_id, target_is_directory=True)
    shutil.copytree(weights_dir() / SMALL.tokenizer_repo_id, base / SMALL.tokenizer_repo_id)
    return base


@pytest.mark.weights
def test_byte_flip_in_tokenizer_detected_before_load(staged_weights, monkeypatch):
    tokenizer_repo = load_registry()[SMALL.tokenizer_repo_id]
    assert verify_repo(tokenizer_repo, staged_weights).ok
    target = staged_weights / SMALL.tokenizer_repo_id / "model.safetensors"
    data = bytearray(target.read_bytes())
    data[-1] ^= 0x01  # a tensor byte, so the safetensors header stays valid and only the hash tells
    target.write_bytes(data)
    report = verify_repo(tokenizer_repo, staged_weights)
    assert report.problems == [f"sha256 mismatch: {SMALL.tokenizer_repo_id}/model.safetensors"]

    def never(*args, **kwargs):
        raise AssertionError("import_kronos must not run when the weights do not verify")

    monkeypatch.setattr(weights_module, "import_kronos", never)
    with pytest.raises(WeightsError, match="not verified"):
        load_models("small", base=staged_weights)


@pytest.mark.weights
def test_loaded_models_are_eval_cpu_and_frozen(loaded_small):
    import torch

    for module in (loaded_small.tokenizer, loaded_small.model):
        assert all(not submodule.training for submodule in module.modules())
        assert all(parameter.device.type == "cpu" for parameter in module.parameters())
        assert all(not parameter.requires_grad for parameter in module.parameters())
        assert all(buffer.device.type == "cpu" for buffer in module.buffers())
    assert loaded_small.device == "cpu"
    assert loaded_small.predictor.device == "cpu"
    assert loaded_small.predictor.max_context == SMALL.max_context == 512
    assert loaded_small.extra["torch_version"] == torch.__version__
    assert loaded_small.load_seconds > 0


@pytest.mark.weights
def test_state_dict_element_counts(loaded_small):
    model_numel = sum(tensor.numel() for tensor in loaded_small.model.state_dict().values())
    tokenizer_numel = sum(tensor.numel() for tensor in loaded_small.tokenizer.state_dict().values())
    assert model_numel == SMALL.state_dict_numel == 24_741_696
    assert tokenizer_numel == SMALL.tokenizer_state_dict_numel == 3_958_258
    # ``.parameters()`` undercounts: persistent buffers are stored in model.safetensors as well.
    assert sum(parameter.numel() for parameter in loaded_small.model.parameters()) < model_numel


@pytest.mark.weights
def test_config_json_equals_the_registry():
    registry = load_registry()
    for repo_id in (SMALL.repo_id, SMALL.tokenizer_repo_id):
        content = json.loads((repo_dir(repo_id) / "config.json").read_text(encoding="utf-8"))
        assert content == registry[repo_id].expected_config, repo_id


@pytest.mark.weights
def test_mismatched_tokenizer_directory_rejected(staged_weights):
    shutil.copyfile(
        weights_dir() / SMALL.repo_id / "config.json",
        staged_weights / SMALL.tokenizer_repo_id / "config.json",
    )
    report = verify_repo(load_registry()[SMALL.tokenizer_repo_id], staged_weights)
    assert len(report.problems) == 1
    assert report.problems[0].startswith(("size mismatch", "sha256 mismatch"))
    assert "config.json" in report.problems[0]
    with pytest.raises(WeightsError, match="not verified"):
        load_models("small", base=staged_weights)


@pytest.mark.weights
def test_load_never_contacts_the_hub(monkeypatch):
    import huggingface_hub.constants as hf_constants
    from huggingface_hub import hub_mixin

    calls: list[tuple] = []

    def boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("hf_hub_download must not be called for a local directory")

    monkeypatch.setattr(hub_mixin, "hf_hub_download", boom)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert hf_constants.HF_HUB_OFFLINE is True
    models = load_models("small")
    assert calls == []
    assert models.spec is SMALL
    assert set(models.weights_sha256) == {
        f"{SMALL.repo_id}/config.json",
        f"{SMALL.repo_id}/model.safetensors",
        f"{SMALL.tokenizer_repo_id}/config.json",
        f"{SMALL.tokenizer_repo_id}/model.safetensors",
    }


def test_socket_guard_blocks_connections_in_this_suite():
    # loopback discard port: a broken guard fails this test without any external connection
    with pytest.raises(AssertionError, match="network guard"):
        socket.create_connection(("localhost", 9), timeout=1)


def test_mps_refused_without_explicit_allowance():
    with pytest.raises(WeightsError, match="CPU only"):
        load_models("small", device="mps")


@pytest.mark.parametrize(
    "repo_id", ["NeoQuasar/Kronos-base", "NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k"]
)
def test_gate4_refuses_gated_repos_without_any_request(tmp_path, repo_id):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"never served")

    repo = load_registry()[repo_id]
    assert repo.policy == POLICY_GATED
    with pytest.raises(Gate4Error, match="gate 4"):
        download_repo(repo, base=tmp_path, transport=httpx.MockTransport(handler))
    assert requests == []
    assert not (tmp_path / repo_id).exists()


def _write_fake_registry(path: Path) -> dict:
    registry = {
        "schema_version": 1,
        "repos": [
            {
                "repo_id": FAKE_REPO,
                "role": "model",
                "revision": FAKE_REVISION,
                "policy": "download_allowed",
                "state_dict_numel": None,
                "expected_config": {"d_model": 8},
                "files": [
                    {
                        "name": "config.json",
                        "size": len(CONFIG_BYTES),
                        "sha256": sha256_bytes(CONFIG_BYTES),
                    },
                    {
                        "name": "model.safetensors",
                        "size": len(WEIGHT_BYTES),
                        "sha256": sha256_bytes(WEIGHT_BYTES),
                    },
                ],
            }
        ],
    }
    path.write_text(json.dumps(registry), encoding="utf-8")
    return load_registry(path)


class DyingStream(httpx.SyncByteStream):
    """Response body that delivers half of the weights, then breaks off with ``error``."""

    def __init__(self, error: BaseException):
        self.error = error

    def __iter__(self):
        yield WEIGHT_BYTES[: len(WEIGHT_BYTES) // 2]
        raise self.error


class FakeHub:
    """``MockTransport`` handler: config served directly, the LFS file through a redirect."""

    def __init__(
        self,
        *,
        redirect_to: str = CDN_URL,
        weight_bytes: bytes = WEIGHT_BYTES,
        weight_stream: httpx.SyncByteStream | None = None,
        config_bytes: bytes = CONFIG_BYTES,
    ):
        self.urls: list[str] = []
        self.redirect_to = redirect_to
        self.weight_bytes = weight_bytes
        self.weight_stream = weight_stream
        self.config_bytes = config_bytes

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        url = request.url
        if url.host == "huggingface.co" and url.path.endswith("/config.json"):
            return httpx.Response(200, content=self.config_bytes)
        if url.host == "huggingface.co" and url.path.endswith("/model.safetensors"):
            return httpx.Response(302, headers={"location": self.redirect_to})
        if url.host == "cdn-lfs.hf.co" and self.weight_stream is not None:
            return httpx.Response(200, stream=self.weight_stream)
        if url.host == "cdn-lfs.hf.co":
            return httpx.Response(200, content=self.weight_bytes)
        return httpx.Response(404)


def test_download_repo_verifies_size_and_sha256_and_renames_the_part_file(tmp_path, monkeypatch):
    repo = _write_fake_registry(tmp_path / "registry.json")[FAKE_REPO]
    hub = FakeHub()
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def recording_replace(src, dst):
        replaced.append((Path(src).name, Path(dst).name))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", recording_replace)
    base = tmp_path / "weights"
    log: list[str] = []
    written = download_repo(repo, base=base, transport=httpx.MockTransport(hub), log=log.append)
    directory = base / FAKE_REPO
    assert [path.name for path in written] == ["config.json", "model.safetensors"]
    assert (directory / "config.json").read_bytes() == CONFIG_BYTES
    assert (directory / "model.safetensors").read_bytes() == WEIGHT_BYTES
    assert list(directory.glob("*.part")) == []
    assert replaced == [
        ("config.json.part", "config.json"),
        ("model.safetensors.part", "model.safetensors"),
    ]
    assert hub.urls == [
        f"https://huggingface.co/{FAKE_REPO}/resolve/{FAKE_REVISION}/config.json",
        f"https://huggingface.co/{FAKE_REPO}/resolve/{FAKE_REVISION}/model.safetensors",
        CDN_URL,
    ]
    assert verify_repo(repo, base).ok
    assert any("verified" in line for line in log)
    # Present and verified files are not fetched again: zero requests on the second call.
    hub.urls.clear()
    assert download_repo(repo, base=base, transport=httpx.MockTransport(hub), log=log.append) == []
    assert hub.urls == []


def test_download_repo_rejects_a_truncated_file(tmp_path):
    repo = _write_fake_registry(tmp_path / "registry.json")[FAKE_REPO]
    hub = FakeHub(weight_bytes=WEIGHT_BYTES[:-1])
    base = tmp_path / "weights"
    with pytest.raises(WeightsError, match="size mismatch"):
        download_repo(repo, base=base, transport=httpx.MockTransport(hub), log=lambda _: None)
    directory = base / FAKE_REPO
    assert not (directory / "model.safetensors").exists()
    assert list(directory.glob("*.part")) == []


def test_download_repo_rejects_wrong_bytes_of_the_right_size(tmp_path):
    repo = _write_fake_registry(tmp_path / "registry.json")[FAKE_REPO]
    hub = FakeHub(weight_bytes=bytes([WEIGHT_BYTES[0] ^ 0xFF]) + WEIGHT_BYTES[1:])
    base = tmp_path / "weights"
    with pytest.raises(WeightsError, match="sha256 mismatch"):
        download_repo(repo, base=base, transport=httpx.MockTransport(hub), log=lambda _: None)
    directory = base / FAKE_REPO
    assert not (directory / "model.safetensors").exists()
    assert list(directory.glob("*.part")) == []


@pytest.mark.parametrize(
    "location",
    [
        "https://example.com/model.safetensors",
        "https://huggingface.co.example.org/model.safetensors",
        "http://huggingface.co/not-https/model.safetensors",
    ],
)
def test_download_repo_refuses_redirects_outside_the_hub(tmp_path, location):
    repo = _write_fake_registry(tmp_path / "registry.json")[FAKE_REPO]
    hub = FakeHub(redirect_to=location)
    base = tmp_path / "weights"
    with pytest.raises(WeightsError, match="refusing redirect"):
        download_repo(repo, base=base, transport=httpx.MockTransport(hub), log=lambda _: None)
    assert location not in hub.urls
    directory = base / FAKE_REPO
    assert not (directory / "model.safetensors").exists()
    assert list(directory.glob("*.part")) == []


@pytest.mark.parametrize(
    "location",
    [
        "https://xn--zzzzzzzzz.example.com/model.safetensors",
        "https://xn--zzzzzzzzz.hf.co/model.safetensors",
        "https:x",
        "mailto:a@b",
    ],
)
def test_download_repo_malformed_redirect_is_a_weights_error(tmp_path, location):
    # while building the redirect request httpx raises idna's UnicodeError (invalid A-label) or
    # httpx.InvalidURL (absolute URL without host), neither an httpx.HTTPError; both are mapped
    repo = _write_fake_registry(tmp_path / "registry.json")[FAKE_REPO]
    hub = FakeHub(redirect_to=location)
    base = tmp_path / "weights"
    with pytest.raises(WeightsError, match="unparseable redirect location"):
        download_repo(repo, base=base, transport=httpx.MockTransport(hub), log=lambda _: None)
    assert location not in hub.urls
    directory = base / FAKE_REPO
    assert not (directory / "model.safetensors").exists()
    assert list(directory.glob("*.part")) == []


def test_download_repo_unparseable_redirect_is_an_httpx_error(tmp_path):
    # httpx parses Location itself (even without following it), before the download code sees
    # the response; a Location it cannot parse at all arrives as an httpx.HTTPError, which
    # fetch-weights reports through its httpx.HTTPError branch.
    repo = _write_fake_registry(tmp_path / "registry.json")[FAKE_REPO]
    hub = FakeHub(redirect_to="http://[::1")
    base = tmp_path / "weights"
    with pytest.raises(httpx.HTTPError, match="location"):
        download_repo(repo, base=base, transport=httpx.MockTransport(hub), log=lambda _: None)
    directory = base / FAKE_REPO
    assert not (directory / "model.safetensors").exists()
    assert list(directory.glob("*.part")) == []


def test_download_repo_reports_http_errors(tmp_path):
    repo = _write_fake_registry(tmp_path / "registry.json")[FAKE_REPO]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(WeightsError, match="HTTP 404"):
        download_repo(
            repo,
            base=tmp_path / "weights",
            transport=httpx.MockTransport(handler),
            log=lambda _: None,
        )


@pytest.mark.parametrize(
    "error",
    [httpx.ReadError("connection reset while streaming"), KeyboardInterrupt()],
    ids=["transport-error", "ctrl-c"],
)
def test_download_repo_removes_the_part_file_when_the_stream_dies(tmp_path, error):
    repo = _write_fake_registry(tmp_path / "registry.json")[FAKE_REPO]
    hub = FakeHub(weight_stream=DyingStream(error))
    base = tmp_path / "weights"
    with pytest.raises(type(error)):
        download_repo(repo, base=base, transport=httpx.MockTransport(hub), log=lambda _: None)
    directory = base / FAKE_REPO
    assert (directory / "config.json").read_bytes() == CONFIG_BYTES  # the verified file stays
    assert not (directory / "model.safetensors").exists()
    assert list(directory.glob("*.part")) == []


@pytest.mark.parametrize(
    "served",
    [b'{"d_model": 9}\n', b"<html>oops</html>", b"\xff\xfe not utf-8"],
    ids=["other-config", "not-json", "not-utf-8"],
)
def test_download_repo_rejects_a_config_that_differs_by_content(tmp_path, served):
    # Gated repositories register config.json by content (sha256 null), checked after the rename.
    fake = _write_fake_registry(tmp_path / "registry.json")[FAKE_REPO]
    repo = replace(
        fake,
        files=(WeightFile("config.json", len(served), None), fake.file("model.safetensors")),
    )
    hub = FakeHub(config_bytes=served)
    base = tmp_path / "weights"
    with pytest.raises(WeightsError, match="config content mismatch"):
        download_repo(repo, base=base, transport=httpx.MockTransport(hub), log=lambda _: None)
    assert list((base / FAKE_REPO).iterdir()) == []
