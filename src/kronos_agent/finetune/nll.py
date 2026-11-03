"""NLL4-TF: the teacher-forced next-4-candle NLL in the inference geometry (``docs/RETRAINING.md``
3.3).

For a decision hour ``D`` the inputs are the 516 hourly candles opening in ``[D - 512 h, D + 4 h)``
as the inference window makes them (``data.windows.context_frame``, ``amount`` policy
``volume_x_mean_ohlc``): rows 0..511 are the context, rows 512..515 the targets ``D .. D + 3 h``.
The exporter writes them once per fold (``nll_inputs.npz``); this module never reads a store.

1. :func:`normalise`: float32, mean and population std of the 512 context rows only, ``(x -
   mean) / (std + 1e-5)``, clipped at ``+-clip``, applied to all 516 rows. The statistics are
   computed on a Fortran-ordered float32 copy of the context, the layout
   ``KronosPredictor.predict_batch`` reduces (K:620-631; a float32 sum depends on the order of
   its terms), so rows 0..511 are bit for bit the predictor's input.
2. Context tokens: ``tokenizer.encode`` of the 512 context rows, as inference encodes them.
   Target tokens: positions 512..515 of the encoding of all 516 rows (the encoder is causal,
   M:345-350).
3. Step ``j = 0..3``, as inference step ``j`` (K:420-442) with the true tokens in place of the
   sampled ones (teacher forcing): ``decode_s1`` on the tokens and time stamps ``[j, 512 + j)``,
   the last position scored at the true s1 of row ``512 + j``; then ``decode_s2`` of that context
   with the true s1 as a query of length 1, scored at the true s2.
4. NLL4-TF = the mean over decisions and steps of ``nll_s1 + nll_s2`` in nats per candle, at
   T = 1 without top-p: ``-log_softmax`` of the float32 logits, evaluated in float64, with sums
   and means in float64.

Both models must be in eval mode on the CPU: nothing is sampled, no dropout runs, no random number
is drawn, and the result is deterministic for a fixed thread count and batch size. The query of
``decode_s2`` has length 1 and every key lies in the past, so the eval-mode cross-attention (not
causal, M:387) cannot look ahead, and its rotary geometry is the inference one. With one decision
per batch the logits equal, bit for bit, those ``auto_regressive_inference`` computes when its
sampler returns the true tokens (tested). NLL compares checkpoints that share the tokenizer (SPEC
9.3); CRPS on fans remains the metric that compares models.

``torch`` and the vendored code are imported inside the functions, after
``runtime.configure_runtime()``.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.windows import naive_timestamps
from ..hashing import canonical_json, sha256_bytes
from ..kronos.features import FEATURE_COLUMNS
from ..runtime import require_configured

NLL_CONTEXT = 512
NLL_STEPS = 4
NLL_ROWS = NLL_CONTEXT + NLL_STEPS
NLL_BATCH_SIZE = 16
NLL_INPUTS_ARRAYS: tuple[str, ...] = ("decision_ms", "values", "meta")
NLL_INPUTS_SCHEMA_VERSION = 1
NLL_AMOUNT_POLICY = "volume_x_mean_ohlc"
NLL_META_KEYS: tuple[str, ...] = (
    "amount_policy",
    "columns",
    "context",
    "decisions",
    "grid_hours",
    "schema_version",
    "steps",
)
TIME_FEATURES = 5  # minute, hour, weekday, day, month (K:472-479)
NORMALISATION_EPSILON = 1e-5  # K:630, FB:126


class NllError(ValueError):
    """NLL4-TF inputs or models that cannot be scored as defined."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_only_copy(array: np.ndarray) -> np.ndarray:
    copy = np.array(array, copy=True)
    copy.flags.writeable = False
    return copy


@dataclass(frozen=True, eq=False)
class NllInputs:
    """The scored decisions: ``decision_ms`` int64 ``(n,)`` (ascending, unique, on the hour) and
    ``values`` float64 ``(n, 516, 6)`` in :data:`FEATURE_COLUMNS` order (rows 0..511 the context,
    512..515 the targets). Both are stored as read-only copies."""

    decision_ms: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        from ..data.timeutil import HOUR_MS

        decisions = self.decision_ms
        values = self.values
        if not isinstance(decisions, np.ndarray) or decisions.dtype != np.int64:
            raise NllError("decision_ms must be an int64 numpy array")
        if decisions.ndim != 1 or decisions.shape[0] < 1:
            raise NllError(
                f"decision_ms must be one-dimensional and not empty, got {decisions.shape}"
            )
        if np.any(np.diff(decisions) <= 0):
            raise NllError("decision_ms must be strictly increasing (ascending and unique)")
        if np.any(decisions % HOUR_MS != 0):
            raise NllError("every decision hour must be on an exact hour")
        expected = (decisions.shape[0], NLL_ROWS, len(FEATURE_COLUMNS))
        if not isinstance(values, np.ndarray) or values.dtype != np.float64:
            raise NllError("values must be a float64 numpy array")
        if values.shape != expected:
            raise NllError(f"values must have shape {expected}, got {values.shape}")
        if not np.isfinite(values).all():
            raise NllError("values must be finite")
        object.__setattr__(self, "decision_ms", _read_only_copy(decisions))
        object.__setattr__(self, "values", _read_only_copy(values))

    @property
    def decisions(self) -> int:
        return int(self.decision_ms.shape[0])

    def head(self, count: int) -> NllInputs:
        """The first ``count`` decisions (the timing command scores a fixed number of them)."""
        if not _is_int(count) or not 1 <= count <= self.decisions:
            raise NllError(f"count must be an integer in [1, {self.decisions}], got {count!r}")
        return NllInputs(self.decision_ms[:count].copy(), self.values[:count].copy())


@dataclass(frozen=True)
class TeacherForced:
    """Per decision and step: the logits of the last position and the true tokens they score."""

    s1_logits: Any  # torch float32 (n, 4, 2**s1_bits)
    s2_logits: Any  # torch float32 (n, 4, 2**s2_bits)
    s1_targets: Any  # torch int64 (n, 4)
    s2_targets: Any  # torch int64 (n, 4)


@dataclass(frozen=True)
class NllResult:
    """NLL4-TF in nats per candle (``value``), with its parts (float64)."""

    value: float
    per_decision: np.ndarray  # (n,): mean over the 4 steps of nll_s1 + nll_s2
    s1: np.ndarray  # (n, 4)
    s2: np.ndarray  # (n, 4)
    decisions: int
    batch_size: int
    threads: int
    seconds: float


def _meta_problems(meta: Any, decision_ms: np.ndarray) -> list[str]:
    """Why the ``meta`` member of an inputs file does not describe ``decision_ms``."""
    from ..data.timeutil import HOUR_MS

    if not isinstance(meta, np.ndarray) or meta.ndim != 0 or meta.dtype.kind != "U":
        return ["meta must be a 0-d unicode array"]
    text = str(meta[()])
    try:
        document = json.loads(text)
    except ValueError:
        return ["meta is not JSON"]
    if not isinstance(document, dict) or sorted(document) != sorted(NLL_META_KEYS):
        return [f"meta must hold exactly {sorted(NLL_META_KEYS)}"]
    if canonical_json(document) != text.encode("ascii", errors="replace"):
        return ["meta is not canonical JSON"]
    problems: list[str] = []
    expected = {
        "schema_version": NLL_INPUTS_SCHEMA_VERSION,
        "context": NLL_CONTEXT,
        "steps": NLL_STEPS,
        "columns": list(FEATURE_COLUMNS),
        "amount_policy": NLL_AMOUNT_POLICY,
        "decisions": int(decision_ms.shape[0]),
    }
    for key, value in expected.items():
        if document[key] != value or isinstance(document[key], bool):
            problems.append(f"meta {key} is {document[key]!r}, expected {value!r}")
    grid = document["grid_hours"]
    if not _is_int(grid) or grid < 1 or 24 % grid:
        problems.append(f"meta grid_hours must divide 24, got {grid!r}")
    elif np.any(decision_ms % (grid * HOUR_MS) != 0):
        problems.append(f"a decision hour is not on the {grid}-hour grid")
    return problems


def load_nll_inputs(path: Path, *, expected_sha256: str | None = None) -> NllInputs:
    """Read an ``nll_inputs.npz`` (interfaces 5.2): exactly the members ``decision_ms``,
    ``values`` and ``meta`` in that order, without pickles, and a ``meta`` that describes them;
    with ``expected_sha256`` the file's bytes must hash to it first. :class:`NllError` otherwise.
    """
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise NllError(f"cannot read the NLL4-TF inputs {path.name}: {exc.strerror}") from None
    if expected_sha256 is not None and sha256_bytes(data) != expected_sha256:
        raise NllError(f"sha256 mismatch: {path.name}")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = [info.filename for info in archive.infolist()]
        expected_members = [f"{name}.npy" for name in NLL_INPUTS_ARRAYS]
        if members != expected_members:
            raise NllError(f"{path.name} must hold exactly {expected_members}, got {members}")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in NLL_INPUTS_ARRAYS}
    except NllError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise NllError(f"{path.name} is not a valid NLL4-TF inputs file: {exc}") from None
    decision_ms = arrays["decision_ms"]
    if not isinstance(decision_ms, np.ndarray) or decision_ms.dtype != np.int64:
        raise NllError(f"{path.name}: decision_ms must be int64")
    problems = _meta_problems(arrays["meta"], decision_ms)
    if problems:
        raise NllError(f"{path.name}: " + "; ".join(problems))
    return NllInputs(decision_ms=decision_ms, values=arrays["values"])


def time_stamps(decision_ms: np.ndarray) -> np.ndarray:
    """float32 ``(n, 516, 5)``: the vendored ``calc_time_stamps`` (K:472-479) of the 516 opens of
    each decision, labelled by open time as the inference window labels them."""
    require_configured()
    from ..data.timeutil import HOUR_MS
    from ..kronos.vendored import import_kronos

    calc_time_stamps = import_kronos().module.kronos.calc_time_stamps
    decisions = np.asarray(decision_ms)
    if decisions.dtype != np.int64 or decisions.ndim != 1:
        raise NllError("decision_ms must be a one-dimensional int64 array")
    offsets = (np.arange(NLL_ROWS, dtype=np.int64) - NLL_CONTEXT) * HOUR_MS
    stamps = np.empty((decisions.shape[0], NLL_ROWS, TIME_FEATURES), dtype=np.float32)
    for index, decision in enumerate(decisions):
        frame = calc_time_stamps(naive_timestamps(decision + offsets))
        stamps[index] = frame.values.astype(np.float32)
    return stamps


def normalise(values: np.ndarray, *, clip: float = 5.0) -> np.ndarray:
    """float32 ``(n, 516, 6)``: each decision normalised with its 512 context rows' statistics.

    Per decision exactly what ``KronosPredictor.predict_batch`` does to its context (K:620-631):
    ``x = values.astype(float32)``; mean and population std over the context rows, reduced on a
    Fortran-ordered copy (the layout of the predictor's ``df[...].values.astype(float32)``);
    ``(x - mean) / (std + 1e-5)``; clipped at ``+-clip``. Rows 512..515 use the context
    statistics.
    """
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[1:] != (NLL_ROWS, len(FEATURE_COLUMNS)):
        raise NllError(
            f"values must have shape (n, {NLL_ROWS}, {len(FEATURE_COLUMNS)}), got {array.shape}"
        )
    if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not clip > 0:
        raise NllError(f"clip must be a positive number, got {clip!r}")
    normalised = np.empty(array.shape, dtype=np.float32)
    for index in range(array.shape[0]):
        x = array[index].astype(np.float32)
        context = np.asfortranarray(x[:NLL_CONTEXT])
        mean, std = np.mean(context, axis=0), np.std(context, axis=0)
        z = (x - mean) / (std + NORMALISATION_EPSILON)
        normalised[index] = np.clip(z, -clip, clip)
    return normalised


def _require_eval_on_cpu(**modules: Any) -> None:
    import torch

    for name, module in modules.items():
        if not isinstance(module, torch.nn.Module):
            raise NllError(f"the {name} must be a torch module, got {type(module).__name__}")
        if any(submodule.training for submodule in module.modules()):
            raise NllError(f"the {name} must be in eval mode (every submodule)")
        tensors = [*module.parameters(), *module.buffers()]
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise NllError(f"the {name} must be on the CPU")


def teacher_forced_logits(
    tokenizer: Any,
    model: Any,
    inputs: NllInputs,
    *,
    clip: float = 5.0,
    batch_size: int = NLL_BATCH_SIZE,
) -> TeacherForced:
    """The logits NLL4-TF scores, step by step, in batches of ``batch_size`` decisions.

    Every tensor a model reads is contiguous, as in ``auto_regressive_inference``; with one
    decision per batch the logits equal the ones inference computes from the same tokens.
    """
    require_configured()
    import torch

    if not isinstance(inputs, NllInputs):
        raise NllError(f"expected NllInputs, got {type(inputs).__name__}")
    if not _is_int(batch_size) or batch_size < 1:
        raise NllError(f"batch_size must be a positive integer, got {batch_size!r}")
    _require_eval_on_cpu(tokenizer=tokenizer, model=model)
    normalised = normalise(inputs.values, clip=clip)
    stamps = time_stamps(inputs.decision_ms)
    parts: dict[str, list[Any]] = {"s1": [], "s2": [], "s1_targets": [], "s2_targets": []}
    with torch.no_grad():
        for start in range(0, inputs.decisions, batch_size):
            rows = torch.from_numpy(normalised[start : start + batch_size])
            stamp = torch.from_numpy(stamps[start : start + batch_size])
            context_s1, context_s2 = tokenizer.encode(rows[:, :NLL_CONTEXT].contiguous(), half=True)
            full_s1, full_s2 = tokenizer.encode(rows, half=True)
            target_s1 = full_s1[:, NLL_CONTEXT:].contiguous()
            target_s2 = full_s2[:, NLL_CONTEXT:].contiguous()
            s1_tokens = torch.cat([context_s1, target_s1], dim=1)
            s2_tokens = torch.cat([context_s2, target_s2], dim=1)
            step_s1: list[Any] = []
            step_s2: list[Any] = []
            for step in range(NLL_STEPS):
                window = slice(step, NLL_CONTEXT + step)
                s1_logits, context = model.decode_s1(
                    s1_tokens[:, window].contiguous(),
                    s2_tokens[:, window].contiguous(),
                    stamp[:, window, :].contiguous(),
                )
                step_s1.append(s1_logits[:, -1, :])
                true_s1 = target_s1[:, step : step + 1].contiguous()
                s2_logits = model.decode_s2(context, true_s1)
                step_s2.append(s2_logits[:, -1, :])
            parts["s1"].append(torch.stack(step_s1, dim=1))
            parts["s2"].append(torch.stack(step_s2, dim=1))
            parts["s1_targets"].append(target_s1)
            parts["s2_targets"].append(target_s2)
    return TeacherForced(
        s1_logits=torch.cat(parts["s1"], dim=0),
        s2_logits=torch.cat(parts["s2"], dim=0),
        s1_targets=torch.cat(parts["s1_targets"], dim=0),
        s2_targets=torch.cat(parts["s2_targets"], dim=0),
    )


def _token_nll(logits: Any, targets: Any) -> np.ndarray:
    import torch

    log_probabilities = torch.log_softmax(logits.to(torch.float64), dim=-1)
    picked = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (-picked).numpy()


def nll4_tf(
    tokenizer: Any,
    model: Any,
    inputs: NllInputs,
    *,
    clip: float = 5.0,
    batch_size: int = NLL_BATCH_SIZE,
) -> NllResult:
    """NLL4-TF of ``model`` (with ``tokenizer``) on ``inputs``: see the module docstring."""
    require_configured()
    import torch

    started = time.perf_counter()
    forced = teacher_forced_logits(tokenizer, model, inputs, clip=clip, batch_size=batch_size)
    with torch.no_grad():
        s1 = _token_nll(forced.s1_logits, forced.s1_targets)
        s2 = _token_nll(forced.s2_logits, forced.s2_targets)
    per_decision = (s1 + s2).mean(axis=1)
    return NllResult(
        value=float(per_decision.mean()),
        per_decision=per_decision,
        s1=s1,
        s2=s2,
        decisions=inputs.decisions,
        batch_size=batch_size,
        threads=torch.get_num_threads(),
        seconds=time.perf_counter() - started,
    )


__all__ = [
    "NLL_BATCH_SIZE",
    "NLL_CONTEXT",
    "NLL_INPUTS_ARRAYS",
    "NLL_INPUTS_SCHEMA_VERSION",
    "NLL_META_KEYS",
    "NLL_ROWS",
    "NLL_STEPS",
    "NllError",
    "NllInputs",
    "NllResult",
    "TeacherForced",
    "load_nll_inputs",
    "nll4_tf",
    "normalise",
    "teacher_forced_logits",
    "time_stamps",
]
