# Third-party code

## Kronos (`kronos/`)

- Upstream: https://github.com/shiyu-coder/Kronos (MIT License, Copyright (c) 2025 ShiYu).
- Pinned commit: `67b630e67f6a18c9e9be918d9b4337c960db1e9a` (2026-04-13T12:38:49Z, the merge of
  PR #243; the HEAD of `master` on the day it was vendored, 2026-10-30).
- Vendored subset: `LICENSE`, `README.md`, `requirements.txt`, `.gitignore` and everything under
  `model/`, `finetune/`, `finetune_csv/` (except the sample data and example images) and `tests/`.
  `examples/`, `figures/` and `webui/` are not vendored. `manifests/kronos_source.json` lists every
  blob of the upstream tree with its size, git blob id and sha256 and a `vendored` flag;
  `manifests/kronos_source.sha256` lists the vendored files in GNU `sha256sum` format.
- Never edited. Re-verify with `uv run kronos-agent verify --kronos` or, without this package,
  `shasum -a 256 -c manifests/kronos_source.sha256` from the repository root.
- The project imports the upstream package under its own top-level name `model`
  (`kronos_agent.kronos.vendored.import_kronos`).

Citation:

```
@misc{shi2025kronos,
  title={Kronos: A Foundation Model for the Language of Financial Markets},
  author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
  year={2025},
  eprint={2508.02739},
  archivePrefix={arXiv},
  primaryClass={q-fin.ST},
  url={https://arxiv.org/abs/2508.02739}
}
```

Pre-trained weights (`NeoQuasar/Kronos-Tokenizer-base`, `NeoQuasar/Kronos-small`, MIT) are not part
of this repository: `uv run kronos-agent fetch-weights --model small` downloads them into `weights/`
and verifies them against `manifests/weights.json`.
