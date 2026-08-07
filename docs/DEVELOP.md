# Development

## Setup

```bash
NVSHMEM_HOME=<nvshmem install root> \
FAST_ULYSSES_CUDA_ARCH=<your arch, e.g. 90> \
pip install -e . --no-build-isolation

pip install -e ".[dev]" --no-build-isolation   # adds pre-commit, pytest, ruff
pre-commit install                             # run hooks on every commit
```

The build reuses the persistent `build/` directory, so edit-rebuild cycles only recompile changed
translation units (plus `ccache` when installed).

## Linting & formatting

pre-commit is the single lint entry point:

```bash
pre-commit run --all-files
```

- Python: **ruff** (check + format, line length 100, py310+).
- C++/CUDA (`fast_ulysses/csrc/`): **clang-format**, pinned to **v15.0.7** in
  `.pre-commit-config.yaml` — the version the current code is formatted with. Keep the pin and the
  local binary in sync if you format manually.

## Tests

```bash
pytest                      # everything runnable on this machine
pytest -m "not multigpu"    # only the single-GPU op tests (fast)
pytest -m multigpu          # only the torchrun-wrapped multi-GPU suites
```

- `tests/test_plan.py` — the addressing (`csrc/a2a_plan.cpp`) replayed over numpy buffers against
  an `all_to_all_single` + permute reference. Needs no GPU and no process group, only the built
  extension: `pytest tests/test_plan.py`.
- `tests/test_multigpu.py` — launches each worker under `tests/distributed/` as a
  `torch.distributed.run` subprocess; skips below 2 GPUs. `FAST_ULYSSES_TEST_NPROC` overrides the
  process count (e.g. `=3` to exercise odd world sizes).
- Workers stay directly runnable for debugging (full output, single suite):

```bash
torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py   # bit-exact vs torch reference
torchrun --nproc_per_node=8 tests/distributed/a2a_async.py         # bit-exact vs the sync call
torchrun --nproc_per_node=8 tests/distributed/a2a_subgroup.py      # tp=2 x sp, two live groups
```

The layout checks above are bit-exact comparisons against `torch.distributed` references (pure
data movement). The four workers below are not: they are ADVERSARIAL, each building one specific
unsafe timing and asserting only that the result is not torn.

```bash
torchrun --nproc_per_node=8 tests/distributed/a2a_window_race.py         # WINDOW_RACE
torchrun --nproc_per_node=8 tests/distributed/a2a_cudagraph.py           # CUDAGRAPH_REPLAY
torchrun --nproc_per_node=8 tests/distributed/a2a_ce_flag_ordering.py    # CE_FLAG_ORDER
torchrun --nproc_per_node=8 tests/distributed/a2a_overlapping_barriers.py  # OVERLAPPING_BARRIERS
```

Each is worth exactly as much as the timing it builds, and a timing decays: a change to the
barriers, or a faster machine, can re-align the ranks and leave a worker that passes while
testing nothing. So every one of them names its NEGATIVE CONTROL in its module docstring — the
single line to delete or change to make it fail, and what that failure looks like. **Re-run the
controls after any barrier change**; a pass with the control applied means the worker is blind,
not that the code is safe. `a2a_cudagraph.py` additionally prints `captured=True/False` on its
last line: a green run with `captured=False` checked nothing.

## Layout

```
fast_ulysses/          Python package (comm.py: UlyssesGroup + async handles)
fast_ulysses/csrc/     C++/CUDA sources (bindings.cpp registers the torch library)
tests/                 pytest suites; tests/distributed/ holds the torchrun workers
benchmark/             throughput / overlap benchmarks and a minimal nsys/ncu driver
docs/                  this documentation
```
