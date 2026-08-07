# Development

[English](DEVELOP.md) · [中文](zh/DEVELOP.md)

## Setup

```bash
FAST_ULYSSES_CUDA_ARCH=<your arch, e.g. 90> pip install -e ".[dev]" --no-build-isolation
pre-commit install
```

The build reuses the persistent `build/` directory, so edit-rebuild cycles only recompile changed
translation units (plus `ccache` when installed). `NVSHMEM_HOME` is optional — see
[INSTALL.md](INSTALL.md).

## Linting

pre-commit is the single entry point:

```bash
pre-commit run --all-files
```

Python is **ruff** (check + format, line length 100, py310+). C++/CUDA under `fast_ulysses/csrc/`
is **clang-format**, pinned to **v15.0.7** in `.pre-commit-config.yaml` — the version the code is
formatted with. Keep the pin and any local binary in sync.

## Tests

```bash
pytest                      # everything runnable here
pytest -m "not multigpu"    # host-only, no GPU needed
pytest -m multigpu          # the torchrun-wrapped suites
```

`tests/test_plan.py` replays the addressing (`csrc/a2a_plan.cpp`) over numpy buffers against an
`all_to_all_single` + permute reference. It needs no GPU and no process group, only the built
extension — which is why it is the one correctness check CI can run.

`tests/test_multigpu.py` launches each worker under `tests/distributed/` as a
`torch.distributed.run` subprocess and skips below 2 GPUs. `FAST_ULYSSES_TEST_NPROC` overrides the
process count (`=3` exercises odd world sizes).

Workers stay directly runnable for debugging:

```bash
torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py
```

| worker | what it asserts |
|---|---|
| `a2a_correctness` | bit-exact against a torch permute + a2a + permute reference |
| `a2a_fallback` | `TorchUlyssesGroup` bit-exact with `UlyssesGroup`, and `make_group`'s choice follows the socket layout |
| `a2a_async` | the async result matches the sync one, and the overlap window is real |
| `a2a_uneven` | `seq_splits` / `head_splits` against `dist.all_to_all` over ragged tensors |
| `a2a_copy_out` | the copying form owns its result; the borrowed form does not |
| `a2a_subgroup`, `a2a_subgroup_divergent` | two stride-2 subgroups live at once, same and divergent shapes |
| `a2a_torch_nvshmem_coexist` | this extension and torch's own NVSHMEM in one process |

The six below are **adversarial**: each builds one specific unsafe timing and asserts only that
the result is not torn.

| worker | timing it builds |
|---|---|
| `a2a_window_race` | a peer's next call arriving while we still read our window |
| `a2a_cudagraph` | a captured call replayed, against the device-side epoch |
| `a2a_ce_flag_ordering` | is the payload visible when the flag announcing it arrives |
| `a2a_ce_fault_injection` | arms `_set_ce_fault` to break that ordering on demand |
| `a2a_overlapping_barriers` | an async call on one tag against a sync call on another |
| `a2a_alias_guard` | an input or `out` that overlaps the tag's own window |

An adversarial worker is worth exactly as much as the timing it builds, and a timing decays: a
barrier change or a faster machine can re-align the ranks and leave a worker that passes while
testing nothing. So each names its **negative control** in its module docstring — the single line
to delete to make it fail, and what that failure looks like. **Re-run the controls after any
barrier change**; a pass with the control applied means the worker is blind, not that the code is
safe. `a2a_ce_fault_injection` is the exception: it arms its own control every run.
`a2a_cudagraph` prints `captured=True/False` — a green run with `captured=False` checked nothing.

`a2a_overlapping_groups.py` is not registered and must not be: it builds two groups that overlap
instead of partitioning, which hangs by design. It documents the constraint; it is not a test.

## Releasing

CI has no GPU runner, so nothing under `tests/distributed/` ever runs there. What CI does prove is
that each configuration compiles for four architectures, links against exactly the expected
libraries with a relocatable RUNPATH, loads under the target torch, and passes `test_plan.py`.

```bash
scripts/build_wheels.sh          # one (torch, CUDA) row, inside a manylinux builder
scripts/check_wheel.py <whl>     # the ELF/metadata gate; also runs inside build_wheels.sh
scripts/preflight_gpu.sh <whl>   # MANDATORY before a tag: the built wheel on a real multi-GPU box
```

`preflight_gpu.sh` prints a block for the release notes. Run it for at least the newest torch row
and one CUDA-12 row; the oldest rows ship on compile-and-load evidence only, and the release notes
should say so.

Benchmarks must run under `scripts/exclusive.sh`, which refuses to start until the requested GPUs
are free and prints `EXCLUSIVE` or `CONTENDED`. A `CONTENDED` number is not a number.

## Layout

```
fast_ulysses/          Python package (comm.py: UlyssesGroup; cli.py: doctor)
fast_ulysses/csrc/     C++/CUDA sources (bindings.cpp registers the torch library)
tests/                 pytest suites; tests/distributed/ holds the torchrun workers
benchmark/             stage breakdown, padding cost, GEMM overlap, an nsys/ncu driver
scripts/               GPU-exclusivity wrapper, wheel build and gate, release preflight
docs/                  this documentation; docs/zh/ is the Chinese translation
```
