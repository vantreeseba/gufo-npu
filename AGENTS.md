# AGENTS.md

gufo-npu is an open inference engine for the AMD XDNA2 NPU in Strix Halo
(Ryzen AI MAX+ 395). The MVP runs Qwen3.5-2B; Qwen3.6-35B-A3B (MoE) is the
second pass. The roadmap is in
[docs/PLAN.md](docs/PLAN.md). Conventions follow
[gufo](https://github.com/gufo-org/gufo).

## Platform and build

Linux x86-64 only. CMake owns flags and installation; the Dockerfile pins the
toolchain (Ubuntu 24.04, XRT from the lemonade-team PPA, mlir-aie + llvm-aie
(Peano) + IRON). Versions are listed in [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md).

```sh
docker build -t gufo-npu .                    # full build, CPU tests, runtime image
docker build --target toolchain -t gufo-npu-toolchain .

# Inside the toolchain image (or a host with the same pins):
cmake --preset release && cmake --build --preset release
ctest --preset cpu                            # hosted-CI tests
ctest --preset npu                            # needs /dev/accel/accel0

# CPU-only, no XRT or kernel compiler:
cmake --preset cpu-test && cmake --build --preset cpu-test && ctest --preset cpu-only
```

Compiling NPU kernels needs no NPU; running them does. Run NPU work with
`--device /dev/accel/accel0 --ulimit memlock=-1:-1`, and mount the host's
`/opt/xilinx/xrt` when its amdxdna driver is newer than the bundled XRT.

## Tests

- Run the smallest test that covers the change. Tests that need the NPU carry
  the CTest label `npu`. A model's tests live in `tests/models/<model>/` and
  carry its label (`ctest -L qwen35_2b`), plus `model` when they need
  weights and `kernels` for NPU op tests. Model tests exit 77 (skip) without
  their model's variables (e.g. `GUFO_NPU_QWEN36_MODEL`); a skip is not a
  pass.
- Each NPU kernel has an op test against the CPU reference
  (relative error < 1e-2). Model changes need the teacher-forced greedy-argmax
  match. Record checks in the model's `docs/models/<model>/QUALITY.md`, and
  retained or rejected optimizations in its `EXPERIMENTS.md`.
- Timings are compared per request and per phase, against FLM and against
  main, with the same prompts and toolchain. Record them in `BENCHMARKS.md`.
- Model tests (CTest label `model`) need, for Qwen3.5-2B:
  `GUFO_NPU_QWEN35_2B_GGUF` (the BF16 GGUF for the HF-oracle tests) and
  `GUFO_NPU_QWEN35_2B_ORACLE` (output of `tools/reference/qwen35_oracle.py`),
  plus `GUFO_NPU_QWEN35_2B_Q4_GGUF` (UD-Q4_K_XL) for the decoder and NPU
  decode tests.
  Run them from a Release build; the CPU reference is slow in Debug.
- Before committing C++: `python3 tools/ci/check-format.py` (`--fix` to apply).

## Kernels

- Designs live in `kernels/designs/<name>.py` and expose
  `compile_to(out_dir, dev)`. Register each with `gufo_npu_add_kernel(NAME
  DESIGN [SOURCES] MODEL)` (`cmake/GufoNpuKernels.cmake`) in
  `kernels/CMakeLists.txt`. A design with more outputs than
  `<name>.xclbin` + `<name>.insts.elf` keeps its shapes, sequence names and
  `OUTPUTS` in a pure-data `<name>_variants.py` (no IRON imports), which the
  build reads at configure time (`build.py --list-outputs --design PATH`).
  `-DGUFO_NPU_MODELS=qwen35_2b` builds only that model's kernels.
- Decode is bound by NPU read bandwidth (~56 GB/s; `bench bandwidth`) and
  by per-dispatch cost.
  Chain dependent dispatches in one `xrt::runlist` and fuse ops so a token
  needs few commands (`gufo-npu bench dispatch` measures this).
- Successful optimizations become the default, without extra switches.
- IRON reuses any kernel object already in a compile work dir, even when the
  source changed; designs clear their work dir before compiling.
- Runs from an instruction ELF (`xrt::module`) and runs given their
  instruction stream as a BO (`.insts.bin`, the path for patching DDR
  offsets per call, as the MoE gathers do) cannot share a runlist
  (`ERT_CMD_STATE_NEW`). A `GemvRuntime` uses one of the two
  (`npu::Flow`): MoE models the instruction buffers, dense models the ELFs. A runtime-sequence wait costs ~4.5 µs, so group DMA
  tasks and await only the last of each group per channel
  (docs/research/NPU-MOE.md).
- In a runtime sequence, waiting on a DMA task (`wait=True`, a task group's
  `finish()`) takes the next completion token of that task's shim channel,
  whichever task sent it. Await the tasks of one channel in the order they
  were queued, or a wait returns early and a reused BD can overwrite a
  transfer still in flight (see `kernels/designs/q4nx_gemv.py`;
  `q4nx_gemv_chain_test` runs the chain 60,000 times to catch this).
- A core starts in floor (truncating) rounding mode; kernels that narrow to
  bf16 set round-to-nearest-even first (see `kernels/aie/q4nx_gemv.cc`).
- One xclbin per kernel family, one instruction sequence per shape (shape
  in runtime parameters), so every call runs in one hardware context and
  can join a runlist. `kernels/designs/q4nx_gemv.py` checks that all its
  shapes build the same array configuration.

## Code layout

Follows gufo: numerical code is per model, so a change made for one model
cannot move another's output or speed; shared code carries no model's
numerics. On the NPU the unit of isolation is the xclbin (one core program,
one hardware context, one flow), as a kernel source file is in gufo.

- Shared:
  - `src/core`: GGUF reading, quant formats (q4nx, bfp16) and their
    packers, tokenizer, vector helpers, pools. A format change (such as
    q4nx rounding) is qualified on every model that uses it.
  - `src/npu`: XRT plumbing (device, buffers, runlists, instruction
    patching), no model constants.
  - `src/models/qwen35`: the Qwen3.5/3.6 family's CPU code (config,
    reference, CPU decoder, MoE routing, MTP, chat), as gufo's `qwen/`
    package serves several sizes. A change here is qualified on each family
    model.
- Per model (`qwen35_2b`, `qwen36_35b_a3b`, `qwen35_4b`): NPU decoder and
  prefill, kernel designs, core programs and xclbins with their shape
  lists, op and model tests (`tests/models/<model>/`), docs
  (`docs/models/<slug>/`: README, BENCHMARKS, QUALITY, EXPERIMENTS) and
  tools. A model may borrow another's kernel only through a split-out
  library, and its docs say so.
- Moving there: today the 2B and the 35B still share
  `src/models/qwen35/npu_decoder.*` and the `q4nx_gemv` design and xclbin,
  and the 2B's prefill sits in `src/models/qwen35/`. The plan is to fork the
  decode kernel per model (`src/models/<model>/kernels/`, xclbin
  `<model>_decode`), trim each copy to its own shapes, split the NPU decoder
  per model and select the model through a registry by GGUF arch and shape.
  New models start in their own package and never edit another model's
  kernels.

## Development

- Keep model code, tests, tools and numerical contracts together.
- One canonical long option per behavior; no aliases.
- Use `gh` for GitHub operations. Conventional Commits, single-line messages;
  PR titles follow `tools/ci/check-pr-title.py`. Squash merges keep the title
  on main. Leave versions and the changelog to release-please.
- Never commit weights, model paths or machine/host names.
