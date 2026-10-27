# AGENTS.md

gufo-npu is an open inference engine for the AMD XDNA2 NPU in Strix Halo
(Ryzen AI MAX+ 395), for one model: Qwen3.6-35B-A3B. The roadmap is in
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
  the CTest label `npu`. Model tests exit 77 (skip) without
  `GUFO_NPU_QWEN36_MODEL`; a skip is not a pass.
- Each NPU kernel has an op test against the CPU reference
  (relative error < 1e-2). Model changes need the teacher-forced greedy-argmax
  match. Record checks in `docs/models/qwen3.6-35b-a3b/QUALITY.md`.
- Timings are compared per request and per phase, against FLM and against
  main, with the same prompts and toolchain. Record them in `BENCHMARKS.md`.
- Before committing C++: `python3 tools/ci/check-format.py` (`--fix` to apply).

## Kernels

- Designs live in `kernels/designs/<name>.py` and expose
  `compile_to(out_dir, dev)`. Add each to `GUFO_NPU_KERNEL_DESIGNS` in
  `kernels/CMakeLists.txt`.
- Decode is bound by NPU DMA bandwidth (~54 GB/s) and by per-dispatch cost.
  Chain dependent dispatches in one `xrt::runlist` and fuse ops so a token
  needs few commands (`gufo-npu bench dispatch` measures this).
- Successful optimizations become the default, without extra switches.

## Development

- Keep model code, tests, tools and numerical contracts together.
- One canonical long option per behavior; no aliases.
- Use `gh` for GitHub operations. Conventional Commits, single-line messages;
  PR titles follow `tools/ci/check-pr-title.py`. Squash merges keep the title
  on main. Leave versions and the changelog to release-please.
- Never commit weights, model paths or machine/host names.
