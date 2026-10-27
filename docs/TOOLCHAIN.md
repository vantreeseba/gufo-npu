# Toolchain

All pins live in the [Dockerfile](../Dockerfile) and
[.devops/requirements-kernels.txt](../.devops/requirements-kernels.txt).
Bump them together and re-run `gufo-npu smoke` and `bench dispatch` on
hardware.

| Component | Pin | Notes |
|---|---|---|
| Base image | Ubuntu 24.04 | mlir-aie wheels need glibc ≥ 2.35 |
| XRT | `1:2.25.0-4~noble1` (lemonade-team PPA) | `libxrt2`, `libxrt-npu2`, `libxrt-dev` |
| mlir-aie | `v1.4.4` | wheels from GitHub releases |
| llvm-aie (Peano) | `22.0.0.2026092101+0006955e` | same pair as amd/IRON CI |
| IRON | `b3ba3cf3a138c21e021d5ee2d6794573cd7a3287` | operator library, Q4NX codec |
| Kernel target | `npu2` | XDNA2: Strix Point, Strix Halo, Krackan |

## Reference oracles

Not part of the image; each lives in its own virtualenv outside the repo.

| Oracle | Pin | Used by |
|---|---|---|
| HF transformers (Qwen3.5-2B) | [tools/reference/requirements.txt](../tools/reference/requirements.txt): transformers 5.21.0, torch 2.13.0+cpu | `qwen35_oracle.py` |
| llama.cpp (Qwen3.6-35B-A3B) | [tools/reference/requirements-llama.txt](../tools/reference/requirements-llama.txt): llama-cpp-python 0.3.36, which vendors llama.cpp `0c1e57098bba43ac29e6e3b677cdceebdd22334f` (2026-10-01) | `qwen36_llama_oracle.py` |

The llama.cpp oracle is built from source, CPU only, with fixed instruction
sets so its numbers do not depend on the build host, inside the toolchain
image (gcc 13, CMake 3.28, Python 3.12) with the venv on a mounted volume:

```sh
python3 -m venv /models/llama-venv
CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_BMI2=ON -DGGML_AVX512=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF" \
  /models/llama-venv/bin/pip install --no-binary llama-cpp-python \
  -r tools/reference/requirements-llama.txt
/models/llama-venv/bin/python -I tools/reference/qwen36_llama_oracle.py \
  --model <gguf> --out /models/oracle-35b
```

Its `llama_print_system_info` then reads `AVX2 FMA F16C BMI2 LLAMAFILE
OPENMP`. The oracle runs at most a few seconds per prompt and maps the GGUF
without copying it (no repacking).

## Host requirements (Strix Halo)

- Linux 7.0+ with in-tree `amdxdna`, or `amdxdna-dkms` / an xdna-driver build.
  NPU firmware ≥ 1.1.
- `/dev/accel/accel0` read/write for your user (usually the `render` group).
- IOMMU on: no `amd_iommu=off` (`amd_iommu=pt` is fine).
- Unlimited memlock. Weights live in pinned buffers, and even `xrt-smi` fails
  with the default 8 MiB. For containers, pass `--ulimit memlock=-1:-1`; on the
  host, raise it in `/etc/security/limits.conf`.
- Optional: keep the NPU awake to avoid a wake-up delay on the first dispatch
  after idle: `echo on | sudo tee /sys/class/accel/accel0/device/power/control`.

`gufo-npu diagnose` checks all of these.

## XRT version vs the host driver

The XRT user-space plugin should match the host's amdxdna driver. If the host
runs a newer out-of-tree driver than the bundled XRT, mount the host XRT:
`-v /opt/xilinx/xrt:/opt/xilinx/xrt:ro`. The entrypoint prefers it when
present. Both setups pass the smoke test on a host with xdna-driver 2.26.
