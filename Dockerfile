# syntax=docker/dockerfile:1

# gufo-npu build and runtime images.
#
# Stages:
#   xrt        Ubuntu 24.04 + the pinned XRT NPU runtime (lemonade-team PPA).
#   toolchain  xrt + compilers, XRT headers and the pinned NPU kernel compiler
#              (mlir-aie, llvm-aie/Peano, IRON). CI and local builds use this.
#   build      Compiles the host binaries and NPU kernels and runs CPU tests.
#              Compiling kernels needs no NPU.
#   runtime    xrt + the installed engine. This is what CI publishes.
#
#   docker build -t gufo-npu .
#   docker build --target toolchain -t gufo-npu-toolchain .
#
# Run on a Strix Halo host (the NPU is /dev/accel/accel0, char major 261):
#   docker run --rm --device /dev/accel/accel0 --ulimit memlock=-1:-1 \
#     [-v /opt/xilinx/xrt:/opt/xilinx/xrt:ro] gufo-npu diagnose
# Mount the host XRT when the host runs an out-of-tree amdxdna driver whose
# version differs from the bundled XRT.

ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS xrt
ARG DEBIAN_FRONTEND=noninteractive
# Keep in sync with docs/TOOLCHAIN.md. The host's amdxdna driver and NPU
# firmware must suit this XRT release; `gufo-npu diagnose` reports both.
ARG XRT_VERSION=1:2.25.0-4~noble1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates gnupg software-properties-common \
 && add-apt-repository -y ppa:lemonade-team/stable \
 && apt-get install -y --no-install-recommends \
      libxrt2="${XRT_VERSION}" libxrt-npu2="${XRT_VERSION}" \
      libxrt-utils="${XRT_VERSION}" libxrt-utils-npu="${XRT_VERSION}" \
      libicu74 \
 && apt-get purge -y software-properties-common && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

FROM xrt AS toolchain
ARG DEBIAN_FRONTEND=noninteractive
ARG XRT_VERSION=1:2.25.0-4~noble1
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build ccache git clang-format \
      python3 python3-dev python3-venv \
      uuid-dev libicu-dev libxrt-dev="${XRT_VERSION}" \
 && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
COPY .devops/requirements-kernels.txt /tmp/requirements-kernels.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements-kernels.txt

FROM toolchain AS build
ARG BUILD_JOBS=4
WORKDIR /src
COPY . .
RUN cmake --preset release \
 && cmake --build --preset release --parallel "${BUILD_JOBS}" \
 && ctest --preset cpu \
 && cmake --install build/release --prefix /opt/gufo-npu

FROM xrt AS runtime
COPY --from=build /opt/gufo-npu /opt/gufo-npu
COPY .devops/entrypoint.sh /opt/gufo-npu/bin/entrypoint.sh
ENV PATH="/opt/gufo-npu/bin:${PATH}"
ENTRYPOINT ["/opt/gufo-npu/bin/entrypoint.sh"]
CMD ["diagnose"]
