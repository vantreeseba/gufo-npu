#!/bin/sh
# Prefer the host's XRT when it is mounted at /opt/xilinx/xrt, so the
# user-space plugin matches the host's amdxdna driver. Otherwise use the XRT
# bundled in the image.
#   docker run -v /opt/xilinx/xrt:/opt/xilinx/xrt:ro ...
if [ -e /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 ]; then
  export XILINX_XRT=/opt/xilinx/xrt
  export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec gufo-npu "$@"
