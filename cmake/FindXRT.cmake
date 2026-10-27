# Finds the XRT user-space runtime (headers + libxrt_coreutil).
#
# Covers the Debian/Ubuntu packages (libxrt-dev, /usr) and source installs
# under /opt/xilinx/xrt (or $XILINX_XRT). Defines XRT::xrt_coreutil.

find_path(XRT_INCLUDE_DIR xrt/xrt_device.h
  HINTS ENV XILINX_XRT
  PATHS /opt/xilinx/xrt
  PATH_SUFFIXES include)
find_library(XRT_COREUTIL_LIBRARY xrt_coreutil
  HINTS ENV XILINX_XRT
  PATHS /opt/xilinx/xrt
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XRT
  REQUIRED_VARS XRT_COREUTIL_LIBRARY XRT_INCLUDE_DIR)

if(XRT_FOUND AND NOT TARGET XRT::xrt_coreutil)
  add_library(XRT::xrt_coreutil UNKNOWN IMPORTED)
  set_target_properties(XRT::xrt_coreutil PROPERTIES
    IMPORTED_LOCATION "${XRT_COREUTIL_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${XRT_INCLUDE_DIR}")
endif()
mark_as_advanced(XRT_INCLUDE_DIR XRT_COREUTIL_LIBRARY)
