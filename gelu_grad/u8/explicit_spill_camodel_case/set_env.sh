#!/usr/bin/env bash
# Source this file in Bash after adapting the settings below to your machine.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Use: source set_env.sh\n' >&2
    exit 1
fi

# Machine and CAModel configuration. Edit these values when moving the archive.
export MEMBAR_CANN_ROOT="/home/lenovo/Ascend/ascend-toolkit/cann-9.0.0-beta.1"
export MEMBAR_HOST_ARCH="x86_64-linux"
export MEMBAR_NPU_TYPE="Ascend950PR_9599"
export MEMBAR_CORE_ARCH="dav-c310-vec"

# Optional colon-separated include/library directories. Empty uses SDK defaults.
export MEMBAR_CXX_INCLUDE_DIRS=""
export MEMBAR_EXTRA_LIBRARY_DIRS=""

# Optional compiler overrides. Empty selects the tools inside MEMBAR_CANN_ROOT.
export MEMBAR_CCEC=""
export MEMBAR_HOST_CXX=""
export MEMBAR_LINKER=""

if [[ ! -f "${MEMBAR_CANN_ROOT}/set_env.sh" ]]; then
    printf 'CANN environment script not found: %s/set_env.sh\nEdit MEMBAR_CANN_ROOT in this file.\n' "${MEMBAR_CANN_ROOT}" >&2
    return 1
fi

# Vendor environment scripts may read unset variables; restore the caller's mode.
_membar_restore_nounset=0
if [[ $- == *u* ]]; then
    _membar_restore_nounset=1
    set +u
fi
_membar_vendor_status=0
source "${MEMBAR_CANN_ROOT}/set_env.sh" || _membar_vendor_status=$?
if [[ ${_membar_restore_nounset} == 1 ]]; then
    set -u
fi
unset _membar_restore_nounset
if [[ ${_membar_vendor_status} != 0 ]]; then
    printf 'Failed to load the CANN environment.\n' >&2
    unset _membar_vendor_status
    return 1
fi
unset _membar_vendor_status

export ACL_PATH="${MEMBAR_CANN_ROOT}"
export ASCEND_TOOLKIT_HOME="${MEMBAR_CANN_ROOT}"
export NPU_TYPE="${MEMBAR_NPU_TYPE}"
export CORE_ARCH="${MEMBAR_CORE_ARCH}"

_membar_libraries=(
    "${MEMBAR_CANN_ROOT}/tools/simulator/${MEMBAR_NPU_TYPE}/lib"
    "${MEMBAR_CANN_ROOT}/simulator/${MEMBAR_NPU_TYPE}/lib"
    "${MEMBAR_CANN_ROOT}/aarch64-linux/simulator/${MEMBAR_NPU_TYPE}/lib"
    "${MEMBAR_CANN_ROOT}/lib64"
    "${MEMBAR_CANN_ROOT}/${MEMBAR_HOST_ARCH}/lib64"
    "${MEMBAR_CANN_ROOT}/${MEMBAR_HOST_ARCH}/devlib"
    "${MEMBAR_CANN_ROOT}/${MEMBAR_HOST_ARCH}/devlib/device"
    "${MEMBAR_CANN_ROOT}/${MEMBAR_HOST_ARCH}/lib64/device/lib64"
)
IFS=: read -r -a _membar_extra_libraries <<< "${MEMBAR_EXTRA_LIBRARY_DIRS}"
_membar_libraries+=("${_membar_extra_libraries[@]}")
for _membar_library in "${_membar_libraries[@]}"; do
    [[ -d "${_membar_library}" ]] || continue
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${_membar_library}:"*) ;;
        *) export LD_LIBRARY_PATH="${_membar_library}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
done
unset _membar_libraries _membar_extra_libraries _membar_library
printf 'CAModel environment: CANN=%s NPU=%s target=%s\n' "${MEMBAR_CANN_ROOT}" "${MEMBAR_NPU_TYPE}" "${MEMBAR_CORE_ARCH}"
