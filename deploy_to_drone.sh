#!/bin/bash
# Build the Crazyflie firmware (with the trained policy compiled in) and flash it.
#
# Run this on the machine with the Crazyradio PA dongle plugged in — NOT on the
# cluster. The trained network is already installed as
# firmware/ai_drone_firmware/network_evaluate.h by `uv run main.py --train`.
#
# Usage:
#   ./deploy_to_drone.sh              # build + flash
#   ./deploy_to_drone.sh --build-only # just build cf2.bin
#   CF_URI=radio://0/80/2M ./deploy_to_drone.sh   # override radio URI
#
# Prerequisites (see firmware/ai_drone_firmware/README.MD):
#   - arm-none-eabi-gcc toolchain
#   - cfloader from the Bitcraze client tools:  pip install cfclient
#   - git-lfs (the firmware submodule needs it)
#   - macOS only: brew install libusb

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
FW_DIR="$REPO_ROOT/firmware/ai_drone_firmware"
CF_URI="${CF_URI:-radio://0/80/2M}"
BUILD_ONLY=false
[ "${1:-}" = "--build-only" ] && BUILD_ONLY=true

# --- preflight checks --------------------------------------------------------
fail=false
for tool in arm-none-eabi-gcc git make; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: '$tool' not found on PATH" >&2
        fail=true
    fi
done
if ! $BUILD_ONLY && ! command -v cfloader >/dev/null 2>&1; then
    echo "ERROR: 'cfloader' not found on PATH (pip install cfclient)" >&2
    fail=true
fi
$fail && exit 1

if [ ! -f "$FW_DIR/network_evaluate.h" ]; then
    echo "ERROR: $FW_DIR/network_evaluate.h missing — run 'uv run main.py --train' first" >&2
    exit 1
fi

# --- fetch firmware's nested submodules --------------------------------------
cd "$FW_DIR"
git submodule update --init --recursive -- external/crazyflie_firmware
git submodule update --init -- external/rl_tools

# --- build -------------------------------------------------------------------
( cd external/crazyflie_firmware && make cf2_defconfig )
make

if [ ! -f build/cf2.bin ]; then
    echo "ERROR: build finished but build/cf2.bin was not produced" >&2
    exit 1
fi
echo "Built $(du -h build/cf2.bin | cut -f1) firmware at $FW_DIR/build/cf2.bin"

if $BUILD_ONLY; then
    echo "--build-only: skipping flash."
    exit 0
fi

# --- flash -------------------------------------------------------------------
echo
echo "Put the Crazyflie into bootloader mode: hold the power button ~3s"
echo "until the blue LEDs blink, then press Enter to flash via $CF_URI"
read -r
cfloader flash build/cf2.bin stm32-fw -w "$CF_URI"

echo
echo "Done. Reboot the drone and connect with cfclient to test."
echo "First flight: use a net/cage, keep cfclient connected as a kill switch."
