#!/bin/bash
# Build gnina for Roihu GPU nodes (NVIDIA GH200) using the CUDA backend.
#
# Roihu's GPU nodes are Grace Hopper superchips: an H100 (sm_90) paired with a
# 72-core ARM Grace CPU. The CPU-only nodes and (probably) the login nodes are
# x86 AMD Turin, so a binary built on a login node will NOT run on a GPU node.
# Check before you build:
#
#   uname -m                                   # on the login node
#   srun -A $PROJECT -p <gpu-partition> --gres=gpu:1 -t 0:05:00 uname -m
#
# If those differ, run this script from inside an interactive GPU allocation so
# everything is compiled aarch64:
#
#   srun -A $PROJECT -p <gpu-partition> --gres=gpu:1 -c 32 -t 4:00:00 --pty bash
#   ./scripts/build_roihu.sh
#
# Usage:
#   export PROJECT=project_2XXXXXX
#   ./scripts/build_roihu.sh
set -euo pipefail

: "${PROJECT:?set PROJECT to your CSC project, e.g. export PROJECT=project_2001234}"

SRC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX=${PREFIX:-/projappl/$PROJECT/gnina-cuda}
WORK=${WORK:-$PREFIX/build}
NPROC=${NPROC:-$(nproc)}
CUDA_ARCH=${CUDA_ARCH:-90}            # H100 / GH200
BOOST_VERSION=${BOOST_VERSION:-1.84.0}

STAMPS=$PREFIX/.stamps
mkdir -p "$PREFIX" "$WORK" "$STAMPS"

stage_done() { [[ -f "$STAMPS/$1" ]]; }
stage_mark() { touch "$STAMPS/$1"; }
say() { printf '\n=== %s ===\n' "$*"; }

say "host architecture: $(uname -m)"
if [[ "$(uname -m)" != "aarch64" ]]; then
  cat >&2 <<'EOF'
WARNING: building on a non-aarch64 host. Roihu's GPU nodes are Grace (ARM), so
this binary will not run there. See the header of this script -- build from
inside an interactive GPU allocation instead. Continuing anyway in 10s.
EOF
  sleep 10
fi

# ---------------------------------------------------------------- environment
say "loading modules"
module --force purge
module load gcc
module load cuda
module load cmake || true             # some stacks fold cmake into a base module

command -v nvcc >/dev/null || { echo "nvcc not found -- check 'module avail cuda'" >&2; exit 1; }
nvcc --version | tail -2

export CC=gcc CXX=g++
export CUDACXX=$(command -v nvcc)
export CMAKE_PREFIX_PATH=$PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# ---------------------------------------------------------------------- boost
if ! stage_done boost; then
  say "boost $BOOST_VERSION"
  cd "$WORK"
  BOOST_US=${BOOST_VERSION//./_}
  [[ -d boost_$BOOST_US ]] || {
    curl -fL -o boost.tar.gz \
      "https://archives.boost.io/release/$BOOST_VERSION/source/boost_$BOOST_US.tar.gz"
    tar xf boost.tar.gz && rm boost.tar.gz
  }
  cd "boost_$BOOST_US"
  ./bootstrap.sh --prefix="$PREFIX" \
    --with-libraries=program_options,system,iostreams,timer,thread,serialization,filesystem,date_time,regex,test
  ./b2 -j"$NPROC" cxxflags="-std=c++17 -fPIC" link=shared install
  stage_mark boost
fi

# -------------------------------------------------------------------- jsoncpp
if ! stage_done jsoncpp; then
  say "jsoncpp"
  cd "$WORK"
  [[ -d jsoncpp ]] || git clone --depth 1 -b 1.9.5 https://github.com/open-source-parsers/jsoncpp.git
  cmake -S jsoncpp -B jsoncpp/build -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DJSONCPP_WITH_TESTS=OFF -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  cmake --build jsoncpp/build -j"$NPROC" --target install
  stage_mark jsoncpp
fi

# ------------------------------------------------------------------ openbabel
if ! stage_done openbabel; then
  say "openbabel 3"
  cd "$WORK"
  [[ -d openbabel ]] || git clone --depth 1 https://github.com/openbabel/openbabel.git
  cmake -S openbabel -B openbabel/build -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release -DWITH_MAEPARSER=OFF -DWITH_COORDGEN=OFF \
    -DPYTHON_BINDINGS=OFF -DBUILD_GUI=OFF -DENABLE_TESTS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  cmake --build openbabel/build -j"$NPROC" --target install
  stage_mark openbabel
fi

# ---------------------------------------------------------------------- torch
# download.pytorch.org only ships x86_64 libtorch archives, so on Grace take
# libtorch out of the aarch64 CUDA pip wheel instead.
# load python before the stage guard: the venv symlinks the module's
# interpreter, so it has to be present on re-runs too
module load python-data 2>/dev/null || module load python 2>/dev/null || true
if ! stage_done torch; then
  say "torch (aarch64 + CUDA wheel)"
  python3 -m venv "$PREFIX/venv"
  # shellcheck disable=SC1091
  source "$PREFIX/venv/bin/activate"
  pip install --upgrade pip
  pip install torch --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
  deactivate
  stage_mark torch
fi
# shellcheck disable=SC1091
source "$PREFIX/venv/bin/activate"
TORCH_CMAKE=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
TORCH_LIB=$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
echo "torch cmake prefix: $TORCH_CMAKE"

# ------------------------------------------------------------------ libmolgrid
if ! stage_done libmolgrid; then
  say "libmolgrid"
  cd "$WORK"
  [[ -d libmolgrid ]] || git clone --depth 1 https://github.com/gnina/libmolgrid.git
  cmake -S libmolgrid -B libmolgrid/build \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_PREFIX_PATH="$PREFIX;$TORCH_CMAKE" \
    -DBUILD_PYTHON=OFF
  cmake --build libmolgrid/build -j"$NPROC" --target install
  stage_mark libmolgrid
fi

# ---------------------------------------------------------------------- gnina
say "gnina (CUDA backend, sm_$CUDA_ARCH)"
cmake -S "$SRC_ROOT" -B "$WORK/gnina" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DGNINA_GPU_BACKEND=CUDA \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_PREFIX_PATH="$PREFIX;$TORCH_CMAKE" \
  -DGNINA_FORCE_EXTERNAL_LIBS=ON \
  -DLIBMOLGRID_LIBRARY="$PREFIX/lib/libmolgrid.a" \
  -DLIBMOLGRID_INCLUDE_DIR="$PREFIX/include"
cmake --build "$WORK/gnina" -j"$NPROC" --target install

say "done"
cat <<EOF
gnina installed to $PREFIX/bin/gnina

To use it in a job:
  module load gcc cuda
  export LD_LIBRARY_PATH=$PREFIX/lib:$PREFIX/lib64:$TORCH_LIB:\$LD_LIBRARY_PATH
  export PATH=$PREFIX/bin:\$PATH
EOF
