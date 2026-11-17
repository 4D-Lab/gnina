#!/bin/bash
# Build gnina for LUMI-G (AMD MI250X, gfx90a) using the HIP backend.
#
# Downloading and compiling are separate phases, because LUMI login nodes have
# outbound network access and compute nodes do not:
#
#   export PROJECT=project_465XXXXXX
#   FETCH_ONLY=1 ./scripts/build_lumi.sh          # on a login node
#   salloc -A $PROJECT -p small -c 32 -t 3:00:00  # LUMI-C: no GPU needed
#   srun ./scripts/build_lumi.sh                  # compiles, no network needed
#
# Compiling for gfx90a is cross-compilation -- hipcc never touches a device --
# so build on a CPU partition rather than spending GPU-hours on an allocation
# that leaves the MI250Xs idle.
#
# Running it in one go on a login node also works -- everything it needs is
# fetched before the first compiler is invoked. Keep NPROC modest there
# (NPROC=8) since login nodes are shared.
#
# Everything lands in $PREFIX (default /project/$PROJECT/gnina-rocm). Stages are
# independent and skip themselves if already done, so you can re-run this after
# fixing a failure without redoing earlier stages. Force a stage to rerun by
# deleting its marker in $PREFIX/.stamps.
set -euo pipefail

: "${PROJECT:?set PROJECT to your LUMI project, e.g. export PROJECT=project_465001234}"

SRC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX=${PREFIX:-/project/$PROJECT/gnina-rocm}
WORK=${WORK:-$PREFIX/build}
NPROC=${NPROC:-$(nproc)}
GPU_ARCH=${GPU_ARCH:-gfx90a}          # MI250X. Do not widen this on LUMI.
BOOST_VERSION=${BOOST_VERSION:-1.84.0}
FETCH_ONLY=${FETCH_ONLY:-0}

# libtorch must be a ROCm build, and its ROCm version should match the loaded
# rocm module: the archive bundles its own ROCm runtime libraries, so a
# mismatch puts two different HIP runtimes in one process. This default matches
# LUMI's rocm/6.3.4. Check what is offered for your module version at
# https://download.pytorch.org/libtorch/ and override LIBTORCH_URL if it moves.
LIBTORCH_URL=${LIBTORCH_URL:-https://download.pytorch.org/libtorch/rocm6.3/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Brocm6.3.zip}

# LUMI's buildtools module ships cmake 4.x, which hard-errors on any package
# config declaring compatibility with cmake < 3.5. jsoncpp 1.9.5's installed
# jsoncppConfig.cmake does exactly that. This is cmake's own documented escape
# hatch; drop it once the dependencies catch up.
CMAKE_COMPAT=(-DCMAKE_POLICY_VERSION_MINIMUM=3.5)

BOOST_US=${BOOST_VERSION//./_}
STAMPS=$PREFIX/.stamps
mkdir -p "$PREFIX" "$WORK" "$STAMPS"

stage_done() { [[ -f "$STAMPS/$1" ]]; }
stage_mark() { touch "$STAMPS/$1"; }
say() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------- environment
say "loading modules"
module --force purge
module load CrayEnv
module load rocm
module load buildtools          # cmake, ninja
module load cray-python         # gnina packs its torch models with python

: "${ROCM_PATH:?the rocm module did not set ROCM_PATH -- check 'module avail rocm'}"
echo "ROCM_PATH=$ROCM_PATH"
"$ROCM_PATH/bin/hipcc" --version | head -2

# Plain ROCm clang++ for C++ -- NOT hipcc. hipcc compiles every .cpp with
# "-x hip", which device-compiles host-only code sitting in __host__ __device__
# headers and ignores CMAKE_HIP_ARCHITECTURES (falling back to gfx906). The .cu
# sources are handled by cmake's HIP language instead, which does honour it.
export CC=$ROCM_PATH/llvm/bin/clang
export CXX=$ROCM_PATH/llvm/bin/clang++
export CMAKE_PREFIX_PATH=$PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
# ROCm's llvm/bin must be on PATH, not just in CXX: boost's bootstrap.sh
# invokes "clang++" by name and ignores $CXX when given --with-toolset=clang.
export PATH=$PREFIX/bin:$ROCM_PATH/llvm/bin:$PATH
command -v clang++ >/dev/null || {
  echo "clang++ not on PATH after loading rocm -- check $ROCM_PATH/llvm/bin" >&2
  exit 1
}
export LD_LIBRARY_PATH=$PREFIX/lib:$PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# torch's ROCm cmake insists on finding this in the environment; without it
# it aborts with "No GPU arch specified for ROCm build" on any node with no GPU
export PYTORCH_ROCM_ARCH="$GPU_ARCH"
TORCH_ROOT=$PREFIX/libtorch

# ============================================================ phase 1: fetch
# Everything in this phase needs the network. Run it on a login node.
if ! stage_done fetch; then
  say "fetching sources (needs network -- login node)"
  cd "$WORK"

  [[ -d "boost_$BOOST_US" ]] || {
    echo "-- boost $BOOST_VERSION"
    curl -fL -o boost.tar.gz \
      "https://archives.boost.io/release/$BOOST_VERSION/source/boost_$BOOST_US.tar.gz"
    tar xf boost.tar.gz && rm boost.tar.gz
  }

  [[ -d jsoncpp ]] || { echo "-- jsoncpp"
    git clone --depth 1 -b 1.9.5 https://github.com/open-source-parsers/jsoncpp.git; }

  [[ -d openbabel ]] || { echo "-- openbabel"
    git clone --depth 1 https://github.com/openbabel/openbabel.git; }

  [[ -d libmolgrid ]] || { echo "-- libmolgrid"
    git clone --depth 1 https://github.com/gnina/libmolgrid.git; }

  [[ -d "$TORCH_ROOT" ]] || { echo "-- libtorch (ROCm)"
    curl -fL -o libtorch.zip "$LIBTORCH_URL"
    unzip -q libtorch.zip -d "$PREFIX" && rm libtorch.zip; }

  # Upstream libmolgrid is CUDA-only: both its sources and its CMakeLists need
  # porting (hipify-perl only does the former). No network needed, but this
  # belongs with source preparation. Read the audit it prints at the end.
  echo "-- porting libmolgrid to HIP"
  "$SRC_ROOT/scripts/hipify_libmolgrid.sh" "$WORK/libmolgrid"

  stage_mark fetch
fi

if [[ "$FETCH_ONLY" == "1" ]]; then
  say "fetch complete"
  echo "Sources are in $WORK. Now compile in an allocation (no GPU needed):"
  echo "  salloc -A $PROJECT -p small -c 32 -t 3:00:00"
  echo "  srun $SRC_ROOT/scripts/build_lumi.sh"
  exit 0
fi

# ------------------------------------------------- deduplicate ROCm runtimes
# libtorch bundles its own copies of the ROCm runtime libraries. They carry the
# same SONAMEs as the module's, but they are *different files*, and the loader
# deduplicates by inode rather than by SONAME -- so both get mapped, putting two
# HIP runtimes in one process. That is unsupported and fails as wrong numbers
# rather than as an error: libmolgrid allocates device memory through one
# runtime and hands the pointer to torch, which reads it through the other.
#
# Replacing each bundled duplicate with a symlink makes every search path lead
# to the same inode, so it is mapped once no matter whether the loader reaches
# it via DT_RPATH, DT_RUNPATH or LD_LIBRARY_PATH -- which is why this is more
# reliable than reordering LD_LIBRARY_PATH. Only the libraries that actually
# hold device state are touched, and only when the module's copy declares the
# same SONAME; the compute libraries (rocBLAS, MIOpen, hipBLASLt) are left
# alone because they carry version-matched data files alongside them.
#
# Safe because LIBTORCH_URL is pinned to the same ROCm minor version as the
# module. Originals are kept next to the symlink as *.bundled.
soname_of() { readelf -d "$1" 2>/dev/null | awk -F'[][]' '/SONAME/{print $2}'; }

dedupe_rocm_libs() {
  [[ -d "$TORCH_ROOT/lib" ]] || return 0
  local f base target n=0
  # Only the libraries that hold device-level state, where a second copy means
  # a second set of contexts/queues/code-object caches. amd_comgr is in the list
  # because HIP loads code objects through it; the compute libraries (rocBLAS,
  # MIOpen, hipBLASLt) are not, because they ship version-matched Tensile data
  # alongside them and a duplicate of one is wasteful rather than incorrect.
  for f in "$TORCH_ROOT"/lib/lib{amdhip64,hsa-runtime64,amd_comgr,hiprtc}.so*; do
    [[ -f $f && ! -L $f ]] || continue
    base=$(basename "$f")
    target="$ROCM_PATH/lib/$base"
    [[ -e $target ]] || continue
    [[ "$(soname_of "$f")" == "$(soname_of "$target")" ]] || {
      echo "   keeping bundled $base (SONAME differs from the module's)"; continue; }
    mv "$f" "$f.bundled"
    ln -s "$target" "$f"
    echo "   $base -> $target"
    n=$((n+1))
  done
  [[ $n -eq 0 ]] || say "deduplicated $n bundled ROCm runtime librar$([[ $n -eq 1 ]] && echo y || echo ies)"
}
dedupe_rocm_libs

# =========================================================== phase 2: compile
# ---------------------------------------------------------------------- boost
if ! stage_done boost; then
  say "boost $BOOST_VERSION"
  cd "$WORK/boost_$BOOST_US"
  # NB: boost.iostreams needs zlib headers for gnina's gzipped output (.sdf.gz);
  # b2 picks up the system zlib automatically. If it reports "zlib : no", the
  # binary will still build but fail to write .gz files at runtime.
  ./bootstrap.sh --prefix="$PREFIX" --with-toolset=clang \
    --with-libraries=program_options,system,iostreams,timer,thread,serialization,filesystem,date_time,regex,test
  ./b2 -j"$NPROC" toolset=clang cxxflags="-std=c++17 -fPIC" link=shared install
  stage_mark boost
fi

# -------------------------------------------------------------------- jsoncpp
if ! stage_done jsoncpp; then
  say "jsoncpp"
  cmake -S "$WORK/jsoncpp" -B "$WORK/jsoncpp/build" "${CMAKE_COMPAT[@]}" -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DJSONCPP_WITH_TESTS=OFF -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  cmake --build "$WORK/jsoncpp/build" -j"$NPROC" --target install
  stage_mark jsoncpp
fi

# ------------------------------------------------------------------ openbabel
if ! stage_done openbabel; then
  say "openbabel 3"
  cmake -S "$WORK/openbabel" -B "$WORK/openbabel/build" "${CMAKE_COMPAT[@]}" -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release -DWITH_MAEPARSER=OFF -DWITH_COORDGEN=OFF \
    -DPYTHON_BINDINGS=OFF -DBUILD_GUI=OFF -DENABLE_TESTS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  cmake --build "$WORK/openbabel/build" -j"$NPROC" --target install
  stage_mark openbabel
fi

# ------------------------------------------------------------------ libmolgrid
if ! stage_done libmolgrid; then
  say "libmolgrid (hipified)"

  # libmolgrid's find module does not know about openbabel's versioned include
  # directory (<prefix>/include/openbabel3), so it finds the library and then
  # leaves OPENBABEL3_INCLUDE_DIR as NOTFOUND. Locate it from the header that
  # is guaranteed to be installed and hand it over.
  ob_header=$(ls "$PREFIX"/include/openbabel*/openbabel/babelconfig.h 2>/dev/null | head -1 || true)
  [[ -n "$ob_header" ]] || { echo "openbabel headers not found under $PREFIX/include" >&2; exit 1; }
  OB_INCLUDE=$(dirname "$(dirname "$ob_header")")
  echo "   openbabel headers: $OB_INCLUDE"

  # AMDGPU_TARGETS/GPU_TARGETS are what hip-config reads. Without them it warns
  # that GPU autodetection failed (there is no GPU on a build node) and falls
  # back to gfx906/gfx942 -- neither of which runs on MI250X.
  cmake -S "$WORK/libmolgrid" -B "$WORK/libmolgrid/build" \
    "${CMAKE_COMPAT[@]}" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
    -DCMAKE_HIP_ARCHITECTURES="$GPU_ARCH" \
    -DAMDGPU_TARGETS="$GPU_ARCH" \
    -DGPU_TARGETS="$GPU_ARCH" \
    -DCMAKE_PREFIX_PATH="$PREFIX;$TORCH_ROOT" \
    -DOPENBABEL3_INCLUDE_DIR="$OB_INCLUDE" \
    -DBUILD_STATIC=ON -DBUILD_SHARED=OFF
  cmake --build "$WORK/libmolgrid/build" -j"$NPROC" --target install
  stage_mark libmolgrid
fi

# ---------------------------------------------------------------------- gnina
say "gnina (HIP backend, $GPU_ARCH)"
cmake -S "$SRC_ROOT" -B "$WORK/gnina" \
  "${CMAKE_COMPAT[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DGNINA_GPU_BACKEND=HIP \
  -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_HIP_ARCHITECTURES="$GPU_ARCH" \
  -DAMDGPU_TARGETS="$GPU_ARCH" \
  -DGPU_TARGETS="$GPU_ARCH" \
  -DCMAKE_PREFIX_PATH="$PREFIX;$TORCH_ROOT" \
  -DGNINA_FORCE_EXTERNAL_LIBS=ON \
  -DGNINA_BUILD_PYTHON=OFF \
  -DGNINA_VERSION="${GNINA_VERSION:-hip-$GPU_ARCH}" \
  -DLIBMOLGRID_LIBRARY="$PREFIX/lib/libmolgrid.a" \
  -DLIBMOLGRID_INCLUDE_DIR="$PREFIX/include"
cmake --build "$WORK/gnina" -j"$NPROC" --target install

say "done"
cat <<EOF
gnina installed to $PREFIX/bin/gnina

Rebuild after editing gnina sources (no reconfigure needed):
  cmake --build $WORK/gnina -j$NPROC --target install

To use it in a job:
  module load CrayEnv rocm
  export PATH=$PREFIX/bin:\$PATH
  # No LD_LIBRARY_PATH needed: the install RPATH covers libtorch and the rest.

Example batch job (one GCD of one MI250X):
  #!/bin/bash
  #SBATCH -A $PROJECT
  #SBATCH -p small-g
  #SBATCH --gpus-per-node=1
  #SBATCH -c 8
  #SBATCH -t 1:00:00
  srun gnina -r rec.pdb -l lig.sdf --autobox_ligand lig.sdf -o docked.sdf.gz

Note --gpu_docking is not available on AMD; see docs/AMD_GPU.md.
EOF
