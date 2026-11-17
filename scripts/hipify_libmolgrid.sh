#!/bin/bash
# Port a libmolgrid checkout to HIP/ROCm.
#
# hipify-perl rewrites CUDA *sources*, but not the build system: libmolgrid's
# CMakeLists declares `project(LibMolGrid LANGUAGES C CXX CUDA)` and calls
# find_package(CUDA REQUIRED), so cmake goes looking for nvcc regardless. This
# script does both halves, then reports anything left that needs a human.
#
#   ./scripts/hipify_libmolgrid.sh /path/to/libmolgrid
#
# Idempotent: each half leaves a marker so re-running is a no-op. Originals are
# kept as *.cuda.bak.
set -euo pipefail

SRC=${1:?usage: hipify_libmolgrid.sh /path/to/libmolgrid}
: "${ROCM_PATH:=/opt/rocm}"
HIPIFY=${HIPIFY:-$ROCM_PATH/bin/hipify-perl}

[[ -x "$HIPIFY" ]] || { echo "hipify-perl not found at $HIPIFY (load the rocm module?)" >&2; exit 1; }
[[ -f "$SRC/CMakeLists.txt" ]] || { echo "$SRC does not look like a libmolgrid checkout" >&2; exit 1; }

cd "$SRC"
say() { printf '\n-- %s\n' "$*"; }

# ------------------------------------------------------------------ sources
SRC_DIRS=()
for d in src include test python; do
  [[ -d "$d" ]] && SRC_DIRS+=("$d")
done
[[ ${#SRC_DIRS[@]} -gt 0 ]] || { echo "no source directories found in $SRC" >&2; exit 1; }

if [[ ! -f .hipified-sources ]]; then
  say "hipifying sources in ${SRC_DIRS[*]}"
  find "${SRC_DIRS[@]}" -type f \
    \( -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.cuh' \) \
    -exec "$HIPIFY" -inplace {} \;
  touch .hipified-sources
else
  say "sources already hipified"
fi

# -------------------------------------------------------------- build system
if [[ ! -f .hipified-cmake ]]; then
  say "porting CMake to HIP"
  cp CMakeLists.txt CMakeLists.txt.cuda.bak

  sed -i -E \
    -e 's/(project[[:space:]]*\([^)]*LANGUAGES[^)]*)CUDA/\1HIP/' \
    -e 's/CMAKE_CUDA_STANDARD/CMAKE_HIP_STANDARD/g' \
    -e '/set[[:space:]]*\([[:space:]]*CMAKE_CUDA_ARCHITECTURES/d' \
    -e '/set[[:space:]]*\([[:space:]]*CMAKE_CUDA_FLAGS_(RELEASE|DEBUG)/d' \
    -e 's/^([[:space:]]*)find_package[[:space:]]*\([[:space:]]*CUDA[[:space:]][^)]*\)/\1find_package(hip REQUIRED)/' \
    CMakeLists.txt

  # -lineinfo and -G are nvcc-only; drop any that survived in generic flags
  sed -i -E 's/(-lineinfo|[[:space:]]-G\b)//g' CMakeLists.txt

  # cmake only maps .cu to CUDA automatically, so name the language explicitly.
  # Must come before the targets are created, hence prepending.
  for d in src test; do
    [[ -f "$d/CMakeLists.txt" ]] || continue
    cu=$(cd "$d" && ls -- *.cu 2>/dev/null | tr '\n' ' ' || true)
    [[ -n "$cu" ]] || continue
    cp "$d/CMakeLists.txt" "$d/CMakeLists.txt.cuda.bak"
    sed -i "1i set_source_files_properties($cu PROPERTIES LANGUAGE HIP)" "$d/CMakeLists.txt"
    echo "   $d: $cu -> LANGUAGE HIP"
  done

  touch .hipified-cmake
else
  say "CMake already ported"
fi

# --------------------------------------------------------------- source fixes
# Incompatibilities hipify does not catch, because they are about C++ semantics
# rather than API names.
if [[ ! -f .hipified-sources-fixed ]]; then
  say "patching HIP source incompatibilities"

  # CUDA's float3/float4 are plain aggregates, so `float3 c = {0,};` is
  # aggregate initialisation. HIP's are class templates whose one-argument
  # constructor is *explicit*, so the same line is ill-formed. Spell out all
  # the components -- valid under both.
  fixed=0
  while IFS= read -r f; do
    [[ -n "$f" ]] || continue
    sed -i -E 's/(\bfloat3[^=;]*=[[:space:]]*)\{[[:space:]]*0[[:space:]]*,?[[:space:]]*\}/\1{0, 0, 0}/g; s/(\bfloat4[^=;]*=[[:space:]]*)\{[[:space:]]*0[[:space:]]*,?[[:space:]]*\}/\1{0, 0, 0, 0}/g' "$f"
    echo "   patched brace-init in $f"
    fixed=1
  done < <(grep -rlIE --exclude='*.prehip' --exclude='*.bak' \
             '\bfloat[34][^=;]*=[[:space:]]*\{[[:space:]]*0[[:space:]]*,?[[:space:]]*\}' \
             "${SRC_DIRS[@]}" 2>/dev/null || true)
  [[ $fixed -eq 1 ]] || echo "   no float3/float4 brace-init to patch"

  # .cpp files call hipMalloc/hipFree, so they need the HIP headers, the
  # __HIP_PLATFORM_AMD__ define and libamdhip64. hip::host carries all three.
  if ! grep -q 'link_libraries(hip::host)' CMakeLists.txt; then
    sed -i -E 's|^([[:space:]]*)find_package\(hip REQUIRED\)|\1find_package(hip REQUIRED)\n\1link_libraries(hip::host)|' CMakeLists.txt
    echo "   added link_libraries(hip::host)"
  fi

  touch .hipified-sources-fixed
fi

# --------------------------------------------------- device-code source fixes
# nvcc tolerates host-only constructs inside __host__ __device__ functions as
# long as the device path is never really used; clang/HIP rejects them
# outright. These are the resulting differences.
if [[ ! -f .hipified-devicecode ]]; then
  say "patching device-code incompatibilities"

  # 1. Grid::check_index throws std::out_of_range (with std::string
  #    concatenation) from a __host__ __device__ function. Route the device
  #    path to a trap instead; the host path is unchanged.
  if [[ -f include/libmolgrid/grid.h ]] && ! grep -q LMG_RANGE_ERROR include/libmolgrid/grid.h; then
    cp include/libmolgrid/grid.h include/libmolgrid/grid.h.cuda.bak
    python3 - <<'PYEOF'
import re
p = 'include/libmolgrid/grid.h'
s = open(p).read()
# Rewrite the throw sites FIRST -- doing it after inserting the macro would
# rewrite the macro's own body into an infinite self-reference. Match a single
# statement (no ';' inside) rather than using DOTALL, which would span lines.
s, n = re.subn(r'throw\s+std::out_of_range\s*\(([^;]*)\)\s*;',
               r'LMG_RANGE_ERROR(\1);', s)
macro = '''
/* Bounds failures cannot throw from device code: clang/HIP rejects throw and
   std::string in __host__ __device__ functions. Trap on the device, throw on
   the host. Added by gnina's scripts/hipify_libmolgrid.sh. */
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#define LMG_RANGE_ERROR(msg) __builtin_trap()
#else
#define LMG_RANGE_ERROR(msg) throw std::out_of_range(msg)
#endif

'''
m = re.search(r'^namespace libmolgrid', s, re.M)
s = s[:m.start()] + macro + s[m.start():] if m else macro + s
open(p, 'w').write(s)
print('   rewrote %d throw site(s)' % n)
PYEOF
    echo "   grid.h: bounds check traps on device instead of throwing"
  fi

  # 2. Out-of-line definitions in the .cu files drop the CUDA_CALLABLE_MEMBER
  #    that their declarations carry. nvcc lets this slide; clang treats the
  #    definition as __host__ only and reports an overload conflict.
  for f in src/*.cu; do
    [[ -f "$f" ]] || continue
    changed=0
    for fn in get_bounds_1d calc_point type_grad_grad atom_density_grad_grad \
              atom_density_grad_grad_other density_grad_dist \
              accumulate_atom_gradient interpolate; do
      if grep -qE "^[[:space:]]*[A-Za-z_][A-Za-z0-9_:<>,& *]*[ *&](GridMaker|GridInterpolater)::$fn\\b" "$f"; then
        sed -i -E "s|^([[:space:]]*)([A-Za-z_][A-Za-z0-9_:<>,\& *]*[ *\&])((GridMaker\|GridInterpolater)::$fn\\b)|\\1CUDA_CALLABLE_MEMBER \\2\\3|" "$f"
        changed=1
      fi
    done
    [[ $changed -eq 1 ]] && echo "   $f: restored CUDA_CALLABLE_MEMBER on out-of-line definitions"
  done

  # 3. grid_interpolater.cu uses CUDA texture objects and cudaMalloc3DArray,
  #    whose HIP counterpart takes an extra flags argument. gnina never calls
  #    GridInterpolater, so drop the device half rather than port textures.
  if [[ -f src/CMakeLists.txt ]] && grep -q 'grid_interpolater.cu' src/CMakeLists.txt; then
    sed -i 's/grid_interpolater\.cu//' src/CMakeLists.txt
    sed -i 's/grid_interpolater\.cu //' src/CMakeLists.txt
    echo "   dropped grid_interpolater.cu (textures; unused by gnina)"
  fi

  touch .hipified-devicecode
fi

# ------------------------------------------------------- trim optional subdirs
# libmolgrid adds python/, docs/ and test/ unconditionally -- there is no
# BUILD_PYTHON option. gnina links only the library, and the bindings drag in
# boost_python plus Python dev headers that a minimal HPC prefix will not have.
# Set LIBMOLGRID_MINIMAL=0 to keep them, or restore CMakeLists.txt.cuda.bak.
LIBMOLGRID_MINIMAL=${LIBMOLGRID_MINIMAL:-1}
if [[ "$LIBMOLGRID_MINIMAL" == "1" && ! -f .hipified-subdirs ]]; then
  say "disabling python/docs/test subdirectories"
  for sub in python docs test; do
    if grep -qE "^[[:space:]]*add_subdirectory[[:space:]]*\([[:space:]]*$sub[[:space:]]*\)" CMakeLists.txt; then
      sed -i -E "s|^([[:space:]]*)(add_subdirectory[[:space:]]*\([[:space:]]*$sub[[:space:]]*\))|\1#\2 # disabled for the gnina HIP build|" CMakeLists.txt
      echo "   disabled add_subdirectory($sub)"
    fi
  done
  touch .hipified-subdirs
fi

# ------------------------------------------------------------------- audit
# Things hipify cannot decide for you. None of these are necessarily wrong --
# they are the places to look first if the grids come out wrong.
say "audit"

warn=0

# One grep decides *and* reports, so a flag always comes with the lines that
# caused it. Skip our own backups and anything that is not compiled.
audit() {
  local label=$1 pattern=$2 hits
  # -I skips binary files: test fixtures are full of byte sequences that
  # look like source patterns and produce phantom flags.
  hits=$(grep -rnIE --exclude='*.prehip' --exclude='*.bak' --exclude='*.cuda.bak' \
           "$pattern" "${SRC_DIRS[@]}" 2>/dev/null \
         | grep -vE '\.(md|rst|txt|json|ipynb):' | head -12 || true)
  [[ -n "$hits" ]] || return 0
  warn=1
  printf '   !! %s\n' "$label"
  printf '%s\n' "$hits" | sed 's/^/      /'
}

audit "uses thrust/cub -- needs rocThrust/hipCUB (shipped with ROCm; same headers and namespace)" \
      '#include <(thrust|cub)/'

audit "warp-width assumptions -- AMD wavefronts are 64 wide on gfx90a, NVIDIA warps are 32" \
      '\bwarpSize\b|\b__shfl|& *31\b|>> *5\b'

audit "CUDA-specific intrinsics or inline PTX" \
      '\b(__ldg|__syncwarp|__activemask|__ballot_sync|asm +volatile)\b'

audit "kernel launched from device code (dynamic parallelism) -- unsupported on AMD" \
      '__device__[^;]*<<<'

# thrust on ROCm comes from rocThrust; if it is missing the build fails with a
# confusing "no such file" on a header hipify never touched.
if grep -rqIE --exclude='*.prehip' '#include <(thrust|cub)/' "${SRC_DIRS[@]}" 2>/dev/null; then
  if [[ -f "$ROCM_PATH/include/thrust/version.h" ]]; then
    echo "   rocThrust present: $ROCM_PATH/include/thrust"
  else
    warn=1
    printf '   !! rocThrust not found under %s/include/thrust -- install it or load a fuller rocm module\n' "$ROCM_PATH"
  fi
fi

if [[ $warn -eq 0 ]]; then
  echo "   nothing flagged"
fi

say "done -- now configure with:"
cat <<EOF
  cmake -S $SRC -B $SRC/build \\
    -DCMAKE_CXX_COMPILER=\$ROCM_PATH/llvm/bin/clang++ \\
    -DCMAKE_HIP_ARCHITECTURES=gfx90a \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DBUILD_STATIC=ON -DBUILD_SHARED=OFF \\
    -DCMAKE_INSTALL_PREFIX=<prefix>
EOF
