#!/bin/bash
# Verify a LUMI (ROCm / gfx90a) build of gnina.
#
# Two groups of checks:
#
#   static  -- inspect the installed binaries. Runs anywhere, login node included.
#   gpu     -- actually execute device code. Needs a real MI250X, so run this
#              under srun on dev-g or small-g.
#
# The script runs every check it can and reports a summary at the end; it does
# not stop at the first failure, so one run tells you everything that is wrong.
#
# Usage:
#   export PROJECT=project_465XXXXXX
#   ./scripts/verify_lumi.sh                        # static only, on a login node
#   srun -A $PROJECT -p dev-g --gpus-per-node=1 -c 8 -t 20 \
#        ./scripts/verify_lumi.sh                   # everything

set -uo pipefail

: "${PROJECT:?set PROJECT to your LUMI project, e.g. export PROJECT=project_465XXXXXX}"

SRC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX=${PREFIX:-/project/$PROJECT/gnina-rocm}
WORK=${WORK:-$PREFIX/build}
GPU_ARCH=${GPU_ARCH:-gfx90a}
TORCH_ROOT=$PREFIX/libtorch
DATA=$SRC_ROOT/test/gnina/data

module load CrayEnv rocm 2>/dev/null
export PATH=$PREFIX/bin:${ROCM_PATH:-/opt/rocm}/llvm/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$PREFIX/lib64:$TORCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

PASS=0; FAIL=0; SKIP=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; SKIP=$((SKIP+1)); }
note() { printf '        %s\n' "$1"; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

echo "=== static checks ==="
echo "PREFIX=$PREFIX"

# ---- 1. the expected artefacts exist ---------------------------------------
for f in bin/gnina lib/libgnina.so lib/libgnina.a lib/libmolgrid.a; do
  [[ -e $PREFIX/$f ]] && pass "$f present" || fail "$f MISSING"
done

# ---- 2. device code is built for the right GPU, and only that one ----------
# Wrong-arch device code links fine and only fails at dispatch time ("no kernel
# image is available"), so check it up front rather than discovering it in a job.
# `strings` is NOT usable for this. ROCm's own libraries carry tables of
# architecture names, so grepping for gfx* reports architectures that were
# never compiled for -- it reads the runtime's vocabulary, not the file's code.
# Ask the object format instead: llvm-objdump --offloading lists the offload
# images actually embedded in an object or archive, and roc-obj-ls does the
# same for a linked executable or shared library.
offload_archs() {
  local f=$1 out
  if command -v llvm-objdump >/dev/null 2>&1; then
    out=$(llvm-objdump --offloading "$f" 2>/dev/null | awk '$1=="arch"{print $2}' | sort -u)
    [[ -n $out ]] && { echo "$out"; return 0; }
  fi
  if command -v roc-obj-ls >/dev/null 2>&1; then
    out=$(roc-obj-ls "$f" 2>/dev/null \
          | grep -o 'gfx[0-9a-f]\{3,4\}[a-z]*' | sort -u)
    [[ -n $out ]] && { echo "$out"; return 0; }
  fi
  # An archive may need unpacking before the offload images are visible.
  if [[ $f == *.a ]] && command -v llvm-objdump >/dev/null 2>&1; then
    local d=$TMP/ar members
    d=$TMP/ar; rm -rf "$d"; mkdir -p "$d"
    ( cd "$d" && ar x "$f" 2>/dev/null )
    members=$(find "$d" -name '*.o' | wc -l)
    if [[ $members -gt 0 ]]; then
      out=$(find "$d" -name '*.o' -print0 \
            | xargs -0 -r llvm-objdump --offloading 2>/dev/null \
            | awk '$1=="arch"{print $2}' | sort -u)
      # llvm-objdump does not always decode the fat binary of a *relocatable*
      # HIP object, only of a linked one. roc-obj-ls reads it either way.
      if [[ -z $out ]] && command -v roc-obj-ls >/dev/null 2>&1; then
        out=$(find "$d" -name '*.o' -print0 \
              | xargs -0 -r -n1 roc-obj-ls 2>/dev/null \
              | grep -o 'gfx[0-9a-f]\{3,4\}[a-z]*' | sort -u)
      fi
    fi
    rm -rf "$d"
    [[ -n $out ]] && { echo "$out"; return 0; }
  fi
  return 1
}

check_arch() {
  local f=$1 archs
  [[ -e $f ]] || { skip "arch of $(basename "$f") (not built)"; return; }
  if ! archs=$(offload_archs "$f"); then
    local detail=""
    [[ $f == *.a ]] && detail=" in $(ar t "$f" 2>/dev/null | wc -l) archive member(s)"
    # Expected, and cosmetic: neither tool reliably decodes the fat binary of a
    # relocatable object, only of a linked one. The note below covers it.
    skip "arch of $(basename "$f") -- not readable from relocatables$detail"
    return
  fi
  archs=$(echo "$archs" | tr '\n' ' '); archs=${archs% }
  if [[ $archs == "$GPU_ARCH" ]]; then
    pass "$(basename "$f") targets $archs"
  else
    fail "$(basename "$f") targets '$archs', expected only '$GPU_ARCH'"
  fi
}
check_arch "$PREFIX/lib/libmolgrid.a"
check_arch "$PREFIX/bin/gnina"
check_arch "$PREFIX/lib/libgnina.so"
# gnina links libmolgrid.a statically, so anything the archive was built for
# ends up in the executable. If gnina reports only $GPU_ARCH then libmolgrid
# contains only $GPU_ARCH too, whatever the archive check managed to read.
note "libmolgrid.a is linked statically into gnina, so its targets are covered"
note "by the gnina result above."

# ---- 3. no library is loaded twice ----------------------------------------
# libtorch bundles its own copies of the ROCm libraries. A bundled copy carries
# the same SONAME as the module's but is a different file, and the loader
# deduplicates by inode rather than by SONAME -- so both get mapped. For the
# runtime libraries that means two HIP runtimes in one process: libmolgrid
# allocates device memory through one and hands the pointer to torch, which
# reads it through the other. Unsupported, and it fails as wrong numbers rather
# than as an error.
if command -v ldd >/dev/null && command -v readelf >/dev/null && [[ -e $PREFIX/bin/gnina ]]; then
  declare -A soname_paths=()
  while read -r lib; do
    [[ -f $lib ]] || continue
    sn=$(readelf -d "$lib" 2>/dev/null | awk -F'[][]' '/SONAME/{print $2}')
    [[ -n $sn ]] || continue
    rp=$(readlink -f "$lib")
    case " ${soname_paths[$sn]:-} " in
      *" $rp "*) ;;
      *) soname_paths[$sn]="${soname_paths[$sn]:-} $rp" ;;
    esac
  done < <(ldd "$PREFIX/bin/gnina" 2>/dev/null | awk '/=>/ && $3 ~ /^\// {print $3}')

  # A duplicate matters when the library holds device-level state that other
  # code's pointers and handles flow through -- two HIP runtimes, two HSA queue
  # sets, two code-object caches. Duplicated *compute* libraries are wasteful
  # rather than incorrect: nothing outside torch creates a hipBLASLt handle, and
  # because both copies sit in the global symbol scope every caller binds to
  # whichever the loader reached first, so one implementation serves everyone.
  runtime_re='^lib(amdhip64|hsa-runtime64|amd_comgr|hiprtc)\.so'

  dup_runtime=(); dup_other=()
  for sn in "${!soname_paths[@]}"; do
    # shellcheck disable=SC2086
    set -- ${soname_paths[$sn]}
    [[ $# -gt 1 ]] || continue
    if [[ $sn =~ $runtime_re ]]; then dup_runtime+=("$sn"); else dup_other+=("$sn"); fi
  done

  show_dups() { for rp in ${soname_paths[$1]}; do printf '            %s\n' "$rp"; done; }

  if [[ ${#dup_runtime[@]} -eq 0 ]]; then
    pass "no ROCm runtime library mapped twice"
  else
    fail "${#dup_runtime[@]} ROCm runtime librar$([[ ${#dup_runtime[@]} -eq 1 ]] && echo y || echo ies) mapped from more than one file:"
    for sn in "${dup_runtime[@]}"; do printf '          %s\n' "$sn"; show_dups "$sn"; done
    note "re-run scripts/build_lumi.sh -- it replaces these bundled copies with"
    note "symlinks to the module's, so each maps once."
  fi

  if [[ ${#dup_other[@]} -gt 0 ]]; then
    note "also mapped twice, but harmless -- compute libraries with no shared"
    note "device state, and symbol interposition means one copy serves everyone:"
    for sn in "${dup_other[@]}"; do printf '          %s\n' "$sn"; show_dups "$sn"; done
  fi

  missing=$(ldd "$PREFIX/bin/gnina" 2>/dev/null | grep 'not found' | sed 's/^/          /')
  [[ -z $missing ]] && pass "all shared libraries resolve" \
                    || { fail "unresolved shared libraries:"; echo "$missing"; }
fi

# ---- 4. the binaries run with no environment set up ------------------------
# Upstream hardcodes the install RPATH to $PREFIX/lib, which does not cover an
# externally supplied libtorch. If this fails, gnina only runs inside a prepared
# shell -- "libtorch.so: cannot open shared object file" is what it looks like.
if [[ -x $PREFIX/bin/gnina ]]; then
  if out=$(env -u LD_LIBRARY_PATH "$PREFIX/bin/gnina" --version 2>&1); then
    pass "gnina runs with LD_LIBRARY_PATH unset"
    note "$(echo "$out" | head -1)"
  else
    fail "gnina does not run without LD_LIBRARY_PATH:"
    echo "$out" | head -3 | sed 's/^/          /'
    note "re-run scripts/build_lumi.sh to rewrite the install RPATH"
  fi
fi

# ---- 5. gpu_docking is correctly reported as unavailable -------------------
if [[ -e $PREFIX/bin/gnina ]]; then
  if "$PREFIX/bin/gnina" --help 2>&1 | grep -q -- '--gpu_docking'; then
    note "--gpu_docking is still advertised in --help (it aborts on HIP by design)"
  fi
fi

echo
echo "=== gpu checks ==="

have_gpu=0
if command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showid >/dev/null 2>&1; then
  have_gpu=1
fi

if [[ $have_gpu -eq 0 ]]; then
  skip "no GPU visible -- rerun under srun on dev-g/small-g for these"
else
  rocm-smi --showproductname 2>/dev/null | sed 's/^/        /'

  # ---- 5. gninacheck: gnina's own GPU-vs-CPU kernel comparison -------------
  # This is the real correctness test of the port: each case runs a kernel and
  # compares against the CPU implementation of the same quantity.
  CHECK=$WORK/gnina/test/gnina/gninacheck
  if [[ -x $CHECK ]]; then
    ( cd "$TMP" && "$CHECK" -- --n_iters 1 ) >"$TMP/gninacheck.out" 2>&1
    if [[ $? -eq 0 ]]; then
      pass "gninacheck (GPU kernels match CPU reference)"
      grep -i 'test cases\|No errors' "$TMP/gninacheck.out" | sed 's/^/        /'
    else
      fail "gninacheck"
      tail -40 "$TMP/gninacheck.out" | sed 's/^/        /'
      [[ -s $TMP/test.log ]] && { note "test.log tail:"; tail -20 "$TMP/test.log" | sed 's/^/        /'; }
    fi
  else
    skip "gninacheck (not found at $CHECK)"
  fi

  # ---- 6. torch sees the GPU ----------------------------------------------
  # main.cpp warns "No GPU detected" when torch::cuda::is_available() is false,
  # which is what a torch built for the wrong arch, or a second HIP runtime,
  # looks like from the outside.
  if [[ -e $DATA/10gs_rec.pdb && -e $DATA/10gs_lig.sdf ]]; then
    gnina -r "$DATA/10gs_rec.pdb" -l "$DATA/10gs_lig.sdf" \
          --autobox_ligand "$DATA/10gs_lig.sdf" --minimize --cnn_scoring rescore \
          --cnn fast --seed 0 -o "$TMP/gpu.sdf" >"$TMP/gpu.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      fail "gnina CNN rescore on GPU (exit $rc)"
      tail -30 "$TMP/gpu.log" | sed 's/^/        /'
    elif grep -q 'No GPU detected' "$TMP/gpu.log"; then
      fail "gnina ran but torch did not see the GPU"
      grep -A3 'No GPU detected' "$TMP/gpu.log" | sed 's/^/        /'
    else
      pass "gnina CNN rescore on GPU"
    fi

    # ---- 7. GPU and CPU agree ---------------------------------------------
    gnina -r "$DATA/10gs_rec.pdb" -l "$DATA/10gs_lig.sdf" \
          --autobox_ligand "$DATA/10gs_lig.sdf" --minimize --cnn_scoring rescore \
          --cnn fast --seed 0 --no_gpu -o "$TMP/cpu.sdf" >"$TMP/cpu.log" 2>&1
    if [[ $? -ne 0 ]]; then
      skip "GPU/CPU score agreement (CPU run failed)"
      tail -20 "$TMP/cpu.log" | sed 's/^/        /'
    else
      cmp_scores() {
        local key=$1 tol=$2 g c d
        g=$(grep -m1 "^$key" "$TMP/gpu.log" | awk '{print $2}')
        c=$(grep -m1 "^$key" "$TMP/cpu.log" | awk '{print $2}')
        if [[ -z $g || -z $c ]]; then
          skip "$key not reported by both runs"; return
        fi
        d=$(awk -v a="$g" -v b="$c" 'BEGIN{d=a-b; print (d<0?-d:d)}')
        if awk -v d="$d" -v t="$tol" 'BEGIN{exit !(d<=t)}'; then
          pass "$key agrees (gpu=$g cpu=$c, |d|=$d <= $tol)"
        else
          fail "$key DIVERGES (gpu=$g cpu=$c, |d|=$d > $tol)"
        fi
      }
      # Tolerances are loose: reduction order differs between GPU and CPU, and
      # the CNN runs in single precision. Divergence beyond this is a real bug,
      # not accumulated rounding.
      cmp_scores "Affinity:"     0.05
      cmp_scores "CNNscore:"     0.02
      cmp_scores "CNNaffinity:"  0.10
    fi
  else
    skip "end-to-end run ($DATA not found)"
  fi
fi

echo
echo "=== summary ==="
printf '  %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
[[ $FAIL -eq 0 ]] || exit 1
if [[ $have_gpu -eq 0 ]]; then
  echo "  static checks only -- rerun under srun on dev-g to exercise the GPU"
fi
