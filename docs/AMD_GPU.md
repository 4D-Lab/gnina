# Building and running gnina on AMD GPUs (ROCm/HIP)

gnina builds against either the CUDA runtime (NVIDIA) or the HIP runtime
(AMD/ROCm). The GPU sources are shared between the two: `gninasrc/lib/gpu_runtime.h`
is the single place where CUDA runtime names are re-exported onto HIP, and it
also defines the portability macros the rest of the tree uses
(`GNINA_DEVICE_COMPILE`, `GNINA_WARPSIZE`, `GNINA_DEVICE_LAUNCH_SUPPORTED`).
Selecting a backend is a cmake option; nothing else in the build changes.

```bash
cmake .. -DGNINA_GPU_BACKEND=HIP -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++
```

`GNINA_GPU_BACKEND` defaults to `CUDA`, so existing NVIDIA builds are unaffected.

Building on CSC's LUMI? See [CSC.md](CSC.md) for a LUMI-specific recipe
(`scripts/build_lumi.sh`) covering the module stack, hipified libmolgrid and
ROCm libtorch.

## Requirements

| | |
|---|---|
| ROCm | 5.7 or newer (6.x recommended) |
| C++ compiler | ROCm's `clang++` — **not `hipcc`**, see below |
| cmake | 3.25+ (HIP language support needs 3.21+) |
| libtorch | a **ROCm** build, matching your ROCm version |
| libmolgrid | a **HIP** build — see [libmolgrid](#libmolgrid) |
| GPU | anything ROCm supports: CDNA (MI100/MI210/MI250/MI300) or RDNA2/RDNA3 |

The C++ compiler must be ROCm's **`clang++`**, not `hipcc`. Two reasons:

* `hipcc` compiles every `.cpp` with `-x hip`, which device-compiles host-only
  code that happens to sit inside `__host__ __device__` headers, and ignores
  `CMAKE_HIP_ARCHITECTURES` for those files (silently falling back to gfx906).
  Let cmake's HIP language handle the `.cu` sources instead -- it honours the
  architecture setting.
* gnina calls `__device__` functions across translation units, so it needs
  relocatable device code (`-fgpu-rdc`), and those objects must be device-linked
  with `--hip-link` -- a clang driver flag. cmake warns if you point it at gcc.

## Full build

```bash
mkdir build && cd build
cmake .. \
  -DGNINA_GPU_BACKEND=HIP \
  -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DLIBMOLGRID_LIBRARY=/usr/local/lib/libmolgrid.a \
  -DLIBMOLGRID_INCLUDE_DIR=/usr/local/include
make -j$(nproc)
```

`docker/rocm-ubuntu-24.04/Dockerfile` does all of this from a clean
`rocm/dev-ubuntu-24.04` base, including hipifying libmolgrid.

### Choosing architectures

By default cmake builds for whatever it detects on the build machine, which is
what you want when compiling for your own card. For a redistributable binary:

```bash
cmake .. -DGNINA_GPU_BACKEND=HIP -DGNINA_HIP_ALL_ARCHS=ON
```

which covers `gfx906 gfx908 gfx90a gfx942 gfx1030 gfx1100 gfx1101 gfx1102`.
Each extra architecture is another full device-code compile, so this is slow.

### libtorch

CNN scoring runs through libtorch, so libtorch must target the same GPU vendor
as gnina. If cmake cannot find a system torch it downloads one, and in HIP mode
it downloads the ROCm build. Pin a different one with:

```bash
cmake .. -DGNINA_GPU_BACKEND=HIP \
  -DGNINA_LIBTORCH_URL="https://download.pytorch.org/libtorch/rocm6.2/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Brocm6.2.zip"
```

A CUDA libtorch in a HIP build will configure and even link, then fail at
runtime when `torch::cuda::is_available()` returns false and every molecule
falls back to CPU scoring.

### libmolgrid

Upstream [libmolgrid](https://github.com/gnina/libmolgrid) is CUDA-only. The
HIP build of gnina therefore refuses to auto-fetch and build it — that would
silently produce a library that cannot run on an AMD card — and instead asks
you to supply one. Hipifying the sources is not enough on its own -- the
CMakeLists declares the CUDA language, and HIP's `float3` is not an aggregate --
so `scripts/hipify_libmolgrid.sh` ports the sources and the build system, then
reports anything left that needs a human:

```bash
git clone https://github.com/gnina/libmolgrid.git
./scripts/hipify_libmolgrid.sh $PWD/libmolgrid
cmake -S libmolgrid -B libmolgrid/build \
  -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DAMDGPU_TARGETS=gfx90a \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_STATIC=ON -DBUILD_SHARED=OFF \
  -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build libmolgrid/build -j$(nproc) --target install
```

Replace `gfx90a` with your GPU's architecture (`rocminfo | grep gfx`). The
script is idempotent and keeps the originals as `*.cuda.bak`.

## Running

```bash
gnina -r receptor.pdb -l ligand.sdf --autobox_ligand ligand.sdf -o docked.sdf
```

Nothing about the command line changes. Two environment differences:

* Select a GPU with `HIP_VISIBLE_DEVICES`, not `CUDA_VISIBLE_DEVICES`.
* On RDNA cards and on some APUs, ROCm may need
  `HSA_OVERRIDE_GFX_VERSION` set to a supported nearby architecture.

## Verifying a build

This port has been verified on an AMD MI250X (gfx90a, ROCm 6.3.4): all five
`gninacheck` cases pass, and CNN scores from the GPU match the `--no_gpu` run.

`gninacheck`, built alongside gnina, runs each GPU kernel and compares its
output against the CPU implementation of the same quantity:

```bash
./test/gnina/gninacheck -- --n_iters 1
```

Run it after any change to the `.cu` sources. It is the only thing that
exercises gnina's own HIP kernels — in normal use they are reachable only
through `--gpu_docking`, which is disabled on this backend (see below), so a
regression in them would not show up in ordinary docking runs.

Two things worth checking on the installed binaries as well, because both fail
late and confusingly rather than at build time:

```bash
# device code is built for the GPU you actually have -- a mismatch links
# cleanly and only fails when a kernel is dispatched
strings -a bin/gnina | grep -o 'gfx[0-9a-f]\{3,4\}[a-z]*' | sort -u

# exactly one HIP runtime is loaded -- the libtorch archive ships its own ROCm
# libraries, and two runtimes in one process corrupt device state
ldd bin/gnina | grep libamdhip64
```

`scripts/verify_lumi.sh` does all of this plus a GPU-vs-CPU score comparison;
it is written for LUMI but the checks themselves are not LUMI-specific.

## What is not supported on AMD

**`--gpu_docking` is unavailable in the HIP build.** That flag runs the whole
BFGS minimizer inside a single kernel, which re-enters the scoring kernels by
launching child kernels from device code — CUDA dynamic parallelism. ROCm has
no equivalent, and there is no way to emulate it without restructuring the
minimizer. gnina exits with an explanatory message if you pass the flag.

This is a narrow loss. `--gpu_docking` is flagged experimental upstream
(`main.cpp` prints "experimental and not recommended" for it, and the
device-side code path carries an unresolved `assert(0)` about missing
synchronisation). Everything gnina normally uses the GPU for is unaffected:

* CNN scoring and rescoring (the default `--cnn_scoring`) — via libtorch
* grid generation in `gninagrid` — via libmolgrid
* the GPU scoring-function evaluation used by `--score_only` and minimization
* `gninavis` CNN visualisation

The corresponding sources are compiled out on HIP behind
`GNINA_DEVICE_LAUNCH_SUPPORTED`, so the CUDA build keeps the feature.

## Notes on the port

**Wavefront width.** NVIDIA warps are always 32 threads. AMD wavefronts are 64
on GCN/CDNA and 32 on RDNA, so the width can no longer be a hard-coded
constant. Device code gets the exact value from the compiler
(`__AMDGCN_WAVEFRONT_SIZE__`) as a compile-time constant; host code, which only
uses it to size thread blocks, gets 64 — a multiple of both possible widths, so
`ROUND_TO_WARP` always yields whole wavefronts. The shuffle-based reductions in
`gpucode.cu` and `conf_gpu.cu` were rewritten in terms of `WARPSIZE` instead of
literal `32`/`31`/`>>5`.

**Shuffles.** On HIP the reductions use `__shfl_down` rather than
`__shfl_down_sync`: the `_sync` variants take a 32-bit lane mask, which cannot
describe a 64-wide wavefront.

**Per-thread default stream.** gnina runs one docking thread per CPU worker and
depends on each getting its own default stream. The HIP build passes
`-fgpu-default-stream=per-thread`, the equivalent of nvcc's
`--default-stream per-thread`.

**Struct layout.** `atom_params` and `force_energy_tup` pack a `float3` next to
a scalar and rely on `float3` being three tightly packed floats.
`gpu_runtime.h` static-asserts this rather than trusting it, since HIP's vector
types are class templates rather than plain structs — a silent layout change
would corrupt every energy and gradient computed on the GPU.
