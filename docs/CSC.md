# Building gnina on CSC systems (LUMI and Roihu)

The two machines need different backends, for different reasons.

| | LUMI-G | Roihu GPU nodes |
|---|---|---|
| GPU | AMD MI250X | NVIDIA GH200 (H100) |
| Backend | `-DGNINA_GPU_BACKEND=HIP` | `-DGNINA_GPU_BACKEND=CUDA` (default) |
| Target arch | `gfx90a` | `sm_90` |
| Host CPU | AMD EPYC (x86_64) | **Grace, 72 ARM cores (aarch64)** |
| `--gpu_docking` | not available (see [AMD_GPU.md](AMD_GPU.md)) | available |

Two ready-made scripts do the whole thing, dependencies included. Both build
each dependency into a project-space prefix and record a stamp file, so a
re-run after a failure picks up where it stopped rather than starting over.

```bash
# get the source (on a login node; put it in project space, not /users)
cd /project/project_465XXXXXX
git clone https://github.com/4D-Lab/gnina.git && cd gnina

# LUMI-G: fetch on a login node (compute nodes have no internet), then compile
export PROJECT=project_465XXXXXX
FETCH_ONLY=1 ./scripts/build_lumi.sh
salloc -A $PROJECT -p small -c 32 -t 3:00:00     # LUMI-C -- the build needs no GPU
srun ./scripts/build_lumi.sh

# Roihu
export PROJECT=project_2XXXXXX && ./scripts/build_roihu.sh
```

---

## LUMI

LUMI-G is 4× MI250X per node, each card presenting two GCDs, so a node exposes
8 logical GPUs. gnina uses one. Build for `gfx90a` and nothing else — a wider
`CMAKE_HIP_ARCHITECTURES` just multiplies compile time.

### Environment

```bash
module --force purge
module load CrayEnv
module load rocm          # sets ROCM_PATH, provides hipcc and hipify-perl
module load buildtools    # cmake, ninja
```

gnina does not use MPI, so the Cray compiler wrappers buy you nothing here.
Use ROCm's plain `clang++` as the C++ compiler — **not `hipcc`**, which
compiles every `.cpp` with `-x hip` and thereby device-compiles host-only code
in `__host__ __device__` headers while ignoring `CMAKE_HIP_ARCHITECTURES`
(defaulting to gfx906). cmake's HIP language handles the `.cu` files and does
honour the architecture. cmake picks ROCm up automatically from `ROCM_PATH`.

```bash
PREFIX=/project/$PROJECT/gnina-rocm   # where build_lumi.sh puts the dependencies
cmake -S . -B build \
  -DGNINA_GPU_BACKEND=HIP \
  -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_PREFIX_PATH="$PREFIX;$PREFIX/libtorch" \
  -DLIBMOLGRID_LIBRARY=$PREFIX/lib/libmolgrid.a \
  -DLIBMOLGRID_INCLUDE_DIR=$PREFIX/include
```

### Submitting the build as a batch job

An interactive `salloc` is the wrong shape for a build that might fail on its
fifth dependency: every failure leaves you re-queueing. Submit it instead —
`scripts/build_lumi.sbatch` wraps the build script, and stages are stamped, so
a job that hits the wall clock resumes where it stopped when you resubmit.

```bash
FETCH_ONLY=1 ./scripts/build_lumi.sh        # once, on a login node
sbatch --account=$PROJECT scripts/build_lumi.sbatch
tail -f gnina-build-<jobid>.log
```

Worth knowing when you *are* in an `salloc`: a failed `srun` step does not end
the allocation. The step exits non-zero, the allocation survives — check with
`squeue --me` and just re-run `srun` rather than starting over.

### Where to build

Compute nodes have no outbound network access, so the download phase has to run
on a login node — that is what `FETCH_ONLY=1` is for. Running the whole script
on a login node works too (everything is fetched before the first compiler
runs); just keep `NPROC=8` or so, since login nodes are shared.

Not in `/users/$USER` — the home quota is small and gnina's dependency tree
produces a lot of files. Use `/project/$PROJECT` for the install (persistent)
and `/scratch/$PROJECT` if you want the build tree on faster storage.

### Which partition to build on

Building for `gfx90a` is cross-compilation: hipcc never opens a device, and the
script passes `CMAKE_HIP_ARCHITECTURES` explicitly so cmake does not try to
autodetect one. Build on a **CPU** partition (`small` on LUMI-C) — the host
architecture is the same x86 as LUMI-G, and a GPU allocation would spend
GPU-hours on a job that leaves the MI250Xs idle.

`dev-g` also works, but its walltime shrinks as node count grows (2 h at 8
nodes, 1 h at 16, 30 min at 32; check yours with `scontrol show partition
dev-g`) and a project may only have **one running dev-g job at a time**. Save it
for the smoke test after the build:

```bash
salloc -A $PROJECT -p dev-g --gpus-per-node=1 -c 8 -t 0:20:00
srun rocm-smi
srun gnina -r rec.pdb -l lig.sdf --autobox_ligand lig.sdf -o /dev/null --cnn_scoring rescore
```

That last command exercises libtorch on the GPU, which is where a ROCm/libtorch
version mismatch shows up.

### Dependencies

`libtorch` must be a **ROCm** build and `libmolgrid` must be **ported to HIP** —
upstream libmolgrid is CUDA-only, and gnina's HIP build refuses to auto-fetch
it rather than silently produce something that cannot run.

Porting libmolgrid is two jobs, not one. `hipify-perl` rewrites the CUDA
sources, but the build system is untouched: libmolgrid declares
`project(LibMolGrid LANGUAGES C CXX CUDA)` and calls `find_package(CUDA
REQUIRED)`, so cmake hunts for nvcc and fails on a ROCm-only machine.
`scripts/hipify_libmolgrid.sh` does both halves — sources via hipify, then the
CMake files — and finishes with an audit of what a textual translator cannot
decide: thrust/cub use, warp-width assumptions, CUDA-only intrinsics, and
device-side kernel launches. `build_lumi.sh` calls it for you.

```bash
./scripts/hipify_libmolgrid.sh /path/to/libmolgrid   # standalone use
```

Match libtorch's ROCm version to the loaded `rocm` module. The
`shared-with-deps` archive bundles its own ROCm runtime libraries, so a
mismatch puts two different HIP runtimes in the same process. The script
defaults to `2.7.0+rocm6.3`, which pairs with LUMI's `rocm/6.3.4`. If the
module version moves, check what is offered under
<https://download.pytorch.org/libtorch/> and override:

```bash
LIBTORCH_URL=https://download.pytorch.org/libtorch/rocm6.4/libtorch-cxx11-abi-shared-with-deps-2.8.0%2Brocm6.4.zip \
  FETCH_ONLY=1 ./scripts/build_lumi.sh
```

A quick way to test a candidate URL before committing to the download:

```bash
curl -sIL -o /dev/null -w '%{http_code}\n' "$LIBTORCH_URL"
```

### Container alternative

If the native build fights you, containerise instead — LUMI runs
Apptainer/Singularity and this repo already ships a ROCm Dockerfile. You need
root somewhere to build the image, but not on LUMI:

```bash
# on a machine with docker
docker build -t gnina-rocm --build-arg GPU_ARCHS=gfx90a -f docker/rocm-ubuntu-24.04/Dockerfile .
docker save gnina-rocm -o gnina-rocm.tar

# on LUMI (no root needed to convert an archive)
singularity build gnina-rocm.sif docker-archive://gnina-rocm.tar
```

Then run it with the project directories bound in:

```bash
srun ... singularity exec -B /project/$PROJECT,/scratch/$PROJECT \
  gnina-rocm.sif gnina -r rec.pdb -l lig.sdf --autobox_ligand lig.sdf -o out.sdf.gz
```

The container carries its own ROCm userspace; it only has to be new enough for
LUMI's kernel driver, which ROCm 6.x is.

### Verifying the build

`scripts/verify_lumi.sh` checks an installed build. It runs every check it can
and prints a summary, so one run tells you everything that is wrong.

```bash
export PROJECT=project_465XXXXXX
./scripts/verify_lumi.sh                    # static checks, login node is fine
```

Static checks: the expected binaries exist, every shared library resolves, only
one `libamdhip64` is pulled in (libtorch ships its own ROCm runtime, and two HIP
runtimes in one process corrupt device state), and the embedded device code
targets `gfx90a` and nothing else. That last one matters because wrong-arch
device code links without complaint and only fails when a kernel is dispatched.

Then the GPU checks, which need a real MI250X:

```bash
srun -A $PROJECT -p dev-g --gpus-per-node=1 -c 8 -t 20 ./scripts/verify_lumi.sh
```

These run `gninacheck` -- gnina's own test binary, which executes each GPU
kernel and compares the result against the CPU implementation of the same
quantity -- then dock-and-rescore `10gs` twice, once on the GPU and once with
`--no_gpu`, and check the two agree within tolerance.

`gninacheck` is worth running after any change to the `.cu` sources. It is the
only thing that exercises gnina's own HIP kernels: in normal use those are
reachable only through `--gpu_docking`, which is disabled on this backend (see
below), so ordinary docking runs would not catch a regression in them.

### Running


```bash
#!/bin/bash
#SBATCH -A project_465XXXXXX
#SBATCH -p small-g
#SBATCH --gpus-per-node=1
#SBATCH -c 8
#SBATCH -t 1:00:00

module load CrayEnv rocm
export PATH=/project/project_465XXXXXX/gnina-rocm/bin:$PATH

srun gnina -r rec.pdb -l lig.sdf --autobox_ligand lig.sdf -o docked.sdf.gz
```

To avoid repeating the `PATH` line in every job script, put it in `~/.bashrc`
(run this with `PROJECT` set; the double quotes write the expanded path):

```bash
echo "export PATH=/project/$PROJECT/gnina-rocm/bin:\$PATH" >> ~/.bashrc
```

Nothing else is needed — the install RPATH covers libtorch and the rest, so
`gnina` runs from any shell without `LD_LIBRARY_PATH`.

Select a GPU with `HIP_VISIBLE_DEVICES`, not `CUDA_VISIBLE_DEVICES`. Use
`small-g` for single-GPU work and `standard-g` when you want whole nodes; for a
docking campaign, one gnina per GCD with 8 array tasks per node maps cleanly
onto LUMI-G.

---

## Roihu

Roihu's GPU nodes are NVIDIA, so this is the ordinary CUDA build — none of the
HIP work applies. The complication is the CPU: GH200 pairs the H100 with a
Grace ARM CPU, while Roihu's CPU partitions (and most likely its login nodes)
are x86 AMD Turin. **A binary compiled on an x86 login node will not run on a
GPU node.** Check first:

```bash
uname -m                                                   # login node
srun -A $PROJECT -p <gpu-partition> --gres=gpu:1 -t 0:05:00 uname -m
```

If they differ, build from inside an interactive GPU allocation so the whole
toolchain runs aarch64:

```bash
srun -A $PROJECT -p <gpu-partition> --gres=gpu:1 -c 32 -t 4:00:00 --pty bash
./scripts/build_roihu.sh
```

The other aarch64 consequence: `download.pytorch.org` publishes libtorch
archives for x86_64 only. On Grace, take libtorch out of the aarch64 CUDA pip
wheel instead and point cmake at it — which is what `build_roihu.sh` does:

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
cmake -S . -B build \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_PREFIX_PATH="$PREFIX;$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')"
```

Partition names on Roihu are not hard-coded in the script — check `sinfo` or
[the CSC docs](https://docs.csc.fi/computing/systems-roihu/) and substitute.

---

## Things that are likely to need a nudge

The LUMI path has been run end to end: gnina builds, links and installs
against ROCm 6.3.4 for gfx90a, and on an MI250X `gninacheck` passes all five
GPU-versus-CPU kernel comparisons with GPU and CPU scores agreeing. The Roihu
path has not been exercised. The parts most likely to need adjustment:

* **Module names and versions.** `module load rocm` / `module load cuda` take
  the site default. Run `module avail` and pin explicit versions if the default
  moves under you.
* **The hipify pass over libmolgrid.** It is textual. It covers the runtime API
  and plain kernels cleanly, but if libmolgrid has grown anything exotic since
  this was written, expect a handful of manual fixes -- read the audit that
  `hipify_libmolgrid.sh` prints, particularly the warp-width check.
* **libmolgrid's optional subdirectories.** `python/`, `docs/` and `test/` are
  added unconditionally (there is no `BUILD_PYTHON` option), and the bindings
  need `boost_python` plus Python dev headers. The port script comments them
  out; set `LIBMOLGRID_MINIMAL=0` to keep them.
* **libtorch/ROCm version skew** on LUMI, as described above.
* **Duplicate ROCm runtimes.** The libtorch archive bundles its own
  `libamdhip64` and `libhsa-runtime64`. They carry the same SONAMEs as the
  module's, but they are different files, and the dynamic loader deduplicates
  by inode rather than by SONAME — so both get mapped and the process ends up
  with two HIP runtimes. libmolgrid then allocates device memory through one
  and hands the pointer to torch, which reads it through the other. It fails as
  wrong numbers, not as an error, which is what makes it worth ruling out
  up front. `build_lumi.sh` replaces the bundled copies with symlinks to the
  module's (keeping the originals as `*.bundled`), and `verify_lumi.sh` checks
  that nothing is mapped twice. The same treatment applies to `libamd_comgr`,
  which HIP loads code objects through. The compute libraries — rocBLAS,
  MIOpen, hipBLASLt — are deliberately left alone: they ship version-matched
  Tensile data alongside the library, and a duplicate of one is wasteful rather
  than incorrect, since nothing outside torch creates a handle for them and
  symbol interposition means one copy serves every caller anyway. Symlinking is more reliable than reordering
  `LD_LIBRARY_PATH`: it makes every search path lead to the same inode, so the
  result does not depend on whether libtorch's libraries use `DT_RPATH` or
  `DT_RUNPATH`.
* **Roihu login-node architecture**, which has not been confirmed (see the
  `uname -m` check above).
