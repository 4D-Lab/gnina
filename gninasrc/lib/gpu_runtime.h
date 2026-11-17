/*
 * gpu_runtime.h -- single point of contact with the GPU runtime.
 *
 * gnina is written against the CUDA runtime API.  To also target AMD GPUs we
 * build the same sources with HIP (ROCm), which is API-compatible with CUDA
 * modulo a systematic "cuda" -> "hip" renaming.  Rather than hipify the tree
 * (which would drop NVIDIA support), this header pulls in the correct runtime
 * and, on the HIP side, re-exports the handful of CUDA names the codebase
 * actually uses.
 *
 * Everything that used to `#include <cuda_runtime.h>` should include this
 * instead.
 *
 * Portability macros defined here:
 *
 *   GNINA_HIP              defined when building for the HIP/ROCm backend
 *   GNINA_GPUCC            defined when the GPU compiler (nvcc/hipcc) is
 *                          processing this translation unit, in either the
 *                          host or the device pass.  Replaces __CUDACC__.
 *   GNINA_DEVICE_COMPILE   defined during a *device* compilation pass.
 *                          Replaces __CUDA_ARCH__ in #ifdef tests.
 *   GNINA_WARPSIZE         warp/wavefront width (see the note below)
 *   GNINA_MAX_WARPS_PER_BLOCK  upper bound on warps in a 1024-thread block
 *   GNINA_DEVICE_LAUNCH_SUPPORTED  defined iff kernels may be launched from
 *                          device code (CUDA dynamic parallelism).  AMD does
 *                          not implement this.
 */

#ifndef GNINA_GPU_RUNTIME_H
#define GNINA_GPU_RUNTIME_H

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) || defined(USE_HIP)
#ifndef GNINA_USE_HIP
#define GNINA_USE_HIP 1
#endif
#endif

#ifdef GNINA_USE_HIP
#define GNINA_HIP 1
#endif

/* ------------------------------------------------------------------ */
/* runtime headers                                                     */
/* ------------------------------------------------------------------ */

#ifdef GNINA_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime.h>
#endif

/* ------------------------------------------------------------------ */
/* compilation-pass detection                                          */
/* ------------------------------------------------------------------ */

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GNINA_GPUCC 1
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define GNINA_DEVICE_COMPILE 1
#endif

/* Kernels can only be launched from device code under CUDA dynamic
   parallelism; ROCm has no equivalent.  Code guarded by this macro must have
   a host-side fallback. */
#ifndef GNINA_HIP
#define GNINA_DEVICE_LAUNCH_SUPPORTED 1
#endif

/* Execution-space qualifier for functions that launch kernels and are called
   from both host and device code.  Where device-side launches are impossible
   such a function can only be a host function. */
#ifdef GNINA_DEVICE_LAUNCH_SUPPORTED
#define GNINA_DEVICE_LAUNCHER __host__ __device__
#else
#define GNINA_DEVICE_LAUNCHER __host__
#endif

/* ------------------------------------------------------------------ */
/* warp / wavefront width                                              */
/* ------------------------------------------------------------------ */

/* NVIDIA warps are 32 threads.  AMD wavefronts are 64 on GCN/CDNA (gfx8/gfx9,
 * e.g. MI100/MI210/MI300) and 32 on RDNA (gfx10/gfx11/gfx12).  The device
 * compiler knows which, so device code gets the exact value as a compile-time
 * constant.  Host code -- which only uses the value to pick launch geometry --
 * gets 64, the conservative choice: it is a multiple of both possible widths,
 * so a block sized with it is always a whole number of wavefronts.
 */
#ifdef GNINA_HIP
#ifdef GNINA_DEVICE_COMPILE
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
#define GNINA_WARPSIZE __AMDGCN_WAVEFRONT_SIZE__
#elif defined(__AMDGCN_WAVEFRONT_SIZE)
#define GNINA_WARPSIZE __AMDGCN_WAVEFRONT_SIZE
#else
#define GNINA_WARPSIZE 64
#endif
#else /* host pass */
#define GNINA_WARPSIZE 64
#endif
#else /* CUDA */
#define GNINA_WARPSIZE 32
#endif

/* Blocks are capped at 1024 threads throughout gnina, so the number of warps
   in a block never exceeds 1024/32.  Used to size shared-memory scratch. */
#define GNINA_MAX_WARPS_PER_BLOCK 32

/* ------------------------------------------------------------------ */
/* CUDA -> HIP name mapping                                            */
/* ------------------------------------------------------------------ */

#ifdef GNINA_HIP

/* types */
using cudaError_t = hipError_t;
using cudaError = hipError_t;
using cudaMemcpyKind = hipMemcpyKind;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaDeviceProp = hipDeviceProp_t;

/* enumerators */
constexpr cudaError_t cudaSuccess = hipSuccess;
constexpr cudaMemcpyKind cudaMemcpyHostToHost = hipMemcpyHostToHost;
constexpr cudaMemcpyKind cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr cudaMemcpyKind cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
constexpr cudaMemcpyKind cudaMemcpyDefault = hipMemcpyDefault;
constexpr hipDeviceAttribute_t cudaDevAttrComputeMode =
    hipDeviceAttributeComputeMode;
constexpr int cudaComputeModeProhibited = hipComputeModeProhibited;
constexpr unsigned cudaHostAllocDefault = hipHostMallocDefault;

/* cudaStreamPerThread / hipStreamPerThread are macros casting a magic
   handle, so alias rather than redeclare. */
#define cudaStreamPerThread hipStreamPerThread

/* API entry points.  Macros (not wrappers) so that they work unchanged in
   __host__ __device__ context and with the runtime's own overloads. */
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetErrorString hipGetErrorString

#endif /* GNINA_HIP */

/* ------------------------------------------------------------------ */
/* qualifier macros for host-only compilation                          */
/* ------------------------------------------------------------------ */

/* CUDA's headers define __align__ even when a .cpp is compiled by the plain
 * host compiler; HIP's define it only under hipcc. gpucode.h and tree_gpu.h
 * use it directly on struct declarations, so supply it when it is missing.
 * (__host__/__device__/__global__ are already handled by both runtimes.)
 */
#if !defined(__align__)
#define __align__(n) __attribute__((aligned(n)))
#endif

/* ------------------------------------------------------------------ */
/* layout assumptions                                                  */
/* ------------------------------------------------------------------ */

/* gnina packs 3-vectors next to a scalar and relies on float3 being three
 * tightly packed floats (see atom_params/force_energy_tup in gpucode.h, and
 * the (vec*) casts in model.cu).  Both CUDA and HIP define float3 this way,
 * but HIP's vector types are class templates rather than plain structs, so
 * assert it rather than trust it -- a silent layout change would corrupt
 * every energy and gradient the GPU computes.
 */
static_assert(sizeof(float3) == 3 * sizeof(float),
    "float3 is not tightly packed; gnina's atom_params/force_energy_tup "
    "packing assumptions do not hold on this GPU runtime");
static_assert(sizeof(float4) == 4 * sizeof(float),
    "float4 is not tightly packed; gnina's 16-byte alignment assumptions "
    "do not hold on this GPU runtime");

#endif /* GNINA_GPU_RUNTIME_H */
