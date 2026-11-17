#ifndef GPU_MATH_H
#define GPU_MATH_H
#include <float.h>
#include "array3d.h"

#include "gpu_runtime.h"
#include "device_buffer.h"
#include <random>

#include "common.h"
#include "array3d.h"
#include "gpu_util.h"

/* This exists solely to provide constructor and [] operator
 funcs. Produced binaries are identical to those using vanilla
 float3. */
struct gfloat3 : float3 {
    gfloat3() = default;
    __host__ __device__ __inline__ gfloat3(float3 f)
        : float3(f) {
    }
    __host__ __device__ inline gfloat3(float x, float y, float z)
        : float3(make_float3(x, y, z)) {
    }
    ;
    __host__ __device__ explicit gfloat3(vec v)
        : float3(make_float3(v[0], v[1], v[2])) {
    }
    ;

    __host__ __device__
    float& operator[](int b) {
      return b == 0 ? x : b == 1 ? y : z;
    }
    ;

    __host__ __device__
    const float& operator[](int b) const {
      return b == 0 ? x : b == 1 ? y : z;
    }
    ;

    gfloat3 &operator=(const gfloat3 &b) = default;

    __host__  __device__  inline float3& operator=(const vec &b) {
      x = b[0];
      y = b[1];
      z = b[2];
      return *this;
    }

    //test for equality with gfloat3 or float3
    __host__ __device__ inline bool operator==(const float3& rhs) {
      return x == rhs.x && y == rhs.y && z == rhs.z;
    }

    __host__ __device__ inline bool operator!=(const float3& rhs) {
      return !(*this == rhs);
    }

};

inline std::ostream& operator<<(std::ostream& os, const gfloat3& f) {
  os << f.x <<","<<f.y<<","<<f.z;
  return os;
}

//Both the shuffle and atomicAdd provided below are not strictly what they say
//they are. They are convenience functions that allow, for example, templated
//code to work correctly even if the types are unsupported by CUDA, but they
//work by applying hardware operations separately to individual elements of the
//respective types.

#ifdef GNINA_GPUCC

/* HIP's __shfl_down() defaults its width to the wavefront size, which is what
 * we want.  The *_sync variants take a 32-bit mask that cannot describe a
 * 64-wide wavefront, so they are deliberately not used on AMD. */
#ifdef GNINA_HIP
#define GNINA_SHFL_DOWN(val, offset) __shfl_down((val), (offset))
#elif __CUDACC_VER_MAJOR__ >= 9
#define GNINA_SHFL_DOWN(val, offset) __shfl_down_sync(0xffffffff, (val), (offset))
#else
#define GNINA_SHFL_DOWN(val, offset) __shfl_down((val), (offset))
#endif

__device__ __inline__ float shuffle_down(float val, int offset) {
  return GNINA_SHFL_DOWN(val, offset);
}

__device__  __inline__  static gfloat3 shuffle_down(const gfloat3 &a, int delta) {
  return gfloat3(GNINA_SHFL_DOWN(a.x, delta),
      GNINA_SHFL_DOWN(a.y, delta),
      GNINA_SHFL_DOWN(a.z, delta));
}

template<class T>
__device__  inline static T pseudoAtomicAdd(T* address, T value) {
  return T(atomicAdd(&((*address)[0]), value[0]),
      atomicAdd(&((*address)[1]), value[1]),
      atomicAdd(&((*address)[2]), value[2]),
      atomicAdd(&((*address)[3]), value[3]));
}

#endif

/* Pre-Pascal NVIDIA parts have no native double atomicAdd.  Every AMD target
   HIP supports does (natively or via a compiler-generated CAS loop), so the
   fallback below is CUDA-only. */
#if defined(GNINA_HIP) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ inline double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
				old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

inline bool almostEqual(float a, float b) {
  float absA = std::fabs(a);
  float absB = std::fabs(b);
  float diff = std::fabs(a - b);

  if (a == b)
    return true;
  else
    if (a == 0 || b == 0 || diff < FLT_MIN)
      return diff < (FLT_EPSILON * FLT_MIN);
    else
      return diff / std::min((absA + absB), FLT_MAX) < FLT_EPSILON;
}

__host__ __device__ inline static
float dot(gfloat3 a, gfloat3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline static
float &get(gfloat3 &a, int b) {
  return b == 0 ? a.x : b == 1 ? a.y : a.z;
}

__host__ __device__ inline static
const float &get(const gfloat3 &a, int b) {
  return b == 0 ? a.x : b == 1 ? a.y : a.z;
}

__host__  __device__  inline static gfloat3 operator-(const gfloat3 &a) {
  return gfloat3(-a.x, -a.y, -a.z);
}

__host__  __device__  inline static gfloat3 operator+(const gfloat3 &a,
    const gfloat3 &b) {
  return gfloat3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__  __device__  inline static gfloat3 operator-(const gfloat3 &a,
    const gfloat3 &b) {
  return gfloat3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__  __device__  inline static gfloat3 operator+=(gfloat3 &a,
    const gfloat3 &b) {
  return a = a + b;
}

__host__  __device__  inline static gfloat3 operator*(const gfloat3 &a,
    const gfloat3 &b) {
  return gfloat3(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

template<typename T>
__host__  __device__   inline static gfloat3 operator*(gfloat3 a, T b) {
  return gfloat3(a.x * b, a.y * b, a.z * b);
}

/* Exact-match overload for gfloat3.  HIP's vector types come with their own
   scalar*vector operator; without this the gfloat3 case would be ambiguous
   between that one and the float3 overload below, since both need a
   derived-to-base conversion. */
template<typename T>
__host__  __device__  inline static gfloat3 operator*(T b, const gfloat3 &a) {
  return gfloat3(a.x * b, a.y * b, a.z * b);
}

#ifndef GNINA_HIP
template<typename T>
__host__  __device__  inline static float3 operator*(T b, float3 a) {
  return gfloat3(a.x * b, a.y * b, a.z * b);
}
#endif

template<typename T, typename U>
class array3d_gpu {
    sz i, j, k;
    T* data { };
  public:
    array3d_gpu(const array3d<U>& carr)
        : i(carr.m_i), j(carr.m_j), k(carr.m_k) {
      CUDA_CHECK_GNINA(thread_buffer.alloc(&data, i * j * k * sizeof(T)));
      definitelyPinnedMemcpy(data, &carr.m_data[0],
          sizeof(T) * carr.m_data.size(), cudaMemcpyHostToDevice);
    }

    __device__ sz dim0() const {
      return i;
    }
    __device__ sz dim1() const {
      return j;
    }
    __device__ sz dim2() const {
      return k;
    }
    __device__ sz dim(sz idx) const {
      switch (idx) {
      case 0:
        return i;
      case 1:
        return j;
      case 2:
        return k;
      default:
        assert(false);
        return 0;
      }
    }

    __device__ T& operator()(sz i, sz j, sz k) {
      return data[i + this->i * (j + this->j * k)];
    }
    __device__   const T& operator()(sz i, sz j, sz k) const {
      return data[i + this->i * (j + this->j * k)];
    }
};

#endif
