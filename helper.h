#ifndef HELPERH
#define HELPERH

#include <curand_kernel.h>

#include "hittable.h"

__host__ __device__ float dot(const Vec3 &v1, const Vec3 &v2);
__host__ __device__ Vec3 cross(const Vec3 &v1, const Vec3 &v2);
__host__ __device__ Vec3 unit_vector(Vec3 v);


__host__ __device__ float dot(const Vec3 &v1, const Vec3 &v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ Vec3 cross(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ Vec3 unit_vector(Vec3 v)
{
    return v / v.length();
}

#endif