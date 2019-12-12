#ifndef MATERIALH
#define MATERIALH

#include "ray.h"
#include "hittable.h"
#include "helper.h"

#define RANDVEC3 Vec3(curand_uniform(localState), curand_uniform(localState), curand_uniform(localState))

__device__ Vec3 randomInUnitSphere(curandState *localState)
{
    Vec3 p;
    do
    {
        p = 2.0f * RANDVEC3 - Vec3(1, 1, 1);
    } while (p.pow2Length() >= 1.0f);
    return p;
}

__device__ Vec3 reflect(const Vec3 &v, const Vec3 &n)
{
    return v - 2.0f * dot(v, n) * n;
}

class Material
{
public:
     __device__ virtual bool scatter(const Ray& rayIn, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState *localState) const = 0;
};

#endif