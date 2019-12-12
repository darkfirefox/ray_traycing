#ifndef LAMBERTIANH
#define LAMBERTIANH

#include "material.h"

class Lambertian : public Material
{
public:
    __device__ Lambertian(const Vec3 &a);
    __device__ virtual bool scatter(const Ray &rayIn, const HitRecord &rec, Vec3 &attenuation, Ray &scattered, curandState *localState) const;
    Vec3 albedo;
};

__device__ Lambertian::Lambertian(const Vec3 &a)
{
    albedo = a;
}

__device__ bool Lambertian::scatter(const Ray &rayIn, const HitRecord &rec, Vec3 &attenuation, Ray &scattered, curandState *localState) const
{
    Vec3 target = rec.point + rec.normal + randomInUnitSphere(localState);
    scattered = Ray(rec.point, target - rec.point);
    attenuation = albedo;
    return true;
}

#endif