#ifndef METALH
#define METALH

#include "material.h"

class Metal : public Material
{
public:
    __device__ Metal(const Vec3 &a, float f);
    __device__ virtual bool scatter(const Ray &rayIn, const HitRecord &rec, Vec3 &attenuation, Ray &scattered, curandState *localState) const;

    Vec3 albedo;
    float fuzz;
};

__device__ Metal::Metal(const Vec3 &a, float f)
{
    albedo = a;
    fuzz = f < 0.6 ? f : 0.6;
}

__device__ bool Metal::scatter(const Ray &rayIn, const HitRecord &rec, Vec3 &attenuation, Ray &scattered, curandState *localState) const
{
    Vec3 reflected = reflect(unit_vector(rayIn.direction()), rec.normal);
    scattered = Ray(rec.point, reflected + fuzz * randomInUnitSphere(localState));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
}

#endif