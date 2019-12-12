#ifndef DIELECTRICH
#define DIELECTRICH

#include "material.h"

__device__ float schlick(float cosine, float ref_idx)
{
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const Vec3 &v, const Vec3 &n, float ni_over_nt, Vec3 &refracted)
{
    Vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

class Dielectric : public Material
{
public:
    __device__ Dielectric(float ri);
    __device__ virtual bool scatter(const Ray &rayIn, const HitRecord &rec, Vec3 &attenuation, Ray &scattered, curandState *localState) const;

    float ref_idx;
};

__device__ Dielectric::Dielectric(float ri)
{
    ref_idx = ri;
}

__device__ bool Dielectric::scatter(const Ray &rayIn, const HitRecord &rec, Vec3 &attenuation, Ray &scattered, curandState *localState) const
{
    Vec3 outwardNormal;
    Vec3 reflected = reflect(rayIn.direction(), rec.normal);
    float ni_over_nt;
    attenuation = Vec3(1.0, 1.0, 1.0);
    Vec3 refracted;
    float reflectProb;
    float cosine;
    if (dot(rayIn.direction(), rec.normal) > 0.0f)
    {
        outwardNormal = -rec.normal;
        ni_over_nt = ref_idx;
        cosine = dot(rayIn.direction(), rec.normal) / rayIn.direction().length();
        cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
    }
    else
    {
        outwardNormal = rec.normal;
        ni_over_nt = 1.0f / ref_idx;
        cosine = -dot(rayIn.direction(), rec.normal) / rayIn.direction().length();
    }
    if (refract(rayIn.direction(), outwardNormal, ni_over_nt, refracted))
        reflectProb = schlick(cosine, ref_idx);
    else
        reflectProb = 1.0f;
    if (curand_uniform(localState) < reflectProb)
        scattered = Ray(rec.point, reflected);
    else
        scattered = Ray(rec.point, refracted);
    return true;
}

#endif