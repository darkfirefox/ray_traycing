#ifndef HITTABLEH
#define HITTABLEH

#include "ray.h"

class Material;

struct HitRecord
{
    float t;
    Vec3 point;
    Vec3 normal;
    Material *material;
};

class Hittable
{
public:
    __device__ virtual bool hit(const Ray &ray, float t_min, float t_max, HitRecord &record) const = 0;
};

#endif
