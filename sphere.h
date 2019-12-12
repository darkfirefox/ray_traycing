#ifndef SPHEREH
#define SPHEREH

#include "hittable.h"
#include "helper.h"
#include "material.h"

class Sphere : public Hittable
{
public:
    __device__ Sphere();
    __device__ Sphere(const Vec3 &center, float radius, Material *mat);
    __device__ virtual bool hit(const Ray &ray, float t_min, float t_max, HitRecord &record) const;


    Material *material;
private:
    Vec3 _center;
    float _radius;
};

__device__ Sphere::Sphere()
{
}
__device__ Sphere::Sphere(const Vec3 &center, float radius, Material *mat)
{
    _center = center;
    _radius = radius;
    material = mat;
}

__device__ bool Sphere::hit(const Ray &ray, float t_min, float t_max, HitRecord &record) const
{
    Vec3 oc = ray.origin() - _center;
    float a = dot(ray.direction(), ray.direction());
    float b = dot(oc, ray.direction());
    float c = dot(oc, oc) - _radius * _radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0)
    {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            record.t = temp;
            record.point = ray.pointAtParameter(record.t);
            record.normal = (record.point - _center) / _radius;
            record.material = material;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            record.t = temp;
            record.point = ray.pointAtParameter(record.t);
            record.normal = (record.point - _center) / _radius;
            record.material = material;
            return true;
        }
    }
    return false;
}

#endif