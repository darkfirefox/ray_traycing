#ifndef BOARDH
#define BOARDH

#include "ray.h"
#include "material.h"

class Board: public Material
{
    public:
    __device__ Board();
    __device__ virtual bool scatter(const Ray &rayIn, const HitRecord &rec, Vec3 &attenuation, Ray &scattered, curandState *localState) const;
};

__device__ Board::Board() {}

__device__ bool Board::scatter(const Ray &rayIn, const HitRecord &rec, Vec3 &attenuation, Ray &scattered, curandState *localState) const
{
    Vec3 target = rec.point + rec.normal;
    Vec3 pt = rec.point;
    scattered = Ray(rec.point, target - rec.point);
    attenuation = (int(.5 * pt.x() + 1000) + int(.5 * pt.z())) & 1 ? Vec3(1,1,1) : Vec3(0.5, .7, .3);
    attenuation = attenuation * .3;
    return true;
}
#endif