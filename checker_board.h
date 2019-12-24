#ifndef CHECKERBOARDH
#define CHECKERBOARDH

#include <float.h>

#include "hittable.h"
#include "ray.h"
#include "material.h"
#include "board.h"

class CheckerBoard: public Hittable 
{
public:
    __device__ CheckerBoard();
    __device__ CheckerBoard(Vec3 &n, Material *mat);
    __device__ virtual bool hit(const Ray &ray, float t_min, float t_max, HitRecord &record) const;

    Vec3 normal;
    Material *material;
};

__device__ CheckerBoard::CheckerBoard() {
    normal = Vec3(0, 1, 0);
    material = new Board();
}

__device__ CheckerBoard::CheckerBoard(Vec3 &n, Material *mat) {
    normal = n;
    material = mat;
}

__device__ bool CheckerBoard::hit(const Ray &ray, float t_min, float t_max, HitRecord &record) const
{
    if (fabsf(ray.direction().y()) > 1e-3)  {
        float temp = -(ray.origin().y() + 4)/ray.direction().y();
        Vec3 pt = ray.pointAtParameter(temp);
        if (temp > t_min && fabs(pt.x()) < 6 && fabs(pt.z()) < 6 && temp < t_max) {
            record.t = temp;
            record.point = pt;
            record.normal = normal;
            record.material = material;
            return true;
        }
    }
    return false;
}

#endif