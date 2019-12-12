#ifndef RAYH
#define RAYH

#include "vec3.h"

class Ray
{
public:
    __device__ Ray();
    __device__ Ray(const Vec3 &org, const Vec3 &dir);
    __device__ Vec3 origin() const;
    __device__ Vec3 direction() const;
    __device__ Vec3 pointAtParameter(float t) const;

private:
    Vec3 _origin;
    Vec3 _direction;
};

__device__ Ray::Ray()
{
}
__device__ Ray::Ray(const Vec3 &org, const Vec3 &dir)
{
    _origin = org;
    _direction = dir;
}
__device__ Vec3 Ray::origin() const { return _origin; }
__device__ Vec3 Ray::direction() const { return _direction; }
__device__ Vec3 Ray::pointAtParameter(float t) const { return _origin + t * _direction; }

#endif