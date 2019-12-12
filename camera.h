#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"

class Camera
{
public:
    __device__ Camera();
    __device__ Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vUp, float vfov, float aspect, float focusDist);
    __device__ Ray getRay(float u, float v);

    Vec3 _origin;
    Vec3 _blCorner;
    Vec3 _hort;
    Vec3 _vert;
};

__device__ Camera::Camera()
{
    _blCorner = Vec3(-2.0, -1.0, -1.0);
    _hort = Vec3(4.0, 0.0, 0.0);
    _vert = Vec3(0.0, 2.0, 0.0);
    _origin = Vec3(0.0, 0.0, 0.0);
}

__device__ Camera::Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vUp, float vfov, float aspect, float focusDist)
{
    Vec3 u, v, w;
    float theta = vfov * M_PI / 180;
    float halfHeight = tan(theta / 2);
    float halfWidth = aspect * halfHeight;
    _origin = lookFrom;
    w = unit_vector(lookFrom - lookAt);
    u = unit_vector(cross(vUp, w));
    v = cross(w, u);
    _blCorner = _origin - halfWidth * u * focusDist - halfHeight * v * focusDist - w;
    _hort = 2 * halfWidth * u * focusDist;
    _vert = 2 * halfHeight * v * focusDist;
}

__device__ Ray Camera::getRay(float u, float v)
{
    return Ray(_origin, _blCorner + u * _hort + v * _vert - _origin);
}

#endif