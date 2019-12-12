#ifndef HITABLELISTH
#define HITABLELISTH

#include "hittable.h"

class HittableList : public Hittable
{
public:
    __device__ HittableList();
    __device__ HittableList(Hittable **list, unsigned int length);
    __device__ virtual bool hit(const Ray &ray, float t_min, float t_max, HitRecord &record) const;

    Hittable **_list;
    unsigned int _length;
};

__device__ HittableList::HittableList() {}

__device__ HittableList::HittableList(Hittable **list, unsigned int length)
{
    _list = list;
    _length = length;
}

__device__ bool HittableList::hit(const Ray &ray, float t_min, float t_max, HitRecord &record) const
{
    HitRecord rec;
    bool hitted = false;
    float maxDistance = t_max;
    for (int i = 0; i < _length; i++)
    {
        if (_list[i]->hit(ray, t_min, maxDistance, rec))
        {
            hitted = true;
            maxDistance = rec.t;
            record = rec;
        }
    }
    return hitted;
}

#endif