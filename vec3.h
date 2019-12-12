#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <iostream>

class Vec3
{
public:
    __host__ __device__ Vec3();
    __host__ __device__ Vec3(float x, float y, float z);

    __host__ __device__ float x();
    __host__ __device__ float y();
    __host__ __device__ float z();

    __host__ __device__ const Vec3 &operator+() { return *this; }
    __host__ __device__ Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float &operator[](int i) { return e[i]; };

    __host__ __device__ Vec3 &operator+=(const Vec3 &v);
    __host__ __device__ Vec3 &operator-=(const Vec3 &v);
    __host__ __device__ Vec3 &operator*=(const Vec3 &v);
    __host__ __device__ Vec3 &operator/=(const Vec3 &v);
    __host__ __device__ Vec3 &operator*=(const float a);
    __host__ __device__ Vec3 &operator/=(const float a);

    __host__ __device__ friend Vec3 operator+(const Vec3 &v1, const Vec3 &v2);
    __host__ __device__ friend Vec3 operator-(const Vec3 &v1, const Vec3 &v2);
    __host__ __device__ friend Vec3 operator*(const Vec3 &v1, const Vec3 &v2);
    __host__ __device__ friend Vec3 operator/(const Vec3 &v1, const Vec3 &v2);
    __host__ __device__ friend Vec3 operator*(float a, const Vec3 &v);
    __host__ __device__ friend Vec3 operator/(Vec3 v, float a);
    __host__ __device__ friend Vec3 operator*(const Vec3 &v, float a);

    friend std::istream &operator>>(std::istream &in, Vec3 &v);
    friend std::ostream &operator<<(std::ostream &out, const Vec3 &v);

    __host__ __device__ float length() const;
    __host__ __device__ float pow2Length() const;
    __host__ __device__ void norm();

    float e[3];
};

__host__ __device__ Vec3::Vec3()
{
    e[0] = 0;
    e[1] = 0;
    e[2] = 0;
}
__host__ __device__ Vec3::Vec3(float x, float y, float z)
{
    e[0] = x;
    e[1] = y;
    e[2] = z;
}

__host__ __device__ float Vec3::x() { return e[0]; }
__host__ __device__ float Vec3::y() { return e[1]; }
__host__ __device__ float Vec3::z() { return e[2]; }

__host__ __device__ Vec3 operator+(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}
__host__ __device__ Vec3 operator-(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}
__host__ __device__ Vec3 operator*(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}
__host__ __device__ Vec3 operator/(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}
__host__ __device__ Vec3 operator*(float a, const Vec3 &v)
{
    return Vec3(a * v.e[0], a * v.e[1], a * v.e[2]);
}
__host__ __device__ Vec3 operator/(Vec3 v, float a)
{
    return Vec3(v.e[0] / a, v.e[1] / a, v.e[2] / a);
}
__host__ __device__ Vec3 operator*(const Vec3 &v, float a)
{
    return Vec3(v.e[0] * a, v.e[1] * a, v.e[2] * a);
}

__host__ __device__ Vec3 &Vec3::operator+=(const Vec3 &v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}
__host__ __device__ Vec3 &Vec3::operator-=(const Vec3 &v)
{
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}
__host__ __device__ Vec3 &Vec3::operator*=(const Vec3 &v)
{
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}
__host__ __device__ Vec3 &Vec3::operator/=(const Vec3 &v)
{
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}
__host__ __device__ Vec3 &Vec3::operator*=(const float a)
{
    e[0] *= a;
    e[1] *= a;
    e[2] *= a;
    return *this;
}
__host__ __device__ Vec3 &Vec3::operator/=(const float a)
{
    e[0] /= a;
    e[1] /= a;
    e[2] /= a;
    return *this;
}

__host__ __device__ float Vec3::length() const
{
    return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
}
__host__ __device__ float Vec3::pow2Length() const
{
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}
__host__ __device__ void Vec3::norm()
{
    float kNorm = 1 / length();
    e[0] *= kNorm;
    e[1] *= kNorm;
    e[2] *= kNorm;
}

std::istream &operator>>(std::istream &in, Vec3 &v)
{
    in >> v.e[0] >> v.e[1] >> v.e[2];
    return in;
}
std::ostream &operator<<(std::ostream &out, const Vec3 &v)
{
    out << v.e[0] << " " << v.e[1] << " " << v.e[2];
    return out;
}


#endif