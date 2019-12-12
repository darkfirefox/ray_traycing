#ifndef CUDAHELPERH
#define CUDAHELPERH

#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "metal.h"
#include "lambertian.h"
#include "dielectric.h"

#define MAX_RECURSION 10

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#define RND (curand_uniform(&localState))

void writeImage(Vec3 *fb, unsigned int height, unsigned int width);
__global__ void renderInit(int maxX, int maxY, curandState *state);
__global__ void render(Vec3 *fb, int maxX, int maxY, int ns,
                       Camera **cam,
                       Hittable **world,
                       curandState *state);
__global__ void createWorld(Hittable **d_list, Hittable **d_world, Camera **d_camera, int width, int height,
                            curandState *state);
__global__ void freeWorld(Hittable **d_list, Hittable **d_world, Camera **d_camera);
__device__ Vec3 color(const Ray &ray, Hittable **world, curandState *localState);
__global__ void curandInit(curandState *state);

void writeImage(Vec3 *fb, unsigned int height, unsigned int width)
{
    printf("P3\n%u %u\n255\n", width, height);
    for (int j = height - 1; j >= 0; j--)
    {
        for (int i = 0; i < width; i++)
        {
            size_t index = j * width + i;
            int ir = int(255.99 * fb[index].x());
            int ig = int(255.99 * fb[index].y());
            int ib = int(255.99 * fb[index].z());
            printf("%d %d %d\n", ir, ig, ib);
        }
    }
}

__global__ void renderInit(int maxX, int maxY, curandState *state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (!(i < maxX && j < maxY))
    {
        return;
    }
    int index = j * maxX + i;
    curand_init(1984, index, 0, &state[index]);
}

__global__ void render(Vec3 *fb, int maxX, int maxY, int ns,
                       Camera **cam,
                       Hittable **world,
                       curandState *state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (!(i < maxX && j < maxY))
        return;
    int index = j * maxX + i;
    curandState localState = state[index];
    Vec3 resultColor(0, 0, 0);
    for (int s = 0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&localState)) / float(maxX);
        float v = float(j + curand_uniform(&localState)) / float(maxY);
        resultColor += color((*cam)->getRay(u, v), world, &localState);
    }
    state[index] = localState;
    resultColor /= float(ns);
    resultColor[0] = sqrt(resultColor[0]);
    resultColor[1] = sqrt(resultColor[1]);
    resultColor[2] = sqrt(resultColor[2]);
    fb[index] = resultColor;
}

__global__ void createWorld(Hittable **d_list, Hittable **d_world, Camera **d_camera, int width, int height,
                            curandState *state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState localState = *state;
        d_list[0] = new Sphere(Vec3(0,-1000,-1), 1000,
                                             new Lambertian(Vec3(0.2, 0.2, 0.2)));
        int i = 1;
        int counts = 22 * 22 + 1;
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float chooseMat = RND;
                Vec3 center(a + RND, 0.2, b + RND);
                if (chooseMat < 0.4f)
                {
                    d_list[i++] = new Sphere(center, RND,
                                             new Lambertian(Vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (chooseMat < 0.8f)
                {
                    d_list[i++] = new Sphere(center, RND,
                                             new Metal(Vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else
                {
                    d_list[i++] = new Sphere(center, RND, new Dielectric(1.5));
                }
            }
        }
        *d_world = new HittableList(d_list, counts);
        Vec3 lookFrom(13, 2, 3);
        Vec3 lookAt(0, 0, 0);
        float distToFocus = 1.0;
        (lookFrom - lookAt).length();
        *d_camera = new Camera(lookFrom,
                               lookAt,
                               Vec3(0, 1, 0),
                               30.0,
                               float(width) / float(height),
                               distToFocus);
    }
}

__global__ void freeWorld(Hittable **d_list, Hittable **d_world, Camera **d_camera)
{
    int counts = 22 * 22 + 1;
    for (int i = 0; i < counts; i++)
    {
        delete ((Sphere *)d_list[i])->material;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__device__ Vec3 color(const Ray &ray, Hittable **world,
                      curandState *localState)
{
    Ray curRay = ray;
    Vec3 curAttenuation = Vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < MAX_RECURSION; i++)
    {
        HitRecord rec;
        if ((*world)->hit(curRay, 0.001f, FLT_MAX, rec))
        {
            Ray scattered;
            Vec3 attenuation;
            if (rec.material->scatter(curRay, rec, attenuation, scattered, localState))
            {
                curAttenuation *= attenuation;
                curRay = scattered;
            }
            else
            {
                return Vec3(0.0, 0.0, 0.0);
            }
        }
        else
        {
            Vec3 unitDirection = unit_vector(curRay.direction());
            float t = 0.5f * (unitDirection.y() + 1.0f);
            Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
            return curAttenuation * c;
        }
    }
    return Vec3(0.0, 0.0, 0.0);
}

__global__ void curandInit(curandState *state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, state);
    }
}

#endif