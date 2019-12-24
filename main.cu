#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <time.h>
#include <float.h>

#include "helper.h"
#include "cuda_helper.h"
#include "hittable_list.h"
#include "sphere.h"

int main()
{
    int width = 1200;
    int height = 600;
    int tx = 16;
    int ty = 16;
    int ns = 100;
    dim3 blocsDim(width / tx + 1, height / ty + 1);
    dim3 threadsDim(tx, ty);
    curandState *d_state, *d_state_cw;

    std::cerr << "Rendering a " << width << "x" << height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    size_t fb_size = width * height * sizeof(Vec3);

    Vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    checkCudaErrors(cudaMalloc((void **)&d_state, width * height *sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&d_state_cw, 1*sizeof(curandState)));

    curandInit<<<1,1>>>(d_state_cw);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Hittable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, (SPHERE_COUNTS + PLANE_COUNTS) * sizeof(Hittable *)));
    Hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(Hittable *)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
    createWorld<<<1, 1>>>(d_list, d_world, d_camera, width, height, d_state_cw);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    renderInit<<<blocsDim, threadsDim>>>(width, height, d_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocsDim, threadsDim>>>(fb, width, height, ns, d_camera, d_world, d_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    writeImage(fb, height, width);

    checkCudaErrors(cudaDeviceSynchronize());
    freeWorld<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();
}