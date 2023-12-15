#include "Boid.h"
#include "CudaFlock.h"
#include <cuda.h>
#include <cuda_runtime.h>


// =============================================== //
// ======== Flock Functions from CudaFlock.h ========= //
// =============================================== //

int CudaFlock::getSize()
{
    return flock.size();
}

Boid CudaFlock::getBoid(int i)
{
    return flock[i];
}

vector<Boid> CudaFlock::getFlock()
{
    return flock;
}

void CudaFlock::addBoid(const Boid& b)
{
    flock.push_back(std::move(b));
}

__global__ void cudaFlocking(Boid* flock, int flockSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < flockSize) {
        flock[idx].run(flock, flockSize);
    }
}

void CudaFlock::flocking() 
{
    int size = flock.size();

    Boid* deviceFlock;
    int numBytes = size * sizeof(Boid);
    
    // Allocate device memory for flock
    cudaMalloc((void**)&deviceFlock, numBytes);
    
    // Copy flock from host to device
    cudaMemcpy(deviceFlock, flock.data(), numBytes, cudaMemcpyHostToDevice);
    
    // Run cudaFlocking kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    cudaFlocking<<<numBlocks, blockSize>>>(deviceFlock, size);
    
    // Copy flock from device to host
    cudaMemcpy(flock.data(), deviceFlock, numBytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(deviceFlock);
}
