#include "Boid.h"
#include "Flock.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "Pvector.h"
#include "Options.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cstdint>
#include "timing.h"

// =============================================== //
// ======== Flock Functions from Flock.h ========= //
// =============================================== //

#define SCREEN_LENGTH 5000

#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
                cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__host__ __device__ int Flock::getSize()
{
    return flock.size();
}

// Read only method that returns a copy of the Boid.
__host__ __device__ Boid Flock::getBoid(int i)
{
    return flock[i];
}

__host__ __device__ vector<Boid> Flock::getFlock()
{
    return flock;
}

__host__ __device__ void Flock::addBoid(const Boid &b)
{
    flock.push_back(std::move(b));
}

__device__ int linearHashFn(int cellX, int cellY, int screenLength, int cellSize)
{
    int numCellsHorizontal = screenLength / cellSize;
    int cellIndex = cellY * numCellsHorizontal + cellX;
    return cellIndex;
}

__device__ int linearHash(int x, int y, int screenLength, int cellSize)
{
    int numCellsHorizontal = screenLength / cellSize;
    int cellX = x / cellSize;
    int cellY = y / cellSize;
    return linearHashFn(cellX, cellY, screenLength, cellSize);
}

__device__ uint32_t zOrderCurveHashFn(uint32_t x, uint32_t y)
{
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    return x | (y << 1);
}

__device__ int zOrderCurveHash(int x, int y, int screenLength, int cellSize)
{
    int numCellsHorizontal = screenLength / cellSize;
    int cellX = x / cellSize;
    int cellY = y / cellSize;
    return zOrderCurveHashFn(cellX, cellY);
}

__device__ uint32_t hilbertCurveHashFn(uint32_t x, uint32_t y, int n)
{
    uint32_t rx, ry, s, d = 0;
    for (s = n / 2; s > 0; s /= 2)
    {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        // Rotate
        if (!ry)
        {
            if (rx)
            {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            uint32_t temp = x;
            x = y;
            y = temp;
        }
    }
    return d;
}

__device__ int hilbertCurveHash(int x, int y, int screenLength, int cellSize)
{
    int numCellsHorizontal = screenLength / cellSize;
    int cellX = x / cellSize;
    int cellY = y / cellSize;
    return hilbertCurveHashFn(cellX, cellY, numCellsHorizontal);
}

__global__ void flockingKernel(Boid *flock, Boid *newFlock, int flockSize, Pvector *sortedTuples, int *cellStarts, int *cellEnds, int *hash, int CELL_SIZE)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < flockSize)
    {
        // make copy of boid into new flock
        newFlock[idx] = flock[idx];
        int numCellsHorizontal = SCREEN_LENGTH / CELL_SIZE;
        int cellIndex = hash[idx];
        int startX = max(0, cellIndex % numCellsHorizontal - 1);
        int startY = max(0, cellIndex / numCellsHorizontal - 1);
        int endX = min(numCellsHorizontal - 1, cellIndex % numCellsHorizontal + 1);
        int endY = min(numCellsHorizontal - 1, cellIndex / numCellsHorizontal + 1);
        float desiredSeparation = newFlock[idx].radius / 2;
        int separationCount = 0;
        int neighborCount = 0;
        Pvector separationVector(0, 0);
        Pvector alignmentVector(0, 0);
        Pvector cohesionVector(0, 0);
        Pvector diff(0, 0);

        for (int cellY = startY; cellY <= endY; cellY++)
        {
            for (int cellX = startX; cellX < endX; cellX++)
            {
                int neighborCellSortedIndex = hilbertCurveHashFn(cellX, cellY, numCellsHorizontal);
                int start = cellStarts[neighborCellSortedIndex];
                int end = cellEnds[neighborCellSortedIndex];
                for (int i = start; i < end; i++)
                {
                    int boidId = sortedTuples[i].y;
                    if (boidId != idx)
                    {
                        Boid neighbor = flock[boidId];
                        float d = newFlock[idx].location.distance(neighbor.location);
                        if ((d > 0) && (d < desiredSeparation))
                        {
                            diff = diff.subTwoVector(newFlock[idx].location, neighbor.location);
                            diff.normalize();
                            diff.divScalar(d);
                            separationVector.addVector(diff);
                            separationCount++;
                        }
                        if ((d > 0) && (d < newFlock[idx].radius))
                        {
                            alignmentVector.addVector(neighbor.velocity);
                            cohesionVector.addVector(neighbor.location);
                            neighborCount++;
                        }
                    }
                }
            }
        }
        // printf("neighborCount: %d\n", neighborCount);
        if (separationCount > 0)
        {
            separationVector.divScalar((float)separationCount);
            if (separationVector.magnitude() > 0)
            {
                separationVector.normalize();
                separationVector.mulScalar(newFlock[idx].maxSpeed);
                separationVector.subVector(newFlock[idx].velocity);
                separationVector.limit(newFlock[idx].maxForce);
            }
        }
        if (neighborCount > 0)
        {
            alignmentVector.divScalar((float)neighborCount);
            alignmentVector.normalize();
            alignmentVector.mulScalar(newFlock[idx].maxSpeed);
            alignmentVector.subVector(newFlock[idx].velocity);
            alignmentVector.limit(newFlock[idx].maxForce);

            cohesionVector.divScalar((float)neighborCount);
            cohesionVector = newFlock[idx].seek(cohesionVector);
        }

        separationVector.mulScalar(1.5);
        alignmentVector.mulScalar(1.0);
        cohesionVector.mulScalar(1.0);

        newFlock[idx].applyForce(separationVector);
        newFlock[idx].applyForce(alignmentVector);
        newFlock[idx].applyForce(cohesionVector);

        // newFlock[idx].flock(flock, flockSize);

        newFlock[idx].update();
    }
}

__global__ void calcHashKernel(int *hash, Boid *boids, int screenLength, int numBoids, int cellSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBoids)
    {
        int x = boids[idx].location.x;
        int y = boids[idx].location.y;
        hash[idx] = hilbertCurveHash(x, y, screenLength, cellSize);
        // printf("x: %d, y: %d hash: %d\n", x, y, hash[idx]);
    }
}

__global__ void particleHashKernel(int *hash, Pvector *unsortedTuples, int numBoids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBoids)
    {
        unsortedTuples[idx].x = hash[idx];
        unsortedTuples[idx].y = idx;
    }
}

__global__ void cellStartsKernel(Pvector *sorted, int *cellStarts, int numBoids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBoids)
    {
        if (idx == 0)
        {
            cellStarts[int(sorted[idx].x)] = idx;
        }
        else if (sorted[idx].x != sorted[idx - 1].x)
        {
            cellStarts[int(sorted[idx].x)] = idx;
        }
    }
}

__global__ void cellEndsKernel(Pvector *sorted, int *cellEnds, int numBoids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBoids)
    {
        if (idx == numBoids - 1)
        {
            cellEnds[int(sorted[idx].x)] = idx + 1;
        }
        else if (sorted[idx].x != sorted[idx + 1].x)
        {
            cellEnds[int(sorted[idx].x)] = idx + 1;
        }
    }
}

__host__ void computeGrid(int *hash, Boid *boids, Pvector *unsortedTuples, Pvector *sortedTuples, int *cellStarts, int *cellEnds, int screenLength, int numBoids, int cellSize, int numThreads, int numBlocks)
{
    // Timer hashTimer;
    calcHashKernel<<<numBlocks, numThreads>>>(hash, boids, screenLength, numBoids, cellSize);
    particleHashKernel<<<numBlocks, numThreads>>>(hash, sortedTuples, numBoids);
    // cudaDeviceSynchronize();
    // printf("Hash time: %f\n", hashTimer.elapsed());

    
    // make host hash and host sorted tuples
    int *hashHost = new int[numBoids];
    Pvector *sortedTuplesHost = new Pvector[numBoids];
    cudaMemcpy(hashHost, hash, numBoids * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sortedTuplesHost, sortedTuples, numBoids * sizeof(Pvector), cudaMemcpyDeviceToHost);

    // Sort the tuples based on the hash values (.x component)
    // Timer sortTimer;
    thrust::sort_by_key(hashHost, hashHost + numBoids, sortedTuplesHost);
    // cudaDeviceSynchronize();
    // printf("Sort time: %f\n", sortTimer.elapsed());

    // Copy sorted tuples from host to device
    cudaMemcpy(sortedTuples, sortedTuplesHost, numBoids * sizeof(Pvector), cudaMemcpyHostToDevice);

    // Timer rangeTimer;
    cellStartsKernel<<<numBlocks, numThreads>>>(sortedTuples, cellStarts, numBoids);
    cellEndsKernel<<<numBlocks, numThreads>>>(sortedTuples, cellEnds, numBoids);
    // cudaDeviceSynchronize();
    // printf("Range time: %f\n", rangeTimer.elapsed());
}

__host__ __device__ void Flock::cudaFlocking()
{
    int size = flock.size();

    Boid *deviceFlock;
    int numBytes = size * sizeof(Boid);

    // Allocate device memory for flock
    cudaMalloc((void **)&deviceFlock, numBytes);

    // Allocate device memory for new flock
    Boid *deviceNewFlock;
    cudaMalloc((void **)&deviceNewFlock, numBytes);

    // Allocate device memory for hash
    int *deviceHash;
    cudaMalloc((void **)&deviceHash, size * sizeof(int));

    // Allocate device memory for unsorted tuples
    Pvector *deviceUnsortedTuples;
    cudaMalloc((void **)&deviceUnsortedTuples, size * sizeof(Pvector));

    // Allocate device memory for sorted tuples
    Pvector *deviceSortedTuples;
    cudaMalloc((void **)&deviceSortedTuples, size * sizeof(Pvector));

    // Allocate device memory for cell starts
    int *deviceCellStarts;
    cudaMalloc((void **)&deviceCellStarts, size * sizeof(int));

    // Allocate device memory for cell ends
    int *deviceCellEnds;
    cudaMalloc((void **)&deviceCellEnds, size * sizeof(int));

    // make flock buffer
    Boid *flockBuffer = new Boid[size];
    for (int i = 0; i < size; i++)
    {
        flockBuffer[i] = flock[i];
    }

    // Copy flock from host to device
    cudaMemcpy(deviceFlock, flockBuffer, numBytes, cudaMemcpyHostToDevice);

    // Compute grid
    // printf("threads: %d\n", numThreads);
    int threads = numThreads;
    int numBlocks = (size + threads - 1) / threads;
    computeGrid(deviceHash, deviceFlock, deviceUnsortedTuples, deviceSortedTuples, deviceCellStarts, deviceCellEnds, SCREEN_LENGTH, size, radius * 2, numThreads, numBlocks);

    // Run cudaFlocking kernel
    // Timer flockingTimer;
    flockingKernel<<<numBlocks, numThreads>>>(deviceFlock, deviceNewFlock, size, deviceSortedTuples, deviceCellStarts, deviceCellEnds, deviceHash, radius * 2);
    

    // Synchronize to ensure all CUDA calls are completed
    cudaDeviceSynchronize();
    // printf("Flocking time: %f\n", flockingTimer.elapsed());
    
    // Copy flock from device to host
    cudaMemcpy(flockBuffer, deviceNewFlock, numBytes, cudaMemcpyDeviceToHost);

    // Update flock
    for (int i = 0; i < size; i++)
    {
        flock[i] = flockBuffer[i];
    }

    // Free device memory
    cudaFree(deviceFlock);
    cudaFree(deviceNewFlock);
    cudaFree(deviceHash);
    cudaFree(deviceUnsortedTuples);
    cudaFree(deviceSortedTuples);
    cudaFree(deviceCellStarts);
    cudaFree(deviceCellEnds);

    // Free host memory
    delete[] flockBuffer;
}

// Runs the run function for every boid in the flock checking against the flock
// itself. Which in turn applies all the rules to the flock.
__host__ __device__ void Flock::flocking()
{
    for (int i = 0; i < flock.size(); i++)
        flock[i].run(flock);
}
