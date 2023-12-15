#include "Boid.h"
#include "Flock.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "Pvector.h"
#include "Options.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// =============================================== //
// ======== Flock Functions from Flock.h ========= //
// =============================================== //

#define SCREEN_LENGTH 1000

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

__global__ void flockingKernel(Boid *flock, int flockSize, Pvector *sortedTuples, int *cellStarts, int *cellEnds, int *hash, int CELL_SIZE)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("idx: %d\n", idx);
    if (idx < flockSize)
    {
        int numCellsHorizontal = SCREEN_LENGTH / CELL_SIZE;
        int cellIndex = hash[idx];
        int startX = max(0, cellIndex % numCellsHorizontal - 1);
        int startY = max(0, cellIndex / numCellsHorizontal - 1);
        int endX = min(numCellsHorizontal - 1, cellIndex % numCellsHorizontal + 1);
        int endY = min(numCellsHorizontal - 1, cellIndex / numCellsHorizontal + 1);
        float desiredSeparation = flock[idx].radius / 2;
        int separationCount = 0;
        int neighborCount = 0;
        Pvector separationVector(0, 0);
        Pvector alignmentVector(0, 0);
        Pvector cohesionVector(0, 0);
        Pvector diff(0, 0);

        for (int y = startY; y <= endY; y++)
        {
            for (int x = startX; x < endX; x++)
            {
                int neighborCellIndex = x + y * numCellsHorizontal;
                int start = cellStarts[neighborCellIndex];
                int end = cellEnds[neighborCellIndex];
                for (int i = start; i < end; i++)
                {
                    if (i != idx)
                    {
                        float d = flock[idx].location.distance(flock[i].location);
                        if ((d > 0) && (d < desiredSeparation))
                        {
                            diff = diff.subTwoVector(flock[idx].location, flock[i].location);
                            diff.normalize();
                            diff.divScalar(d);
                            separationVector.addVector(diff);
                            separationCount++;
                        }
                        if ((d > 0) && (d < flock[idx].radius))
                        {
                            alignmentVector.addVector(flock[i].velocity);
                            cohesionVector.addVector(flock[i].location);
                            neighborCount++;
                        }
                    }
                }
            }
        }
        if (separationCount > 0)
        {
            separationVector.divScalar((float)separationCount);
            if (separationVector.magnitude() > 0)
            {
                separationVector.normalize();
                separationVector.mulScalar(flock[idx].maxSpeed);
                separationVector.subVector(flock[idx].velocity);
                separationVector.limit(flock[idx].maxForce);
            }
        }
        if (neighborCount > 0)
        {
            alignmentVector.divScalar((float)neighborCount);
            alignmentVector.normalize();
            alignmentVector.mulScalar(flock[idx].maxSpeed);
            alignmentVector.subVector(flock[idx].velocity);
            alignmentVector.limit(flock[idx].maxForce);

            cohesionVector.divScalar((float)neighborCount);
            cohesionVector.divScalar((float)neighborCount);
            cohesionVector = flock[idx].seek(cohesionVector);
        }

        separationVector.mulScalar(1.5);
        alignmentVector.mulScalar(1.0);
        cohesionVector.mulScalar(1.0);

        flock[idx].applyForce(separationVector);
        flock[idx].applyForce(alignmentVector);
        flock[idx].applyForce(cohesionVector);

        // flock[idx].flock(flock, flockSize);
        flock[idx].update();
    }
}

__global__ void calcHashKernel(int *hash, Boid *boids, int screenLength, int numBoids, int cellSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBoids)
    {
        int x = (int)(boids[idx].location.x / cellSize);
        int y = (int)(boids[idx].location.y / cellSize);
        hash[idx] = x + y * screenLength;
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
    calcHashKernel<<<numBlocks, numThreads>>>(hash, boids, screenLength, numBoids, cellSize);
    particleHashKernel<<<numBlocks, numThreads>>>(hash, unsortedTuples, numBoids);

    // Convert raw pointers to Thrust device pointers
    thrust::device_ptr<Pvector> thrust_unsortedTuples(unsortedTuples);
    thrust::device_ptr<Pvector> thrust_sortedTuples(sortedTuples);

    // Copy unsorted tuples to sorted tuples
    thrust::copy(thrust_unsortedTuples, thrust_unsortedTuples + numBoids, thrust_sortedTuples);

    // Sort the tuples based on the hash values (.x component)
    thrust::sort(thrust_sortedTuples, thrust_sortedTuples + numBoids, [] __device__(const Pvector &a, const Pvector &b)
                 { return a.x < b.x; });

    // Copy sorted tuples back to sortedTuples
    thrust::copy(thrust_sortedTuples, thrust_sortedTuples + numBoids, sortedTuples);

    cudaCheckError(cudaDeviceSynchronize());

    cellStartsKernel<<<numBlocks, numThreads>>>(sortedTuples, cellStarts, numBoids);
    cellEndsKernel<<<numBlocks, numThreads>>>(sortedTuples, cellEnds, numBoids);
}

__host__ __device__ void Flock::cudaFlocking()
{
    int size = flock.size();

    Boid *deviceFlock;
    int numBytes = size * sizeof(Boid);

    // Allocate device memory for flock
    cudaMalloc((void **)&deviceFlock, numBytes);

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

    // Compute grid
    int numThreads = 256;
    int numBlocks = (size + numThreads - 1) / numThreads;
    computeGrid(deviceHash, deviceFlock, deviceUnsortedTuples, deviceSortedTuples, deviceCellStarts, deviceCellEnds, SCREEN_LENGTH, size, radius * 2, numThreads, numBlocks);

    // make flock buffer
    Boid *flockBuffer = new Boid[size];
    for (int i = 0; i < size; i++)
    {
        flockBuffer[i] = flock[i];
    }

    // Copy flock from host to device
    cudaMemcpy(deviceFlock, flockBuffer, numBytes, cudaMemcpyHostToDevice);

    // Run cudaFlocking kernel
    flockingKernel<<<numBlocks, numThreads>>>(deviceFlock, size, deviceSortedTuples, deviceCellStarts, deviceCellEnds, deviceHash, radius * 2);

    // Synchronize to ensure all CUDA calls are completed
    cudaDeviceSynchronize();

    // Copy flock from device to host
    cudaMemcpy(flockBuffer, deviceFlock, numBytes, cudaMemcpyDeviceToHost);

    // Update flock
    for (int i = 0; i < size; i++)
    {
        flock[i] = flockBuffer[i];
    }

    // Free device memory
    cudaFree(deviceFlock);
}

// Runs the run function for every boid in the flock checking against the flock
// itself. Which in turn applies all the rules to the flock.
__host__ __device__ void Flock::flocking()
{
    for (int i = 0; i < flock.size(); i++)
        flock[i].run(flock);
}
