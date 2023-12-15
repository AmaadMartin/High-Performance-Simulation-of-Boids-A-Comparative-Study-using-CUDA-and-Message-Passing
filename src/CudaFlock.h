#include <iostream>
#include <vector>
#include "Boid.h"
#include "Flock.h"

#ifndef CUDAFLOCK_H_
#define CUDAFLOCK_H_

class CudaFlock {
public:
    //Constructors
    __host__ __device__ CudaFlock() {}
    __host__ __device__ CudaFlock(vector<Boid> boids) {
        flock = boids;
    } 
    // Accessor functions
    __host__ __device__ int getSize();
    //Read only and read/write methods.
    __host__ __device__ Boid getBoid(int i);
    __host__ __device__ vector<Boid> getFlock();
    // Mutator Functions
    __host__ __device__ void addBoid(const Boid& b);
    __host__ __device__ void flocking();
private:
    vector<Boid> flock;
};

#endif
