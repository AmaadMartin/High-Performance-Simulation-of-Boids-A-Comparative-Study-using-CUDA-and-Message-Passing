#include <iostream>
#include <vector>
#include "Boid.h"

#ifndef FLOCK_H_
#define FLOCK_H_


// Brief description of Flock Class:
// This file contains the class needed to create a flock of boids. It utilizes
// the boids class and initializes boid flocks with parameters that can be
// specified. This class will be utilized in main.

class Flock {
public:
    //Constructors
    __host__ __device__ Flock() {}
    __host__ __device__ Flock(vector<Boid> boids) {
        flock = boids;
        radius = 50;
    }
    // Accessor functions
    __host__ __device__ int getSize();
    //Read only and read/write methods.
    __host__ __device__ Boid getBoid(int i);
    __host__ __device__ vector<Boid> getFlock();
    // Mutator Functions
    __host__ __device__ void addBoid(const Boid& b);
    __host__ __device__ void cudaFlocking();  
    __host__ __device__ void flocking();
private:
    vector<Boid> flock;  
    float radius;
};

#endif
