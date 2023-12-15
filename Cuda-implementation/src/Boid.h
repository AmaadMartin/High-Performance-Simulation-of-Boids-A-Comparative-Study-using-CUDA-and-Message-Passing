#include "Pvector.h"
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BOID_H_
#define BOID_H_

// The Boid Class
//
// Attributes
//  bool predator: flag that specifies whether a given boid is a predator.
//  Pvector location: Vector that specifies a boid's location.
//  Pvector velocity: Vector that specifies a boid's current velocity.
//  Pvector acceleration: Vector that specifies a boid's current acceleration.
//  float maxSpeed: Limits magnitude of velocity vector.
//  float maxForce: Limits magnitude of acceleration vector. (F = m*a!)
//
// Methods
//  applyForce(Pvector force): Adds the given vector to acceleration
//
//  Pvector Separation(vector<Boid> Boids): If any other boids are within a
//      given distance, Separation computes a a vector that distances the
//      current boid from the boids that are too close.
//
//  Pvector Alignment(vector<Boid> Boids): Computes a vector that causes the
//      velocity of the current boid to match that of boids that are nearby.
//
//  Pvector Cohesion(vector<Boid> Boids): Computes a vector that causes the
//      current boid to seek the center of mass of nearby boids.

class Boid {
public:
    Pvector location;
    Pvector velocity;
    Pvector acceleration;
    float maxSpeed;
    float maxForce;
    float radius;
    int id;
    
    __host__ __device__ Boid();
    __host__ __device__ void applyForce(const Pvector& force);
    __host__ __device__ Pvector Separation(Boid b);
    __host__ __device__ Pvector Separation(Boid* flock, int flockSize);
    __host__ __device__ Pvector Separation(const vector<Boid>& Boids);
    __host__ __device__ Pvector Alignment(Boid b);
    __host__ __device__ Pvector Alignment(Boid* flock, int flockSize);
    __host__ __device__ Pvector Alignment(const vector<Boid>& Boids);
    __host__ __device__ Pvector Cohesion(Boid b);
    __host__ __device__ Pvector Cohesion(Boid* flock, int flockSize);
    __host__ __device__ Pvector Cohesion(const vector<Boid>& Boids);
    __host__ __device__ Pvector seek(const Pvector& v);
    __host__ __device__ void run(Boid* v, int flockSize);
    __host__ __device__ void run(const vector<Boid>& v);
    __host__ __device__ void update();
    __host__ __device__ void bound();
    __host__ __device__ void flock(Boid b);
    __host__ __device__ void flock(Boid* v, int flockSize);
    __host__ __device__ void flock(const vector<Boid>& v);
    __host__ __device__ float angle(const Pvector& v);

};

#endif
