#include <iostream>

using namespace std;

#ifndef PVECTOR_H_
#define PVECTOR_H_

// The Pvector class implements Euclidian vectors -- that is, each vector has
// both a magnitude and a direction. We use Pvectors for implementing movement
// and the three Boid rules -- cohesion, separation, and matching velocity
// through the use of acceleration, force, and velocity vectors.

class Pvector {

public:
    float x;
    float y;

    //Constructors
    __host__ __device__ Pvector() {}

    __host__ __device__ Pvector(float xComp, float yComp)
    {
        x = xComp;
        y = yComp;
    }
    //Mutator Functions
    __host__ __device__ void set(float x, float y);

    //Scalar functions scale a vector by a float
    __host__ __device__ void addVector(const Pvector& v);
    __host__ __device__ void addScalar(float x);

    __host__ __device__ void subVector(const Pvector& v);
    __host__ __device__ Pvector subTwoVector(const Pvector& v, const Pvector& v2);
    __host__ __device__ void subScalar(float x);

    __host__ __device__ void mulVector(const Pvector& v);
    __host__ __device__ void mulScalar(float x);

    __host__ __device__ void divVector(const Pvector& v);
    __host__ __device__ void divScalar(float x);

    __host__ __device__ void limit(double max);

    //Calculating Functions
    __host__ __device__ float distance(const Pvector& v);
    __host__ __device__ float dotProduct(const Pvector& v);
    __host__ __device__ float magnitude();
    __host__ __device__ void setMagnitude(float x);
    __host__ __device__ float angleBetween(const Pvector& v);
    __host__ __device__ void normalize();

    __host__ __device__ Pvector copy(const Pvector& v);
};

#endif
