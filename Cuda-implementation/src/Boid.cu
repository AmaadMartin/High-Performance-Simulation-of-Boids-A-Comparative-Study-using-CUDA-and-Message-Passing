// #include <iostream>
// #include <vector>
// #include <string>
// #include <math.h>
// #include "Boid.h"
// #include "Pvector.h"

// #define SPAWN_WIDTH 1000
// #define MAX_VELOCITY 1
// #define MAX_SPEED 3.5
// #define MAX_FORCE 0.5
// #define PI 3.141592635

// // =============================================== //
// // ======== Boid Functions from Boid.h =========== //
// // =============================================== //

// __device__ Boid::Boid()
// {
//     acceleration = Pvector(0, 0);
//     velocity = Pvector(0, 0);
//     location = Pvector(0, 0);
//     maxSpeed = MAX_SPEED;
//     maxForce = MAX_FORCE;
//     radius = 50;
// }

// __device__ void Boid::applyForce(const Pvector& force)
// {
//     acceleration.addVector(force);
// }

// __device__ Pvector Boid::Separation(const Boid* boids, int numBoids)
// {
//     float desiredseparation = radius / 2;
//     Pvector steer(0, 0);
//     int count = 0;
//     for (int i = 0; i < numBoids; i++) {
//         float d = location.distance(boids[i].location);
//         if ((d > 0) && (d < desiredseparation)) {
//             Pvector diff(0,0);
//             diff = diff.subTwoVector(location, boids[i].location);
//             diff.normalize();
//             diff.divScalar(d);
//             steer.addVector(diff);
//             count++;
//         }
//     }
//     if (count > 0)
//         steer.divScalar((float)count);
//     if (steer.magnitude() > 0) {
//         steer.normalize();
//         steer.mulScalar(maxSpeed);
//         steer.subVector(velocity);
//         steer.limit(maxForce);
//     }
//     return steer;
// }

// __device__ Pvector Boid::Alignment(const Boid* boids, int numBoids)
// {
//     float neighbordist = radius;
//     Pvector sum(0, 0);
//     int count = 0;
//     for (int i = 0; i < numBoids; i++) {
//         float d = location.distance(boids[i].location);
//         if ((d > 0) && (d < neighbordist)) {
//             sum.addVector(boids[i].velocity);
//             count++;
//         }
//     }
//     if (count > 0) {
//         sum.divScalar((float)count);
//         sum.normalize();
//         sum.mulScalar(maxSpeed);
//         Pvector steer;
//         steer = steer.subTwoVector(sum, velocity);
//         steer.limit(maxForce);
//         return steer;
//     } else {
//         Pvector temp(0, 0);
//         return temp;
//     }
// }

// __device__ Pvector Boid::Cohesion(const Boid* boids, int numBoids)
// {
//     float neighbordist = radius;
//     Pvector sum(0, 0);
//     int count = 0;
//     for (int i = 0; i < numBoids; i++) {
//         float d = location.distance(boids[i].location);
//         if ((d > 0) && (d < neighbordist)) {
//             sum.addVector(boids[i].location);
//             count++;
//         }
//     }
//     if (count > 0) {
//         sum.divScalar(count);
//         return seek(sum);
//     } else {
//         Pvector temp(0, 0);
//         return temp;
//     }
// }

// __device__ Pvector Boid::seek(const Pvector& v)
// {
//     Pvector desired;
//     desired.subVector(v);
//     desired.normalize();
//     desired.mulScalar(maxSpeed);
//     acceleration.subTwoVector(desired, velocity);
//     acceleration.limit(maxForce);
//     return acceleration;
// }

// __device__ void Boid::update()
// {
//     acceleration.mulScalar(.4);
//     velocity.addVector(acceleration);
//     velocity.limit(maxSpeed);
//     location.addVector(velocity);
//     acceleration.mulScalar(0);
// }

// __device__ void Boid::run(const Boid* boids, int numBoids)
// {
//     flock(boids, numBoids);
//     update();
// }

// __device__ void Boid::flock(const Boid* boids, int numBoids)
// {
//     Pvector sep = Separation(boids, numBoids);
//     Pvector ali = Alignment(boids, numBoids);
//     Pvector coh = Cohesion(boids, numBoids);
//     sep.mulScalar(1.5);
//     ali.mulScalar(1.0);
//     coh.mulScalar(1.0);
//     applyForce(sep);
//     applyForce(ali);
//     applyForce(coh);
// }

// __device__ float Boid::angle(const Pvector& v)
// {
//     float angle = (float)(atan2(v.x, -v.y) * 180 / PI);
//     return angle;
// }
