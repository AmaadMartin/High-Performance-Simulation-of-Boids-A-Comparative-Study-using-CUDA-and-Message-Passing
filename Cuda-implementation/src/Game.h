#include <iostream>
#include "Flock.h"
#include "Options.h"

#ifndef GAME_H
#define GAME_H

// Game handles the instantiation of a flock of boids, game input, asks the
// model to compute the next step in the stimulation, and handles all of the
// program's interaction with SFML. 

class Game {
private:
    Options options;

    __host__ __device__ void SimulationStep();
public:
    Flock flock;
    __host__ __device__ Game(Options startupOptions);
    __host__ __device__ void Run();
    
};

#endif
