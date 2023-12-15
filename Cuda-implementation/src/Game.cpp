#include <iostream>
#include "Flock.h"
#include "Boid.h"
#include "Pvector.h"
#include "Game.h"
#include "Options.h"
// #include "Common.h"

// Construct window using SFML
__host__ __device__ Game::Game(Options startupOptions)
{    
    // Create flock from file
    vector<Boid> boids = startupOptions.loadFromFile();
    options = startupOptions;
    flock = Flock(boids, options.numThreads);
}

// Run the simulation
__host__ __device__ void Game::Run()
{
    for (int i = 0; i < options.numIterations; i++) {
        SimulationStep();
    }
}

__host__ __device__ void Game::SimulationStep()
{
    // Applies the three rules to each boid in the flock and changes them accordingly.
    if (options.CUDA) {
        flock.cudaFlocking();
    } else {
        flock.flocking();
    }
}