#include <iostream>
#include "Flock.h"
#include "Boid.h"
#include "Pvector.h"
#include "Game.h"
#include "Common.h"

// Construct window using SFML
Game::Game(StartupOptions startupOptions)
{    
    // Create flock from file
    flock = loadFromFile(startupOptions);
    options = startupOptions;
}

// Run the simulation
void Game::Run()
{
    for (int i = 0; i < options.numIterations; i++) {
        SimulationStep();
    }
}

void Game::SimulationStep()
{
    // Applies the three rules to each boid in the flock and changes them accordingly.
    flock.flocking();
}
