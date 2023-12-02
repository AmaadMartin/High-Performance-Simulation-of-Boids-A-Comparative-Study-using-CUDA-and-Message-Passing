#include <iostream>
#include "Flock.h"
#include "Boid.h"
#include "Pvector.h"
#include "Common.cpp"

#ifndef GAME_H
#define GAME_H

// Game handles the instantiation of a flock of boids, game input, asks the
// model to compute the next step in the stimulation, and handles all of the
// program's interaction with SFML. 

class Game {
private:
    StartupOptions options;

    void SimulationStep();
    void HandleInput();

    inline bool parseArgs(int argc, char *argv[], StartupOptions &rs);

public:
    Flock flock;
    Game(StartupOptions startupOptions);
    void Run();
    
};

#endif
