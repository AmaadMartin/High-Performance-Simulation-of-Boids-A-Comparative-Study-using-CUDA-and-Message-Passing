#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "Game.h"
// #include "Common.h"
#include "Options.h"
#include "Flock.h"
#include "CudaFlock.h"
#include "Boid.h"
#include "timing.h"

#define ERROR_THRESHOLD 1

void saveToFile(std::string fileName, Flock flock)
{
    std::ofstream f(fileName);
    // create file if it doesn't exist

    Boid boid = Boid();

    f << std::setprecision(9);
    for (int i = 0; i < flock.getFlock().size(); i++)
    {
        boid = flock.getFlock()[i];
        f << boid.location.x << " " << boid.location.y << " "
          << boid.velocity.x << " " << boid.velocity.y << std::endl;
        // printf("Boid %d: %f %f %f %f\n", i, boid.location.x, boid.location.y, boid.velocity.x, boid.velocity.y);
    }
    // printf("Saved to file %s\n", fileName.c_str());
}

void refCompare(std::string ref, Flock flock)
{
    std::ifstream f(ref);
    // assert((bool)f && "Cannot open input file");

    Boid boid = Boid();

    std::string line;
    int i = 0;
    while (std::getline(f, line))
    {
        boid = flock.getFlock()[i];
        std::stringstream sstream(line);
        std::string str;
        std::getline(sstream, str, ' ');
        float x = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        float y = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        float vx = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        float vy = (float)atof(str.c_str());
        // printf("Boid %d: %f %f %f %f\n", i, boid.location.x, boid.location.y, boid.velocity.x, boid.velocity.y);
        // printf("Ref %d: %f %f %f %f\n", i, x, y, vx, vy);
        if (std::abs(boid.location.x - x) > ERROR_THRESHOLD ||
            std::abs(boid.location.y - y) > ERROR_THRESHOLD ||
            std::abs(boid.velocity.x - vx) > ERROR_THRESHOLD ||
            std::abs(boid.velocity.y - vy) > ERROR_THRESHOLD)
        {
            printf("Boid %d: %f %f %f %f\n", boid.id, boid.location.x, boid.location.y, boid.velocity.x, boid.velocity.y);
            printf("Ref %d: %f %f %f %f\n", i, x, y, vx, vy);
            printf("Difference: %f %f %f %f\n", std::abs(boid.location.x - x), std::abs(boid.location.y - y), std::abs(boid.velocity.x - vx), std::abs(boid.velocity.y - vy));
            printf("Test failed\n");
            exit(1);
        }
        i++;
    }
    printf("Test passed\n");
}

int main(int argc, char *argv[])
{
    Options options = Options(argc, argv);
    printf("Running with %d iterations\n", options.numIterations);

    Game game = Game(options);
    Timer timer;
    game.Run();
    double time = timer.elapsed();
    printf("Time: %f\n", time);
    
    // if (options.CUDA){
    //     refCompare("benchmark-files/random-100-ref.txt", game.flock);
    // } else {
    //     saveToFile(options.outputFile, game.flock);
    // }
    return 0;
}
