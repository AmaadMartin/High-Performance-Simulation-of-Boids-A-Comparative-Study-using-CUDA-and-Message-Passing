#ifndef COMMON_H
#define COMMON_H

#include <string>
#include "Boid.h"
#include "Flock.h"

struct StartupOptions
{
    int numIterations;
    bool CUDA;
    float radius;
    std::string outputFile;
    std::string inputFile;
};


bool parseArgs(int argc, char *argv[], StartupOptions &rs);
vector<Boid> loadFromFile(StartupOptions options);
void saveToFile(std::string fileName, Flock flock);
bool refCompare(std::string fileName, std::vector<Boid> flock)

#endif // COMMON_H
