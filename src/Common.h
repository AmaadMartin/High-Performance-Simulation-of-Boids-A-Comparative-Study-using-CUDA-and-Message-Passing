#ifndef COMMON_H
#define COMMON_H

#include <string>
#include "Boid.h"
#include "Flock.h"

struct StartupOptions
{
    int numIterations;
    bool loadBalance;
    float radius;
    std::string outputFile;
    std::string inputFile;
};

bool parseArgs(int argc, char *argv[], StartupOptions &rs);
Flock loadFromFile(StartupOptions options);
void saveToFile(std::string fileName, Flock flock);

#endif // COMMON_H
