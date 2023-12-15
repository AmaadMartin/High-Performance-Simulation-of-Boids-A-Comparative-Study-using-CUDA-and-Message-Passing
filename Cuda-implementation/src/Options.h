#ifndef OPTIONS_H
#define OPTIONS_H

#include <string>
#include "Boid.h"
#include "Flock.h"
#include <vector>

class Options {
public:
    int numIterations;
    bool CUDA;
    float radius;
    int numThreads;
    std::string outputFile;
    std::string inputFile;

    __host__ __device__ Options() {}
    __host__ __device__ Options(int argc, char *argv[]);
    __host__ __device__ vector<Boid> loadFromFile();

private:
    bool parseArgs(int argc, char *argv[]);
};

#endif // OPTIONS_H