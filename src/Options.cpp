#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "Boid.h"
#include "Flock.h"
#include "Options.h"
#include <vector>



__host__ __device__ bool Options::parseArgs(int argc, char *argv[])
{
    CUDA = false;
    numIterations = 1;
    radius = 50;
    for (int i = 1; i < argc; i++)
    {
        if (i < argc - 1)
        {
            if (strcmp(argv[i], "-i") == 0)
                numIterations = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "-in") == 0)
                inputFile = argv[i + 1];
            else if (strcmp(argv[i], "-o") == 0)
                outputFile = argv[i + 1];
            else if (strcmp(argv[i], "-r") == 0)
                radius = (float)atof(argv[i + 1]);
        }
        if (strcmp(argv[i], "-cu") == 0)
        {
            CUDA = true;
        }
    }
    return true;
}

__host__ __device__ Options::Options(int argc, char *argv[])
{
    parseArgs(argc, argv);
}

__host__ __device__ vector<Boid> Options::loadFromFile()
{
    std::string fileName = inputFile;
    std::ifstream f(fileName);
    // assert((bool)f && "Cannot open input file");

    std::vector<Boid> flock;

    int numBoids = 0;

    std::string line;
    while (std::getline(f, line))
    {
        Boid boid = Boid();
        std::stringstream sstream(line);
        std::string str;
        std::getline(sstream, str, ' ');
        boid.location.x = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        boid.location.y = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        boid.velocity.x = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        boid.velocity.y = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        boid.acceleration.x = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        boid.acceleration.y = (float)atof(str.c_str());
        boid.id = numBoids;
        // printf("Boid %d: %f %f %f %f\n", boid.id, boid.location.x, boid.location.y, boid.velocity.x, boid.velocity.y);
        flock.push_back(boid);
        numBoids++;
    }
    return flock;
}


