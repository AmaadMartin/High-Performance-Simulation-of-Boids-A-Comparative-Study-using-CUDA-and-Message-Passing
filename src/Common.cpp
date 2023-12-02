#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "Boid.h"
#include "Flock.h"
#include "Common.h"

inline bool parseArgs(int argc, char *argv[], StartupOptions &rs)
{
    rs.loadBalance = false;
    rs.numIterations = 1;
    rs.radius = 50;
    for (int i = 1; i < argc; i++)
    {
        if (i < argc - 1)
        {
            if (strcmp(argv[i], "-i") == 0)
                rs.numIterations = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "-in") == 0)
                rs.inputFile = argv[i + 1];
            else if (strcmp(argv[i], "-o") == 0)
                rs.outputFile = argv[i + 1];
            else if (strcmp(argv[i], "-r") == 0)
                rs.radius = (float)atof(argv[i + 1]);
        }
        if (strcmp(argv[i], "-lb") == 0)
        {
            rs.loadBalance = true;
        }
    }
    return true;
}

inline Flock loadFromFile(StartupOptions options)
{
    std::string fileName = options.inputFile;
    float radius = options.radius;
    std::ifstream f(fileName);
    assert((bool)f && "Cannot open input file");
    Flock flock = Flock();

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
        
        boid.radius = radius;
        flock.addBoid(boid);
    }
    return flock;
}

/**
 * Saves the boid information to a file.
 *
 * @param fileName The name of the output file.
 * @param flock The flock containing the boid information.
 */
inline void saveToFile(std::string fileName, Flock flock)
{
    std::ofstream f(fileName);
    // create file if it doesn't exist
    assert((bool)f && "Cannot open output file");

    Boid boid = Boid();

    f << std::setprecision(9);
    for (int i = 0; i < flock.getFlock().size(); i++)
    {
        boid = flock.getFlock()[i];
        f << boid.location.x << " " << boid.location.y << " "
          << boid.velocity.x << " " << boid.velocity.y << std::endl;
    }
    assert((bool)f && "Failed to write to output file");
}
