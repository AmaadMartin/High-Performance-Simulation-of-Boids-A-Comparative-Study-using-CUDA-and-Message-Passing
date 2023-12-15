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

bool parseArgs(int argc, char *argv[], StartupOptions &rs)
{
    rs.CUDA = false;
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
        if (strcmp(argv[i], "-cu") == 0)
        {
            rs.CUDA = true;
        }
    }
    return true;
}

std::vector<Boid> loadFromFile(StartupOptions options)
{
    std::string fileName = options.inputFile;
    float radius = options.radius;
    std::ifstream f(fileName);
    assert((bool)f && "Cannot open input file");

    std::vector<Boid> flock;

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
        flock.push_back(boid);
    }
    return flock;
}

void saveToFile(std::string fileName, Flock flock)
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

bool refCompare(std::string fileName, std::vector<Boid> flock)
{
    std::ifstream file(fileName);
    assert((bool)file && "Cannot open reference file");

    std::string line;
    int index = 0;
    while (std::getline(file, line))
    {

        std::stringstream sstream(line);
        std::string str;
        std::getline(sstream, str, ' ');
        float refLocationX = std::stof(str);
        std::getline(sstream, str, ' ');
        float refLocationY = std::stof(str);
        std::getline(sstream, str, ' ');
        float refVelocityX = std::stof(str);
        std::getline(sstream, str, ' ');
        float refVelocityY = std::stof(str);

        const Boid &boid = flock[index];
        if (boid.location.x != refLocationX ||
            boid.location.y != refLocationY ||
            boid.velocity.x != refVelocityX ||
            boid.velocity.y != refVelocityY)
        {
            // print out the first mismatch and line
            std::cout << "Mismatch at line " << index << std::endl;
            std::cout << "Expected: " << refLocationX << " " << refLocationY << " "
                      << refVelocityX << " " << refVelocityY << std::endl;
            std::cout << "Actual: " << boid.location.x << " " << boid.location.y << " "
                      << boid.velocity.x << " " << boid.velocity.y << std::endl;
            return false;
        }

        index++;
    }

    return true;
}
