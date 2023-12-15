#ifndef FILES_H
#define FILES_H

#include <string>
#include "Boid.h"
#include "Flock.h"

class Files {
public:
    Files();
    ~Files();
    vector<Boid> loadFromFile(std::string fileName);
    void saveToFile(std::string fileName, Flock flock);
    bool refCompare(std::string fileName, std::vector<Boid> flock);
};