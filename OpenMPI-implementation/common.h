#ifndef COMMON_H_
#define COMMON_H_

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct StartupOptions {
  int numIterations = 2;
  // int numBoids = 5;
  float spaceSize = 10.0f;
  bool loadBalance = false;
  float radius = 50.0f;
  std::string outputFile;
  std::string inputFile;
};

struct StepParameters {
  float cullRadius = 50.0f;
  float maxSpeed = 10.0f;
  float cohesionFactor = 0.01f;
  float separationFactor = 0.01f;
  float alignmentFactor = 0.01f;
  float deltaTime = 0.2f;
};

inline StepParameters getBenchmarkStepParams(float spaceSize) {
  StepParameters result;
  return result;
}

inline StartupOptions parseOptions(int argc, char *argv[]) {
  StartupOptions rs;
    for (int i = 1; i < argc; i++) {
        if (i < argc - 1) {
            if (strcmp(argv[i], "-i") == 0)
                rs.numIterations = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "-in") == 0)
                rs.inputFile = argv[i + 1];
            // else if (strcmp(argv[i], "-o") == 0)
            //     rs.outputFile = argv[i + 1];
            else if (strcmp(argv[i], "-r") == 0)
                rs.radius = (float)atof(argv[i + 1]);
        }
        if (strcmp(argv[i], "-lb") == 0){
            rs.loadBalance = true;
        }
    }
    return rs;
}

struct Vec2 {
  float x, y;
  Vec2(float vx = 0.0f, float vy = 0.0f) : x(vx), y(vy) {}
  static float dot(Vec2 v0, Vec2 v1) { return v0.x * v1.x + v0.y * v1.y; }
  float &operator[](int i) { return ((float *)this)[i]; }
  Vec2 operator*(float s) const { return Vec2(*this) *= s; }
  Vec2 operator*(Vec2 vin) const { return Vec2(*this) *= vin; }
  Vec2 operator+(Vec2 vin) const { return Vec2(*this) += vin; }
  Vec2 operator-(Vec2 vin) const { return Vec2(*this) -= vin; }
  Vec2 operator/(float s) const {return Vec2(*this) /= s; }
  Vec2 operator-() const { return Vec2(-x, -y); }
  Vec2 &operator/=(float s) {
    x /= s;
    y /= s;
    return *this;
  }
  Vec2 &operator+=(Vec2 vin) {
    x += vin.x;
    y += vin.y;
    return *this;
  }
  Vec2 &operator-=(Vec2 vin) {
    x -= vin.x;
    y -= vin.y;
    return *this;
  }
  Vec2 &operator=(float v) {
    x = y = v;
    return *this;
  }
  Vec2 &operator*=(float s) {
    x *= s;
    y *= s;
    return *this;
  }
  Vec2 &operator*=(Vec2 vin) {
    x *= vin.x;
    y *= vin.y;
    return *this;
  }

  float length2() const { return x * x + y * y; }
  float length() const { return sqrt(length2()); }
};

struct Boid {
  int id;
  Vec2 position;
  Vec2 velocity;
};

inline void loadFromFile(std::string fileName, std::vector<Boid> &boids)
{
    // float radius = options.radius;
    // std::ifstream f(fileName);
    // assert((bool)f && "Cannot open input file");

    std::cout << "Attempting to open file: " << fileName << std::endl;

    std::ifstream f(fileName);
    if (!f) {
        int err = errno;  // Capture the errno value before any other system calls
        std::cerr << "Error opening file '" << fileName << "': " << std::strerror(err) << std::endl;
        exit(1);  // Exit or handle the error as appropriate for your application
    }

    
    std::string line;
    while (std::getline(f, line))
    {
        Boid boid = Boid();
        std::stringstream sstream(line);
        std::string str;
        std::getline(sstream, str, ' ');
        boid.position.x = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        boid.position.y = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        boid.velocity.x = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        boid.velocity.y = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        // boid.acceleration.x = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        // boid.acceleration.y = (float)atof(str.c_str());
        
        // boid.radius = radius;
        // flock.addBoid(boid);
        boids.push_back(boid);
    }
}
#endif