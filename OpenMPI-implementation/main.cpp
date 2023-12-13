#include "common.h"
#include "mpi.h"
#include "quad-tree.h"
#include "timing.h"

#define MASTER	0
#define NUM_SIMULATES 5

void simulateStep(const QuadTree &quadTree,
                  const std::vector<Boid> &boids,
                  std::vector<Boid> &newBoids, StepParameters params) {
  
  for (size_t i = 0; i < boids.size(); i++) {
    auto boid = boids[i];
    
 
    std::vector<Boid> neighborBoids;

    quadTree.getParticles(neighborBoids, boid.position, params.cullRadius);


    int numNeighbors = neighborBoids.size();
    Vec2 centerOfMass(0, 0);
    Vec2 separation(0, 0);
    Vec2 averageVelocity(0, 0);

    // accumulate attractive forces to apply to particle i
    for (size_t j = 0; j < numNeighbors; j++) {
      if (&neighborBoids[j] == &boid) 
        continue;
      //calculate seperation, alignment, and cohesion
        centerOfMass+=neighborBoids[j].position;     
        separation -= (neighborBoids[j].position - boid.position);
        averageVelocity += neighborBoids[j].velocity;
    }
    

    if (numNeighbors > 0) {
        centerOfMass /= numNeighbors;
        centerOfMass = (centerOfMass - boid.position)*params.cohesionFactor; 
        averageVelocity /= numNeighbors;
        averageVelocity = (averageVelocity - boid.velocity) * params.alignmentFactor;
    }

    separation *= params.separationFactor;

    // apply velocity to particle i. truncate if necessary
    boid.velocity += centerOfMass + separation + averageVelocity;

    if (boid.velocity.length() > params.maxSpeed) {
      boid.velocity = (boid.velocity / boid.velocity.length()) * params.maxSpeed;
    }

    boid.position += boid.velocity*params.deltaTime;
    newBoids[i] = boid;
  }
}

bool inGrid(const Boid &boid, int taskid, int numtasks, float maxX, float minX, float maxY, float minY) {
  float x = boid.position.x - minX;
  float y = boid.position.y - minY;
  float gridSizeX = (maxX - minX) / sqrt(numtasks);
  float gridSizeY = (maxY - minY) / sqrt(numtasks);
  float gridX = x / gridSizeX;
  float gridY = y / gridSizeY;
  int gridXInt = (int)gridX;
  int gridYInt = (int)gridY;
  int grid = gridXInt + gridYInt * sqrt(numtasks);
  return grid == taskid;
}

bool gridsIntersect(int grid1, int grid2, float cullRadius, float bounds[]) {
  float x1 = bounds[grid1*4] - cullRadius;
  float y1 = bounds[grid1*4 + 1] - cullRadius;
  float x2 = bounds[grid1*4 + 2] + cullRadius;
  float y2 = bounds[grid1*4 + 3] + cullRadius;

  float x3 = bounds[grid2*4];
  float y3 = bounds[grid2*4 + 1];
  float x4 = bounds[grid2*4 + 2];
  float y4 = bounds[grid2*4 + 3];

  if(x4 < x1) {
    return false;
  } else if(x3 > x2) {
    return false;
  } else if(y4 < y1) {
    return false;
  } else if(y3 > y2) {
    return false;
  } else {
    return true;
  }
}

bool allValid(bool validGrids[], int numtasks) {
  for(int i = 0; i < numtasks; i++) {
    printf("validGrids[%d]: %d\n", i, validGrids[i]);
    if(!validGrids[i]) {
      return false;
    }
  }
  return true;
}

bool isValid(int taskid, std::vector<Boid> boids, float bounds[]) {
  float x1 = bounds[taskid*4];
  float y1 = bounds[taskid*4 + 1];
  float x2 = bounds[taskid*4 + 2];
  float y2 = bounds[taskid*4 + 3];

  for(int i = 0; i < boids.size(); i++) {
    if(boids[i].position.x < x1 || boids[i].position.x > x2 || boids[i].position.y < y1 || boids[i].position.y > y2) {
      return false;
    }
  }
  return true;
}

std::vector<Boid> getPartnerBoids(int taskid, int partner, std::vector<Boid> boids, float bounds[], float cullRadius) {
  float x1 = bounds[partner*4] - cullRadius;
  float y1 = bounds[partner*4 + 1] - cullRadius;
  float x2 = bounds[partner*4 + 2] + cullRadius;
  float y2 = bounds[partner*4 + 3] + cullRadius;

  std::vector<Boid> partnerBoids;
  for(int i = 0; i < boids.size(); i++) {
    if(boids[i].position.x >= x1 && boids[i].position.x <= x2 && boids[i].position.y >= y1 && boids[i].position.y <= y2) {
      partnerBoids.push_back(boids[i]);
    }
  }
  return partnerBoids;
}

bool completelyInRadius(int partner, int taskid, float bounds[], float cullRadius) {
  float x1 = bounds[taskid*4] - cullRadius;
  float y1 = bounds[taskid*4 + 1] - cullRadius;
  float x2 = bounds[taskid*4 + 2] + cullRadius;
  float y2 = bounds[taskid*4 + 3] + cullRadius;

  float x3 = bounds[partner*4];
  float y3 = bounds[partner*4 + 1];
  float x4 = bounds[partner*4 + 2];
  float y4 = bounds[partner*4 + 3];

  if (x3 >= x1 && x4 <= x2 && y3 >= y1 && y4 <= y2) {
    return true;
  } else {
    return false;
  }
}

int main(int argc, char *argv[]) {
  int numtasks, taskid, rc, dest, offset, i, j, tag1,
      tag2, source, chunksize, leftover, numBoids, 
      size, displacement, totalSize;


  float minX, minY, maxX, maxY, minPoint, maxPoint; 
  double mysum, sum;  

  tag1 = 1;
  tag2 = 2;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  // Get total number of processes specificed at start of run
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  int partnerSizes[numtasks];

  MPI_Status stats[numtasks];
  MPI_Request reqs[numtasks];

  int sendCounts[numtasks], newBoidsSendCounts[numtasks], newBoidsSizes[numtasks], newBoidDispls[numtasks], displs[numtasks], taskIdDispls[numtasks];

  for (int i = 0; i < numtasks; i++) {
    sendCounts[i] = 4;
    newBoidsSendCounts[i] = 1;
    displs[i] = i * 4;
    taskIdDispls[i] = i;
  }

  float bounds[numtasks * 4];
  float taskBounds[4];

  StartupOptions options = parseOptions(argc, argv);

  MPI_Datatype MPI_BOID;

  int block_lengths[3] = {1, 2, 2};  // Number of elements in each block

  //Might need to fix this this is weird
  MPI_Aint displacements[3] = { offsetof(Boid, id),
                                offsetof(Boid, position),
                                offsetof(Boid, velocity)};

  MPI_Datatype types[3] = {MPI_INT, MPI_FLOAT, MPI_FLOAT};  // Data types of each block

  MPI_Type_create_struct(3, block_lengths, displacements, types, &MPI_BOID);
  MPI_Type_commit(&MPI_BOID);

  std::vector<Boid> boids, newBoids, nextBoids, gridBoids, treeBoids, partnerBoids[numtasks], finalBoids;
  std::vector<int> partners;

  loadFromFile(options.inputFile, boids);

  StepParameters stepParams = getBenchmarkStepParams(options.spaceSize);

  Timer totalSimulationTimer; 
  QuadTree tree;  
  int iteration = 0;
  
  while (iteration < options.numIterations) {
    // printf("taskid: %d, iteration: %d\n", taskid, iteration);
    // get minX, minY, maxX, maxY
    for (int i = 0; i < boids.size(); i++){
      if (i == 0) {
        minX = boids[i].position.x;
        minY = boids[i].position.y;
        maxX = boids[i].position.x;
        maxY = boids[i].position.y;
      } else {
        minX = std::min(minX, boids[i].position.x);
        minY = std::min(minY, boids[i].position.y);
        maxX = std::max(maxX, boids[i].position.x);
        maxY = std::max(maxY, boids[i].position.y);
      }
    }

    maxX += 1;
    maxY += 1;

    size = 0;
    gridBoids.clear();
    for (int i = 0; i < boids.size(); i++) {
      if (inGrid(boids[i], taskid, numtasks, maxX, minX, maxY, minY)) {
        size++;
        gridBoids.push_back(boids[i]);
      }
    }
    
    newBoids.resize(size);

    // gather sizes into newParticleSizes
    MPI_Allgatherv(&size, 1, MPI_INT, newBoidsSizes, newBoidsSendCounts, taskIdDispls, MPI_INT, MPI_COMM_WORLD);
    
    // get tasks displacement
    displacement = 0;
    totalSize = 0;
    for(int i = 0; i < numtasks; i++) {
      if(i < taskid){
        displacement += newBoidsSizes[i];
      }
      totalSize += newBoidsSizes[i];
    }
    // gather displacements into newParticleDispls
    MPI_Allgatherv(&displacement, 1, MPI_INT, newBoidDispls, newBoidsSendCounts, taskIdDispls, MPI_INT, MPI_COMM_WORLD);


    for (int k = 0; k < NUM_SIMULATES && iteration < options.numIterations; k++) {
      // get taskBounds
      for(int i = 0; i < size;i++){
        if (i == 0) {
          taskBounds[0] = gridBoids[i].position.x;
          taskBounds[1] = gridBoids[i].position.y;
          taskBounds[2] = gridBoids[i].position.x;
          taskBounds[3] = gridBoids[i].position.y;
        } else {
          taskBounds[0] = std::min(taskBounds[0], gridBoids[i].position.x);
          taskBounds[1] = std::min(taskBounds[1], gridBoids[i].position.y);
          taskBounds[2] = std::max(taskBounds[2], gridBoids[i].position.x);
          taskBounds[3] = std::max(taskBounds[3], gridBoids[i].position.y);
        }
      }

      // gather taskbounds into bounds
      MPI_Allgatherv(taskBounds, 4, MPI_FLOAT, bounds, sendCounts, displs, MPI_FLOAT, MPI_COMM_WORLD);

      // copy gridParticles to treeParticles
      treeBoids = gridBoids;

      // check for partners
      partners.clear();

      for (int i = 0; i < numtasks; i++) {
        if (i != taskid && (gridsIntersect(taskid, i, stepParams.cullRadius, bounds))) {
          partners.push_back(i);
        }
      }

      // send particles to partners
      for(int i = 0; i < partners.size(); i++) {
        if(completelyInRadius(partners[i], taskid, bounds, stepParams.cullRadius)){
          partnerBoids[partners[i]] = gridBoids;
        } else {
          partnerBoids[partners[i]] = getPartnerBoids(taskid, partners[i], gridBoids, bounds, stepParams.cullRadius);
        }
        partnerSizes[partners[i]] = partnerBoids[partners[i]].size();
        MPI_Isend(&partnerSizes[partners[i]], 1, MPI_INT, partners[i], tag1, MPI_COMM_WORLD, &reqs[i]);
        MPI_Isend(partnerBoids[partners[i]].data(), partnerSizes[partners[i]], MPI_BOID, partners[i], tag2, MPI_COMM_WORLD, &reqs[i]);
      }
      
      // receive particles from partners
      for(int partner : partners) {
        int partnerSize;
        MPI_Recv(&partnerSize, 1, MPI_INT, partner, tag1, MPI_COMM_WORLD, &stats[0]);
        treeBoids.resize(treeBoids.size() + partnerSize);
        MPI_Recv(treeBoids.data() + treeBoids.size() - partnerSize, partnerSize, MPI_BOID, partner, tag2, MPI_COMM_WORLD, &stats[0]);
      }

      // build tree
      QuadTree::buildQuadTree(treeBoids, tree);

      // simulate step 
      simulateStep(tree, gridBoids, newBoids, stepParams);
      gridBoids =  newBoids;
      iteration++;
    }

    // gather newParticles into particles
    MPI_Allgatherv(newBoids.data(), size, MPI_BOID, boids.data(), newBoidsSizes, newBoidDispls, MPI_BOID, MPI_COMM_WORLD);
  }
  double totalSimulationTime = totalSimulationTimer.elapsed();

  finalBoids.resize(boids.size());
  for(int i = 0; i < boids.size(); i++) {
    finalBoids[boids[i].id] = boids[i];
  }

  if (taskid == MASTER) {
    printf("total simulation time: %.6fs\n", totalSimulationTime);
    // saveToFile(options.outputFile, finalBoids);
  }

  MPI_Type_free(&MPI_BOID);
  
  MPI_Finalize();
}