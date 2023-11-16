# Title:
High-Performance-Simulation-of-Boids-A-Comparative-Study-using-CUDA-and-Message-Passing

# Summary:
In our project, "High-Performance Simulation of Boids: A Comparative Study using CUDA and Message Passing", we aim to simulate boid (bird-like entities) behavior in 3D, focusing on parallel computing techniques through CUDA on GPUs and message passing. Key to our approach is the incorporation of advanced load balancing and efficient memory management, ensuring that interacting boids are stored adjacently in memory to optimize performance.

# Background:
The concept of a "boid" stems from simulating the aggregate motion of flocks, herds, or schools seen in birds, land animals, or fish. This form of motion, intricate yet fluid and coordinated, has rarely been captured accurately in computer animation​​. Our project involves simulating this complex behavior using boids, an approach that treats each member of the flock as an individual but interdependent entity.

Boids operate under a distributed behavioral model. Each boid acts as an independent actor, navigating based on its perception of the environment, the laws of simulated physics, and a set of pre-defined behaviors. The interaction of these simple behaviors across multiple boids results in the complex, aggregate motion characteristic of natural flocks​​.

One significant challenge in boid simulation is the computational complexity, which increases with the number of boids. A naive implementation of the algorithm results in complexity growing as the square of the flock's population. Our project aims to address this by exploring distributed processing and constant time algorithms, where the computational load does not increase disproportionately with the number of boids​​.

The need for parallelism in boid simulation is evident from the interaction of multiple boids and the computational intensity of the process. By efficiently distributing the computation across multiple CPUs or GPU threads, we aim to achieve real-time simulation capabilities, even with large numbers of boids. This parallel approach is vital in handling the complex computations of flock dynamics while maintaining a smooth and realistic motion of the boids.

# Challenges of Parallelizing the boids algorithm:

Parallelization and Load Balancing:
Implementing the simulation in a parallel computing environment involves dividing the workload across multiple processors or GPU cores using some load balancing scheme. This becomes more challenging with dynamic simulations where the computational load can vary significantly over time.

Memory Management:
Efficient memory management is crucial, especially in ensuring that interacting boids are stored close in memory. This proximity is essential for reducing latency and improving data access patterns, which is particularly challenging given the dynamic nature of the simulation where boids constantly move and interact with different neighbors.

# Resources:
We will be using the following repo as starter code https://github.com/jyanar/Boids.git. We will also be following the “Flocks, Herds, and Schools: A Distributed Behavioral Model” paper by Craig W. Reynolds and “Simulating Species Interactions and Complex Emergence in Multiple Flocks of Boids with GPUs” by Alwyn V. Husselmann et al. We will also be using the GHC clusters and psc machines for compute.

# Goals and Deliverables:
Plan to Achieve:
Implement functional boid simulations using CUDA for GPU 
Implement functional bod simulation using OpenMPI for CPU
Conduct performance analysis and comparison between the CUDA and CPU implementations
Implement advanced load balancing and efficient memory management for optimal placement of interacting boids.
Hope to Achieve:
Extend boid model to include more complex behaviors and interactions.
Develop a real-time interactive simulation with user-modifiable parameters.
Create visualization tools for analyzing load distribution and memory usage.

Showcase a live demonstration of the boid simulation.
Present comparative performance analysis charts.
Offer an interactive element for attendees to modify simulation parameters.

# Platform Choice:
We will be using OpenMPI for our CPU implementation as the load balancing methods could benefit from message passing. For our GPU implementation we will use CUDA as the 2D coordinate system of the grids used for load balancing will lend itself well.

# Schedule:
Week 1: 

Research existing parallelization methods for boid simulations using CUDA and multicore CPU techniques.
Analyze and test the initial boid simulation algorithm from the provided starter code.
Begin the development of the CUDA-based implementation for GPU.

Week 2: 

Continue and complete the CUDA implementation for GPU.
Start developing the parallel boid simulation for message passing.
Implement basic load balancing and memory management strategies.

Week 3: 

Finalize the CPU implementation.
Conduct initial performance comparisons between the CUDA and CPU implementations.
Start integrating more complex boid behaviors as part of the extended goals.
Prepare for the initial milestone report, documenting progress and challenges.

Week 4:

Complete and submit the milestone report.
Enhance the simulation with interactive elements for real-time parameter adjustments.
Optimize existing code for better performance and efficiency.

Week 5:
Use this as a buffer period to refine and test all implementations.
Start developing visualization tools for performance and load analysis.
Begin drafting content for the poster session and project website.

Week 6:

Finalize and polish all code implementations, ensuring robustness and efficiency.
Complete the poster, website, and write-up for the project.
Final preparations for the poster presentation.
Present the poster and demonstrate the project (exact date and time to be determined).


