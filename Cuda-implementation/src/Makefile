# # $ make
# # $ ./boids-sim

# CC = g++
# NVCC = nvcc -dc
# CFLAGS = -c
# DEBUG = -g
# NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

# all: boids

# boids: Boid.o Flock.o Pvector.o Game.o Common.o main.o 
# 	$(CC) $(DEBUG) Boid.o Flock.o Pvector.o Game.o Common.o main.o -o boids-sim 

# cudaBoids: cudaBoid.o cudaPvector.o cudaGame.o cudaCommon.o CudaFlock.o cudaMain.o 
# 	$(NVCC) $(NVCCFLAGS) $(DEBUG) cudaBoid.o cudaPvector.o cudaGame.o cudaCommon.o CudaFlock.o cudaMain.o -o cuda-boids-sim

# Boid.o : Boid.cpp
# 	$(CC) $(CFLAGS) $(DEBUG) Boid.cpp

# Pvector.o : Pvector.cpp
# 	$(CC) $(CFLAGS) $(DEBUG) Pvector.cpp
 
# Flock.o : Flock.cpp
# 	$(CC) $(CFLAGS) $(DEBUG) Flock.cpp

# Game.o : Game.cpp
# 	$(CC) $(CFLAGS) $(DEBUG) Game.cpp

# Common.o : Common.cpp
# 	$(CC) $(CFLAGS) $(DEBUG) Common.cpp

# CudaFlock.o : CudaFlock.cu CudaFlock.h
# 	$(NVCC) $(CFLAGS) $(DEBUG) -o CudaFlock.o CudaFlock.cu

# cudaBoid.o : Boid.cu Boid.h
# 	$(NVCC) $(CFLAGS) $(DEBUG) -o cudaBoid.o Boid.cu

# cudaPvector.o : Pvector.cu Pvector.h
# 	$(NVCC) $(CFLAGS) $(DEBUG) -o cudaPvector.o Pvector.cu

# cudaGame.o : Game.cpp Game.h
# 	$(NVCC) $(CFLAGS) $(DEBUG) -o cudaGame.o Game.cpp

# cudaCommon.o : Common.cpp Common.h
# 	$(NVCC) $(CFLAGS) $(DEBUG) -o cudaCommon.o Common.cpp

# cudaMain.o : main.cpp
# 	$(NVCC) $(CFLAGS) $(DEBUG) -o cudaMain.o main.cpp

# clean:
# 	\rm *.o boids-sim cudaBoid
#     // thrust::sort_by_key(thrust::device, unsortedTuples, unsortedTuples + numBoids, hash);
# tar:
# 	tar cfv boids.tar Pvector.h Pvector.cpp Boid.h Boid.cpp \
# 		Flock.h Flock.cpp Game.cpp Game.h

objects = Boid.o Flock.o Pvector.o Game.o Options.o main.o 

all: $(objects)
		nvcc $(objects) -o app

%.o: %.cpp
		nvcc -x cu -I. -dc $< -o $@ --extended-lambda --expt-relaxed-constexpr

clean:
		rm -f *.o app