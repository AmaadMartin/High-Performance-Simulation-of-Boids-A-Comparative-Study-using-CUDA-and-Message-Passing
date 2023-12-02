# $ make
# $ ./boids-sim

CC = g++
CFLAGS = -c
DEBUG = -g

all: boids

boids: Boid.o Flock.o Pvector.o Game.o Common.o main.o 
	$(CC) $(DEBUG) Boid.o Flock.o Pvector.o Game.o Common.o main.o -o boids-sim 

Boid.o : Boid.cpp
	$(CC) $(CFLAGS) $(DEBUG) Boid.cpp

Pvector.o : Pvector.cpp
	$(CC) $(CFLAGS) $(DEBUG) Pvector.cpp

Flock.o : Flock.cpp
	$(CC) $(CFLAGS) $(DEBUG) Flock.cpp

Game.o : Game.cpp
	$(CC) $(CFLAGS) $(DEBUG) Game.cpp

Common.o : Common.cpp
	$(CC) $(CFLAGS) $(DEBUG) Common.cpp

clean:
	\rm *.o boids-sim

tar:
	tar cfv boids.tar Pvector.h Pvector.cpp Boid.h Boid.cpp \
		Flock.h Flock.cpp Game.cpp Game.h