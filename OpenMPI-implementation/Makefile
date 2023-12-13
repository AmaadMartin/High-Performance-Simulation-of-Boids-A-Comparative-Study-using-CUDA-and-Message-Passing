CFLAGS := -std=c++14 -fvisibility=hidden -lpthread -Wall -Wextra

# Assuming MPI libraries are required
# MPILIBS := -lmpi

# Detect if we are in a debug configuration
ifeq (,$(CONFIGURATION))
	CONFIGURATION := release
endif

ifeq (debug,$(CONFIGURATION))
CFLAGS += -g
else
CFLAGS += -O2
endif

# Include directory for header files
INCLUDES := -I OpenMPI-implementation/

# Source and Header files
SOURCES := main.cpp
HEADERS := *.h

# Compiler
CXX := mpic++

# Output executable
EXECUTABLE := main

# Phony targets
.PHONY: all clean

# Default target
all: $(EXECUTABLE)

# Compile the main executable
$(EXECUTABLE): $(SOURCES) $(HEADERS)
	$(CXX) $(CFLAGS) $(INCLUDES) $(SOURCES) -o $(EXECUTABLE) $(MPILIBS)

# Clean the build
clean:
	rm -f $(EXECUTABLE)
