CXX = g++

SEQ_FLAGS = -Wall -O3 -funroll-loops -msse4 -ffast-math -march=native -flto
THREADED_FLAGS = $(SEQ_FLAGS) -fopenmp
DEBUG_FLAGS = -Wall -pg -fno-omit-frame-pointer

SRCS = main.cpp fluid_solver.cpp EventManager.cpp

.PHONY: all phase2 seq debug clean runseq runpar

all: phase2

# Release build
phase2:
    $(CXX) $(THREADED_FLAGS) $(SRCS) -o fluid_sim

seq:
    $(CXX) $(SEQ_FLAGS) $(SRCS) -o fluid_sim_seq

debug:
    $(CXX) $(DEBUG_FLAGS) $(SRCS) -o fluid_sim

runseq:
    ./fluid_sim_seq

runpar:
    OMP_NUM_THREADS=20 ./fluid_sim

clean:
    @echo Cleaning up...
    @rm fluid_sim
    @rm fluid_sim_seq gmon.out perf.data 2>/dev/null || true
    @echo Done.