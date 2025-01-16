CXX = nvcc
RELEASE_FLAGS = -O2 -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets  --restrict --use_fast_math -Xptxas -O3 -Xcompiler -Ofast,-funroll-loops,-funsafe-math-optimizations
DEBUG_FLAGS = --compiler-options -Wall -pg -fno-omit-frame-pointer

SRCS = main.cu fluid_solver.cu EventManager.cpp

# Problem Size
SIZE ?= 168

.PHONY: release debug run clean
.DEFAULT_GOAL = release

release:
	$(CXX) $(RELEASE_FLAGS) -DSIZE=$(SIZE) $(SRCS) -o fluid_sim

debug:
	$(CXX) $(DEBUG_FLAGS) -DSIZE=$(SIZE) $(SRCS) -o fluid_sim

run:
	@sbatch run.sh

clean:
	@echo Cleaning up...
	@rm fluid_sim
	@rm gmon.out perf.data 2>/dev/null || true
	@echo Done
