#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

//#define SIZE 168
#ifndef SIZE
#define SIZE 168
#endif
#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;
static float diff = 0.0001f;
static float visc = 0.0001f;

// Device pointers
static float *d_u, *d_v, *d_w, *d_u_prev, *d_v_prev, *d_w_prev;
static float *d_dens, *d_dens_prev;

// Host pointers
static float *h_u, *h_v, *h_w, *h_u_prev, *h_v_prev, *h_w_prev;
static float *h_dens, *h_dens_prev;

int allocate_data() {
    int size = (M + 2) * (N + 2) * (O + 2);
    size_t bytes = size * sizeof(float);

    // Allocate host memory
    h_u = new float[size];
    h_v = new float[size];
    h_w = new float[size];
    h_u_prev = new float[size];
    h_v_prev = new float[size];
    h_w_prev = new float[size];
    h_dens = new float[size];
    h_dens_prev = new float[size];

    // Allocate device memory
    cudaMalloc((void**)&d_u, bytes);
    cudaMalloc((void**)&d_v, bytes);
    cudaMalloc((void**)&d_w, bytes);
    cudaMalloc((void**)&d_u_prev, bytes);
    cudaMalloc((void**)&d_v_prev, bytes);
    cudaMalloc((void**)&d_w_prev, bytes);
    cudaMalloc((void**)&d_dens, bytes);
    cudaMalloc((void**)&d_dens_prev, bytes);

    if (!h_u || !h_v || !h_w || !h_u_prev || !h_v_prev || !h_w_prev || 
        !h_dens || !h_dens_prev || !d_u || !d_v || !d_w || !d_u_prev || 
        !d_v_prev || !d_w_prev || !d_dens || !d_dens_prev) {
        std::cerr << "Cannot allocate memory" << std::endl;
        return 0;
    }
    return 1;
}

void clear_data() {
    int size = (M + 2) * (N + 2) * (O + 2);
    for (int i = 0; i < size; i++) {
        h_u[i] = h_v[i] = h_w[i] = h_u_prev[i] = h_v_prev[i] = 
        h_w_prev[i] = h_dens[i] = h_dens_prev[i] = 0.0f;
    }
    
    size_t bytes = size * sizeof(float);
    cudaMemcpy(d_u, h_u, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_prev, h_u_prev, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_prev, h_v_prev, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_prev, h_w_prev, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dens, h_dens, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dens_prev, h_dens_prev, bytes, cudaMemcpyHostToDevice);
}

void free_data() {
    // Free host memory
    delete[] h_u;
    delete[] h_v;
    delete[] h_w;
    delete[] h_u_prev;
    delete[] h_v_prev;
    delete[] h_w_prev;
    delete[] h_dens;
    delete[] h_dens_prev;

    // Free device memory
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_u_prev);
    cudaFree(d_v_prev);
    cudaFree(d_w_prev);
    cudaFree(d_dens);
    cudaFree(d_dens_prev);
}

void apply_events(const std::vector<Event> &events) {
    for (const auto &event : events) {
        if (event.type == ADD_SOURCE) {
            int i = M / 2, j = N / 2, k = O / 2;
            h_dens[IX(i, j, k)] = event.density;
            cudaMemcpy(&d_dens[IX(i, j, k)], &h_dens[IX(i, j, k)], 
                      sizeof(float), cudaMemcpyHostToDevice);
        } else if (event.type == APPLY_FORCE) {
            int i = M / 2, j = N / 2, k = O / 2;
            h_u[IX(i, j, k)] = event.force.x;
            h_v[IX(i, j, k)] = event.force.y;
            h_w[IX(i, j, k)] = event.force.z;
            cudaMemcpy(&d_u[IX(i, j, k)], &h_u[IX(i, j, k)], 
                      sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(&d_v[IX(i, j, k)], &h_v[IX(i, j, k)], 
                      sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(&d_w[IX(i, j, k)], &h_w[IX(i, j, k)], 
                      sizeof(float), cudaMemcpyHostToDevice);
        }
    }
}

float sum_density() {
    int size = (M + 2) * (N + 2) * (O + 2);
    cudaMemcpy(h_dens, d_dens, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_density = 0.0f;
    for (int i = 0; i < size; i++) {
        total_density += h_dens[i];
    }
    return total_density;
}

void simulate(EventManager &eventManager, int timesteps) {
    for (int t = 0; t < timesteps; t++) {
        std::vector<Event> events = eventManager.get_events_at_timestamp(t);
        apply_events(events);
        
        vel_step(M, N, O, d_u, d_v, d_w, d_u_prev, d_v_prev, d_w_prev, visc, dt);
        dens_step(M, N, O, d_dens, d_dens_prev, d_u, d_v, d_w, diff, dt);
    }
}

int main() {
    EventManager eventManager;
    eventManager.read_events("events.txt");
    int timesteps = eventManager.get_total_timesteps();

    if (!allocate_data())
        return -1;
    clear_data();

    simulate(eventManager, timesteps);

    float total_density = sum_density();
    std::cout << "Total density after " << timesteps 
              << " timesteps: " << total_density << std::endl;

    free_data();
    return 0;
}
