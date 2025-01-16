#include "fluid_solver.h"
#include <cmath>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// @@ add_source

__global__ void add_source_kernel(int M, int N, int O, float *x, float *s, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i <= M + 1 && j <= N + 1 && k <= O + 1) {
        int index = IX(i, j, k);
        x[index] += dt * s[index];
    }
}


// Função host para chamar o kernel
void add_source(int M, int N, int O, float *d_x, float *d_s, float dt) {
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + 2 + threadsPerBlock.z - 1) / threadsPerBlock.z);

    add_source_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, d_x, d_s, dt);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// @@ set_bnd

__global__ void set_bnd_kernel(int M, int N, int O, int b, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Front and back faces
    if (i <= M && j <= N) {
        if (i >= 1 && j >= 1) {
            x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
            x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
        }
    }
    
    // Left and right faces
    if (i <= N && j <= O) {
        if (i >= 1 && j >= 1) {
            x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
            x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
        }
    }
    
    // Top and bottom faces
    if (i <= M && j <= O) {
        if (i >= 1 && j >= 1) {
            x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
            x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
        }
    }
    
    // Handle corners with main thread
    if (i == 0 && j == 0) {
        x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
        x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
        x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
        x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
    }
}

void set_bnd(int M, int N, int O, int b, float *d_x) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (M + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_x);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// @@ lin_solve

// Templates para redução
template <unsigned int blockSize>
__device__ void warpReduceMax(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2) sdata[tid] = fmaxf(sdata[tid], sdata[tid + 1]);
}

// Kernel para pontos com redução otimizada
template <unsigned int blockSize>
__global__ void lin_solve_kernel(int M, int N, int O, float* x, float* x0, float a, float c, float* block_max, int offset) {
    extern __shared__ float sdata[];
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    sdata[tid] = 0.0f;
    
    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
        if ((i + j + k + offset) % 2 == 0) {
            float old_x = x[IX(i, j, k)];
            x[IX(i, j, k)] = (x0[IX(i, j, k)] + 
                             a * (x[IX(i-1, j, k)] + x[IX(i+1, j, k)] +
                                 x[IX(i, j-1, k)] + x[IX(i, j+1, k)] +
                                 x[IX(i, j, k-1)] + x[IX(i, j, k+1)])) / c;
            
            sdata[tid] = fabs(x[IX(i, j, k)] - old_x);
        }
    }
    
    __syncthreads();
    
    // Redução em memória compartilhada
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]);
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]);
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + 64]);
        }
        __syncthreads();
    }
    
    if (tid < 32) warpReduceMax<blockSize>(sdata, tid);
    
    if (tid == 0) {
        block_max[blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y] = sdata[0];
    }
}

// Função para encontrar o máximo no host
float find_max(float* data, int size) {
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        max_val = fmaxf(max_val, data[i]);
    }
    return max_val;
}

// Função principal atualizada
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    const unsigned int blockSize = 512;
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((M + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + 2 + threadsPerBlock.z - 1) / threadsPerBlock.z);
    
    int num_blocks = numBlocks.x * numBlocks.y * numBlocks.z;
    float *d_block_max;
    cudaMalloc((void**)&d_block_max, num_blocks * sizeof(float));
    float *h_block_max = new float[num_blocks];
    
    const float tol = 1e-7f;
    int iter = 0;
    float max_change;
    
    do {
        lin_solve_kernel<blockSize><<<numBlocks, threadsPerBlock, blockSize * sizeof(float)>>>(M, N, O, x, x0, a, c, d_block_max, 0);
        
        lin_solve_kernel<blockSize><<<numBlocks, threadsPerBlock, blockSize * sizeof(float)>>>(M, N, O, x, x0, a, c, d_block_max, 1);
        
        set_bnd(M, N, O, b, x);
        
        cudaMemcpy(h_block_max, d_block_max, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
        max_change = find_max(h_block_max, num_blocks);
        
        iter++;
    } while (max_change > tol && iter < 20);
    
    cudaFree(d_block_max);
    delete[] h_block_max;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// @@ diffuse - original

void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// @@ advect

__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt, float dtX, float dtY, float dtZ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O) {
        float x = i - dtX * u[IX(i, j, k)];
        float y = j - dtY * v[IX(i, j, k)];
        float z = k - dtZ * w[IX(i, j, k)];

        // Clamp to grid boundaries
        x = fmaxf(0.5f, fminf(M + 0.5f, x));
        y = fmaxf(0.5f, fminf(N + 0.5f, y));
        z = fmaxf(0.5f, fminf(O + 0.5f, z));

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1;
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        d[IX(i, j, k)] =
            s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
                  t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
            s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
                  t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
    }
}

void advect(int M, int N, int O, int b, float *d_d, float *d_d0, float *d_u, float *d_v, float *d_w, float dt) {
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + 2 + threadsPerBlock.z - 1) / threadsPerBlock.z);

    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;
    advect_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_d, d_d0, d_u, d_v, d_w, dt, dtX, dtY, dtZ);
    set_bnd(M, N, O, b, d_d);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// @@ project

__global__ void project1_kernel(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i > 0 && i <= M && j > 0 && j <= N && k > 0 && k <= O) {
        div[IX(i, j, k)] = -0.5f * (
            u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
            v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
            w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]
        ) / max(M, max(N, O));
        p[IX(i, j, k)] = 0.0f;
    }
}

__global__ void project2_kernel(int M, int N, int O, float *u, float *v, float *w, float *p) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i > 0 && i <= M && j > 0 && j <= N && k > 0 && k <= O) {
        u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    }
}

// Função host para projeção
void project(int M, int N, int O, float *d_u, float *d_v, float *d_w, float *d_p, float *d_div) {
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + 2 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + 2 + threadsPerBlock.z - 1) / threadsPerBlock.z);

    project1_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, d_u, d_v, d_w, d_p, d_div);
    
    //cudaDeviceSynchronize();

    set_bnd(M, N, O, 0, d_div);
    set_bnd(M, N, O, 0, d_p);
    
    lin_solve(M, N, O, 0, d_p, d_div, 1, 6);
    
    project2_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, d_u, d_v, d_w, d_p);
    
    //cudaDeviceSynchronize();

    set_bnd(M, N, O, 1, d_u);
    set_bnd(M, N, O, 2, d_v);
    set_bnd(M, N, O, 3, d_w);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// @@ dens_step - original

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// @@ vel_step - original

void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);
  SWAP(u0, u);
  diffuse(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse(M, N, O, 3, w, w0, visc, dt);
  project(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);
}
