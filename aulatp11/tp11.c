/// EXC 1
__global__  
void stencilKernel(float *a, float *c) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; // ID global
    int lid = threadIdx.x; // ID local dentro do bloco

    // Declaração de memória compartilhada
    __shared__ float temp[NUM_THREADS_PER_BLOCK + 4]; 

    // Carregar os valores necessários para a memória compartilhada
    temp[lid + 2] = a[id]; // Dados principais
    if (lid < 2) { 
        // Carregar bordas extras
        if (id >= 2) temp[lid] = a[id - 2];
        if (id + blockDim.x < SIZE) temp[lid + blockDim.x + 2] = a[id + blockDim.x];
    }

    // Sincronizar as threads do bloco
    __syncthreads();

    // Inicializar o valor de saída
    c[id] = 0;

    // Computação usando a memória compartilhada
    for (int n = -2; n <= 2; n++) {
        if ((id + n >= 0) && (id + n < SIZE)) {
            c[id] += temp[lid + 2 + n];
        }
    }
}


/// EXC 2
//a
__global__  
void mmKernel(float *a, float *b, float *c, int N) { 
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Linha da matriz C
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Coluna da matriz C

    if (i < N && j < N) { 
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += a[i * N + k] * b[k * N + j];
        }
        c[i * N + j] = sum;
    }
}

int main() {
    int N = 1024; // Tamanho da matriz
    dim3 threadsPerBlock(16, 16); // Cada bloco tem 16x16 threads
    dim3 blocksPerGrid(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Chamar o kernel
    mmKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
}


//b
dim3 threadsPerBlock(256, 1); // Bloco de 256x1 threads
dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, N); // Ajuste do grid

__global__  
void mmKernelOptimized(float *a, float *b, float *c, int N) { 
    int i = blockIdx.y; // Linha da matriz C
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Coluna da matriz C

    if (i < N && j < N) { 
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += a[i * N + k] * b[k * N + j];
        }
        c[i * N + j] = sum;
    }
}


//c
__global__  
void mmKernelShared(float *a, float *b, float *c, int N) { 
    __shared__ float tileA[16][16]; // Bloco da matriz A
    __shared__ float tileB[16][16]; // Bloco da matriz B

    int row = blockIdx.y * blockDim.y + threadIdx.y; // Linha em C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Coluna em C

    float sum = 0.0;

    for (int t = 0; t < (N + 16 - 1) / 16; t++) {
        // Carregar blocos de A e B na memória compartilhada
        if (row < N && t * 16 + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = a[row * N + t * 16 + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && t * 16 + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = b[(t * 16 + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads(); // Sincronizar threads do bloco

        // Computar a multiplicação parcial
        for (int k = 0; k < 16; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads(); // Garantir que os dados antigos não sejam sobrescritos
    }

    // Escrever o resultado
    if (row < N && col < N) {
        c[row * N + col] = sum;
    }
}
