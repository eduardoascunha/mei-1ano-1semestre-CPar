#include "stencil.h"
// Inclui o cabeçalho "stencil.h", que provavelmente contém declarações e funções auxiliares usadas no programa.

#define NUM_BLOCKS 128
//#define NUM_BLOCKS 256
//#define NUM_BLOCKS 512
// Define o número de blocos CUDA usados para o cálculo. Apenas uma dessas definições deve estar ativa.

#define NUM_THREADS_PER_BLOCK 256
// Define o número de threads por bloco. No CUDA, cada bloco contém várias threads.

#define SIZE NUM_BLOCKS*NUM_THREADS_PER_BLOCK
// Define o tamanho total do vetor, que é o produto do número de blocos e threads por bloco.

using namespace std;
// Permite usar as funções da biblioteca padrão C++ sem o prefixo `std::`.

__global__
// Indica que a função a seguir será executada na GPU como um kernel CUDA.
void stencilKernel (float *a, float *c) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Calcula o identificador único da thread, combinando o índice do bloco e o índice da thread dentro do bloco.

    c[id] = 0;
    // Inicializa o elemento correspondente no vetor de saída como zero.

    for (int n = -2; n <= 2; n++) {
        // Itera pelos vizinhos no intervalo de -2 a +2 em relação à posição atual.

        if ((id + n >= 0) && (id + n < SIZE)) {
            // Verifica se o índice vizinho está dentro dos limites do vetor.

            c[id] += a[id + n];
            // Soma o valor do vizinho no vetor de entrada ao elemento correspondente no vetor de saída.
        }
    }
}
// Fim da função kernel.

void stencil (float *a, float *c) {
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    // Marca o tempo inicial para medir o desempenho da execução sequencial no CPU.

    for (int i = 0; i < SIZE; i++) {
        // Itera por todos os elementos do vetor.

        for (int n = -2; n <= 2; n++) {
            // Para cada elemento, considera os vizinhos no intervalo de -2 a +2.

            if ((i + n >= 0) && (i + n < SIZE))
                c[i] += a[i + n];
                // Soma o valor dos vizinhos ao elemento correspondente no vetor de saída.
        }
    }

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    // Marca o tempo final.

    cout << endl << "Sequential CPU execution: "
         << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
         << " microseconds" << endl << endl;
    // Calcula e imprime o tempo gasto para a execução sequencial em microsegundos.
}
// Fim da função sequencial no CPU.

void launchStencilKernel (float *a, float *c) {
    float *da, *dc;
    // Declara ponteiros para memória na GPU.

    int bytes = SIZE * sizeof(float);
    // Calcula o tamanho em bytes necessário para armazenar o vetor.

    cudaMalloc ((void**) &da, bytes);
    cudaMalloc ((void**) &dc, bytes);
    // Aloca memória na GPU para os vetores de entrada e saída.

    checkCUDAError("mem allocation");
    // Verifica se a alocação de memória foi bem-sucedida.

    cudaMemcpy (da, a, bytes, cudaMemcpyHostToDevice);
    // Copia os dados do vetor de entrada do CPU (host) para a GPU (device).

    checkCUDAError("memcpy h->d");
    // Verifica se a cópia para o device foi bem-sucedida.

    startKernelTime ();
    // Inicia a medição do tempo de execução do kernel.

    stencilKernel <<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK >>> (da, dc);
    // Lança o kernel CUDA com o número de blocos e threads definidos.

    stopKernelTime ();
    // Para a medição do tempo do kernel.

    checkCUDAError("kernel invocation");
    // Verifica se a execução do kernel foi bem-sucedida.

    cudaMemcpy (c, dc, bytes, cudaMemcpyDeviceToHost);
    // Copia os dados do vetor de saída da GPU para o CPU.

    checkCUDAError("memcpy d->h");
    // Verifica se a cópia para o host foi bem-sucedida.

    cudaFree(da);
    cudaFree(dc);
    // Libera a memória alocada na GPU.

    checkCUDAError("mem free");
    // Verifica se a liberação da memória foi bem-sucedida.
}
// Fim da função que lança o kernel CUDA.

int main( int argc, char** argv) {
    float a[SIZE], b[SIZE], c[SIZE];
    // Declara vetores no host para os dados de entrada e saída.

    for (unsigned i = 0; i < SIZE; ++i)
        a[i] = (float) rand() / RAND_MAX;
        // Inicializa o vetor de entrada com valores aleatórios normalizados entre 0 e 1.

    stencil (a, b);
    // Executa a versão sequencial no CPU.

    launchStencilKernel (a, c);
    // Executa a versão paralela no GPU.

    return 0;
    // Finaliza o programa.
}