#include<stdio.h>
#include<stdlib.h>

#ifndef size
#define size 512
#endif

double *A, *B, *C;

void alloc() {
    A = (double *) malloc(size*size*sizeof(double));
    B = (double *) malloc(size*size*sizeof(double));
    C = (double *) malloc(size*size*sizeof(double));
}

void init() {
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            A[i*size+j] = rand();
            B[i*size+j] = rand();
            C[i*size+j] = 0;
        }
    }
}

void mmult() {
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            for(int k=0; k<size; k++) {
                C[i*size+j] += A[i*size+k] * B[k*size+j];
            }
        }
    }
}

int main() {
    alloc();
    init();
    mmult();

    printf("%f\n", C[size/2+5]);
}



/////////////////
// exc1 só com bcast
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#ifndef size
#define size 512
#endif

double *A, *B, *C;

void alloc() {
    A = (double *) malloc(size * size * sizeof(double));
    B = (double *) malloc(size * size * sizeof(double));
    C = (double *) malloc(size * size * sizeof(double));
}

void init() {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = rand() % 100; 
            B[i * size + j] = rand() % 100;
            C[i * size + j] = 0;
        }
    }
}

int main(int argc, char **argv) {
    int rank, num_procs;

    // Inicializa MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Aloca memória apenas no processo mestre
    if (rank == 0) {
        alloc();
        init();
    }

    // Alocar espaço para matriz B em todos os processos
    if (rank != 0) {
        B = (double *) malloc(size * size * sizeof(double));
    }

    // Broadcast da matriz B para todos os processos
    MPI_Bcast(B, size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Divisão das linhas da matriz C
    int rows_per_process = size / num_procs;
    int extra_rows = size % num_procs;

    // Calcular quais linhas este processo deve tratar
    int start_row = rank * rows_per_process + (rank < extra_rows ? rank : extra_rows);
    int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);

    // Alocar espaço para as linhas da matriz A necessárias
    int num_rows = end_row - start_row;
    double *local_A = (double *) malloc(num_rows * size * sizeof(double));
    double *local_C = (double *) malloc(num_rows * size * sizeof(double));

    // Processo mestre envia as linhas de A para os outros processos
    if (rank == 0) {
        for (int p = 1; p < num_procs; p++) {
            int p_start_row = p * rows_per_process + (p < extra_rows ? p : extra_rows);
            int p_end_row = p_start_row + rows_per_process + (p < extra_rows ? 1 : 0);
            MPI_Send(&A[p_start_row * size], (p_end_row - p_start_row) * size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }
        // Mestre mantém suas próprias linhas
        for (int i = 0; i < num_rows * size; i++) {
            local_A[i] = A[start_row * size + i];
        }
    } else {
        MPI_Recv(local_A, num_rows * size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Cada processo calcula suas linhas de C
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < size; j++) {
            local_C[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                local_C[i * size + j] += local_A[i * size + k] * B[k * size + j];
            }
        }
    }

    // Processo mestre reúne as linhas calculadas
    if (rank == 0) {
        for (int i = 0; i < num_rows * size; i++) {
            C[start_row * size + i] = local_C[i];
        }
        for (int p = 1; p < num_procs; p++) {
            int p_start_row = p * rows_per_process + (p < extra_rows ? p : extra_rows);
            int p_end_row = p_start_row + rows_per_process + (p < extra_rows ? 1 : 0);
            MPI_Recv(&C[p_start_row * size], (p_end_row - p_start_row) * size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_C, num_rows * size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // Processo mestre exibe o resultado para validação
    if (rank == 0) {
        printf("%f\n", C[size / 2 + 5]);
    }

    // Liberar memória
    free(local_A);
    free(local_C);
    if (rank != 0) {
        free(B);
    }
    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }

    // Finaliza MPI
    MPI_Finalize();
    return 0;
}

// exc1 com bcast scatter e gather
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#ifndef size
#define size 512
#endif

double *A, *B, *C;

void alloc() {
    A = (double *) malloc(size * size * sizeof(double));
    B = (double *) malloc(size * size * sizeof(double));
    C = (double *) malloc(size * size * sizeof(double));
}

void init() {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i * size + j] = rand() % 100; 
            B[i * size + j] = rand() % 100;
            C[i * size + j] = 0;
        }
    }
}

int main(int argc, char **argv) {
    int rank, num_procs;

    // Inicializa o ambiente MPI
    MPI_Init(&argc, &argv);

    // Obtém o identificador do processo atual (rank) e o número total de processos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Calcula o número de linhas da matriz A que cada processo irá tratar
    int rows_per_process = size / num_procs;  // Linhas por processo
    int extra_rows = size % num_procs;        // Linhas extras para distribuir desigualmente

    int *send_counts = NULL;  // Quantidade de elementos para cada processo (para Scatterv)
    int *displs = NULL;       // Deslocamentos de dados para cada processo (para Scatterv)

    if (rank == 0) {  // Processo mestre
        alloc();  // Aloca memória para A, B, e C no mestre
        init();   // Inicializa as matrizes A e B com valores aleatórios

        // Configuração de send_counts e displs para Scatterv e Gatherv
        send_counts = (int *) malloc(num_procs * sizeof(int));  // Array de contagens
        displs = (int *) malloc(num_procs * sizeof(int));       // Array de deslocamentos
        int offset = 0;  // Deslocamento inicial

        // Calcula send_counts e displs para cada processo
        for (int p = 0; p < num_procs; p++) {
            // Cada processo recebe rows_per_process linhas, com processos extras recebendo +1 linha
            int num_rows = rows_per_process + (p < extra_rows ? 1 : 0);
            send_counts[p] = num_rows * size;  // Total de elementos (linhas * colunas)
            displs[p] = offset;                // Posição inicial dos dados no array A
            offset += num_rows * size;        // Incrementa o deslocamento
        }
    }

    // Todos os processos precisam receber a matriz B
    if (rank != 0) {
        B = (double *) malloc(size * size * sizeof(double));  // Aloca B nos processos não mestres
    }
    MPI_Bcast(B, size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);  // Envia B para todos os processos

    // Determina o número de linhas locais para o processo atual
    int local_rows = rows_per_process + (rank < extra_rows ? 1 : 0);

    // Aloca memória local para a parte de A e a parte correspondente de C
    double *local_A = (double *) malloc(local_rows * size * sizeof(double));  // Parte local de A
    double *local_C = (double *) malloc(local_rows * size * sizeof(double));  // Parte local de C

    // Distribui partes da matriz A entre os processos
    MPI_Scatterv(A, send_counts, displs, MPI_DOUBLE, local_A,
                 local_rows * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Cada processo calcula suas linhas de C
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < size; j++) {
            local_C[i * size + j] = 0;  // Inicializa o valor
            for (int k = 0; k < size; k++) {
                local_C[i * size + j] += local_A[i * size + k] * B[k * size + j];  // Soma parcial
            }
        }
    }

    // Coleta as partes calculadas de C de volta no processo mestre
    MPI_Gatherv(local_C, local_rows * size, MPI_DOUBLE, C,
                send_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Processo mestre exibe um elemento da matriz C para validação
    if (rank == 0) {
        printf("%f\n", C[size / 2 * size + 5]);  // Exibe um valor arbitrário da matriz C

        // Libera memória no processo mestre
        free(A);
        free(B);
        free(C);
        free(send_counts);
        free(displs);
    }

    // Libera memória local em todos os processos
    free(local_A);
    free(local_C);
    if (rank != 0) {
        free(B);  // Libera B nos processos que o alocaram
    }

    // Finaliza o ambiente MPI
    MPI_Finalize();
    return 0;
}


////////////
////////////
// exc2
#include <mpi.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "PrimeServer.cpp"  // Assuming PrimeServer class is in this file

#define MAXP 1000000  // Máximo número para calcular os primos
#define SMAXP 1000    // Raiz quadrada do número máximo
#define PACK MAXP/10  // Número de números por pacote

int main(int argc, char **argv) {
    int rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Instanciando PrimeServer para cada filtro
    PrimeServer *ps1 = new PrimeServer();
    PrimeServer *ps2 = new PrimeServer();
    PrimeServer *ps3 = new PrimeServer();

    // Dividindo a faixa de primos para cada filtro
    ps1->initFilter(1, SMAXP / 3, SMAXP);           // PrimeServer 1
    ps2->initFilter(SMAXP / 3 + 1, 2 * SMAXP / 3, SMAXP); // PrimeServer 2
    ps3->initFilter(2 * SMAXP / 3 + 1, SMAXP, SMAXP);   // PrimeServer 3

    int *ar = new int[PACK / 2];  // Buffer para armazenar os números gerados

    // Processos irão gerar e processar os pacotes de números
    for (int i = 0; i < 10; i++) {
        // Processo 0 gera o pacote e distribui entre os outros processos
        if (rank == 0) {
            generate(i * PACK, (i + 1) * PACK, ar);  // Gera o pacote de números
        }

        // Envia o pacote de números para o próximo processo no pipeline
        MPI_Bcast(ar, PACK / 2, MPI_INT, 0, MPI_COMM_WORLD);

        // Cada processo aplica o filtro de primos
        ps1->mprocess(ar, PACK / 2);
        ps2->mprocess(ar, PACK / 2);
        ps3->mprocess(ar, PACK / 2);

        // Envia o pacote processado para o próximo processo, se necessário
        if (rank < nprocs - 1) {
            MPI_Send(ar, PACK / 2, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    // Apenas o último processo exibe as estatísticas
    if (rank == nprocs - 1) {
        ps3->end();  // Mostra a quantidade de primos encontrados
    }

    // Limpeza da memória
    delete[] ar;

    MPI_Finalize();
    return 0;
}
