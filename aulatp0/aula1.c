#include <stdio.h>
#include <stdlib.h>

// Função para alocar memória para uma matriz
int** allocateMatrix(int size) {
    int **matrix = (int **)malloc(size * sizeof(int *));
    for (int i = 0; i < size; i++) {
        matrix[i] = (int *)malloc(size * sizeof(int));
    }
    return matrix;
}

// Função para inicializar a matriz com valores aleatorios
void initializeMatrix(int **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (rand() % 9);
        }
    }
}

// Função para inicializar a matriz com valores aleatorios
void initializeMatrixC(int **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = 0;
        }
    }
}

// Função para imprimir a matriz
void printMatrix(int **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Função para liberar a memória alocada
void freeMatrix(int **matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void multMatrix(int **A, int **B, int **C, int N){
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            for(int k=0; k<N; k++)
                C[i][j] += A[i][k] * B[k][j];
}



int main() {
    int N = 5; // Tamanho das matrizes (pode ser alterado conforme necessário)    
    // Criando e inicializando as matrizes A, B e C
    int **A = allocateMatrix(N);
    int **B = allocateMatrix(N);
    int **C = allocateMatrix(N);
    
    initializeMatrix(A, N);
    initializeMatrix(B, N);
    initializeMatrixC(C, N);
    
    // Imprimindo as matrizes
    printf("Matriz A:\n");
    printMatrix(A, N);
    printf("\nMatriz B:\n");
    printMatrix(B, N);
    printf("\nMatriz C:\n");
    printMatrix(C, N);

    multMatrix(A,B,C, N);
    printf("\n POS MULTIPLICACAO\n");
    printf("Matriz A:\n");
    printMatrix(A, N);
    printf("\nMatriz B:\n");
    printMatrix(B, N);
    printf("\nMatriz C:\n");
    printMatrix(C, N);

    
    // Liberando a memória alocada
    freeMatrix(A, N);
    freeMatrix(B, N);
    freeMatrix(C, N);
    
    return 0;
}
