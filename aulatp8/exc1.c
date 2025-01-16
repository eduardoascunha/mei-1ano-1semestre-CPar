// exc1
// a e b
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, msg;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        msg = 123456;
        MPI_Send(&msg, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&msg, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
        printf("Processo %d recebeu: %d\n", rank, msg);

        if (rank < size - 1) { // envia para o próximo processo na pipeline
            MPI_Send(&msg, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}


// c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, msg, i;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        for (i = 0; i < 10; i++) {
            msg = 123456 + i;
            MPI_Send(&msg, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    } else {
        for (i = 0; i < 10; i++) {
            MPI_Recv(&msg, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
            printf("Processo %d recebeu a mensagem %d: %d\n", rank, i, msg);

            if (rank < size - 1) {
                MPI_Send(&msg, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
