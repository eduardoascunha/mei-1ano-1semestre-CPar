#include <stdio.h>
#include "quicksort.h"

void partition(float*arr, int *i, int *j);
void quickSort_internal(float* arr, int left, int right) ;

void quickSort(float* arr, int size) 
{

	int i = 0, j = size;
  	/* PARTITION PART */
        partition(arr, &i, &j);

	if (0 < j){ 
        	quickSort_internal(arr, 0, j);
    	}
	if (i< size){
        	quickSort_internal(arr, i, size);
	}
}

void partition(float*arr, int *i, int *j){
	//float pivot = arr[((*j)+(*i)) / 2];
	float pivot = arr[(*j)];
	while ((*i) <= (*j)) {
		while (arr[(*i)] < pivot)
			(*i)++;
		while (arr[(*j)] > pivot)
			(*j)--;
		if ((*i) <= (*j)) {
			float tmp = arr[(*i)];
			arr[(*i)] = arr[(*j)];
			arr[(*j)] = tmp;
			(*i)++;
			(*j)--;
		}
	}
}

void partition_internal(float*arr, int *i, int *j){
	//float pivot = arr[((*j)+(*i)) / 2];
	float pivot = arr[(*j)];
	while ((*i) <= (*j)) {
		while (arr[(*i)] < pivot)
			(*i)++;
		while (arr[(*j)] > pivot)
			(*j)--;
		if ((*i) <= (*j)) {
			float tmp = arr[(*i)];
			arr[(*i)] = arr[(*j)];
			arr[(*j)] = tmp;
			(*i)++;
			(*j)--;
		}
	}
}

void quickSort_internal(float* arr, int left, int right) 
{
	int i = left, j = right;
  	/* PARTITION PART */
        partition_internal(arr, &i, &j) ;

	/* RECURSION PART */
	if (left < j){ 
        	quickSort_internal(arr, left, j);
    	}
	if (i< right){
        	quickSort_internal(arr, i, right);
    	}
}



////////////////////
// @@@@@@@@@@@@@@@@
///////////////////

#include <stdio.h>
#include <omp.h>
#include "quicksort.h"

void partition(float* arr, int *i, int *j);
void quickSort_internal(float* arr, int left, int right);

void quickSort(float* arr, int size) {
    int i = 0, j = size;
    /* PARTITION PART */
    partition(arr, &i, &j);

    #pragma omp parallel // Inicia a região paralela
    {
        #pragma omp single // Apenas uma thread executa a parte de criação de tarefas
        {
            if (0 < j) {
                quickSort_internal(arr, 0, j);
            }
            if (i < size) {
                quickSort_internal(arr, i, size);
            }
        }
    }
}

void partition(float* arr, int *i, int *j) {
    float pivot = arr[(*j)];
    while ((*i) <= (*j)) {
        while (arr[(*i)] < pivot)
            (*i)++;
        while (arr[(*j)] > pivot)
            (*j)--;
        if ((*i) <= (*j)) {
            float tmp = arr[(*i)];
            arr[(*i)] = arr[(*j)];
            arr[(*j)] = tmp;
            (*i)++;
            (*j)--;
        }
    }
}

void quickSort_internal(float* arr, int left, int right) {
    int i = left, j = right;
    /* PARTITION PART */
    partition(arr, &i, &j);

    /* RECURSION PART */
    if (left < j) {
        #pragma omp task // Cria uma tarefa para a primeira metade
        quickSort_internal(arr, left, j);
    }
    if (i < right) {
        #pragma omp task // Cria uma tarefa para a segunda metade
        quickSort_internal(arr, i, right);
    }

    #pragma omp taskwait // Espera todas as tarefas terminarem
}
