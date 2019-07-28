/* GEMM is a General Matrix Multiply - a subroutine in the Basic Linear Algebra Subprograms library*/

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifdef HOST
/* ======================================================= */
/* Simple implementation of dgemm */
/* ======================================================= */
static void simple_dgemm(int n, double alpha, const double *A, const double *B,
                         double beta, double *C) {
  int i, j, k;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j){
      double prod = 0;
      for (k = 0; k < n; ++k){
	prod += A[k * n + i] * B[j * n + k];
      }
      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}

void print_rst(int n, double *h_C_simple, double *h_C)
{
  int i, j;
  for (i = 0; i < n; i++){
    printf("----->i:%d\n", i);
    for (j = 0; j < n; j++){
      printf("%.1f(%.1f) ", h_C_simple[i*n +j], h_C[i*n + j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  double *h_A, *h_B, *h_C, *h_C_simple;
  double alpha = 1.0f;
  double beta = 0.0f;
  int n2, N;
  int i;

  struct timeval tv1, tv2;


  /* get the size of the matrix from the command line */
  if (argc <2 ) N= 275;
  else N = atoi(argv[1]);
  n2 = N * N;

  printf("\nRunning dgemm test for %d by %d matricies.\n", N, N);
  
  /* Allocate host memory for the matrices */
  h_A = (double *)malloc(n2 * sizeof(double) );
  h_B = (double *)malloc(n2 * sizeof(double) );
  h_C = (double *)malloc(n2 * sizeof(double) );
  h_C_simple = (double *)malloc(n2 * sizeof(double) );

  /* Fill the matrices with test data */
  for (i = 0; i < n2; i++){
    h_A[i] = rand() / (double)RAND_MAX;
    h_B[i] = rand() / (double)RAND_MAX;
    h_C[i] = rand() / (double)RAND_MAX;
    h_C_simple[i] = h_C[i];
  }

  //print_rst(N, h_C);

  printf("\tTesting simple C implementation of dgemm function.\n");
  gettimeofday(&tv1, NULL);
  /* Performs operation using plain C code */
  simple_dgemm(N, alpha, h_A, h_B, beta, h_C_simple);
  gettimeofday(&tv2, NULL);
  printf("\t\tdone...\n");
  printf("\t\tExecution time (in millisec): %.2f\n",
	 (double)(tv2.tv_usec-tv1.tv_usec)/1000 + 
	 (double)(tv2.tv_sec -tv1.tv_sec )*1000);

  print_rst(N, h_C_simple, h_C);

  
  /* Memory clean up */
  free(h_A); free(h_B); free(h_C); free(h_C_simple);

  return(0);
}

#else

__constant__ double alpha = 1.0f;
__constant__ double beta = 0.0f;

/* ======================================================= */
/* Cuda implementation of dgemm */
/* ======================================================= */
//__global__ void cuda_dgemm(int n, double alpha, const double *A, const double *B, double beta, double *C) {
__global__ void cuda_dgemm(int n, const double *A, const double *B, double *C) {
  int i = 0;
  int j = 0;
  int k = threadIdx.x;
  //double prod = 0;
  //prod = A[k * n + i] * B[j * n + k];
  //C[j * n + i] = alpha * prod + beta * C[j * n + i];
/*
  for(j=0; j<n; j++) {
    C[i * j + i] = alpha * (A[i * j + i] * B[i * j + i]) + beta * C[i * j + i];
  }
*/
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j){
      double prod = 0;
	prod = A[k * n + i] * B[j * n + k];
      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}

void print_rst(int n, double *h_C_simple, double *h_C)
{
  int i, j;
  for (i = 0; i < n; i++){
    printf("----->i:%d\n", i);
    for (j = 0; j < n; j++){
      printf("%.1f(%.1f) ", h_C_simple[i*n +j], h_C[i*n + j]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  double *h_A, *h_B, *h_C, *h_C_cuda;
  double *d_A, *d_B, *d_C;
  int n2, N;
  int i;
  int size=0;

  struct timeval tv1, tv2;

  /* get the size of the matrix from the command line */
  if (argc <2 ) N= 275;
  else N = atoi(argv[1]);
  n2 = N * N;

  printf("\nRunning dgemm test for %d by %d matricies.\n", N, N);

  /* Allocate host memory for the matrices */
  h_A = (double *)malloc(n2 * sizeof(double) );
  h_B = (double *)malloc(n2 * sizeof(double) );
  h_C = (double *)malloc(n2 * sizeof(double) );
  h_C_cuda = (double *)malloc(n2 * sizeof(double) );

  /* Fill the matrices with test data */
  for (i = 0; i < n2; i++){
    h_A[i] = rand() / (double)RAND_MAX;
    h_B[i] = rand() / (double)RAND_MAX;
    h_C[i] = rand() / (double)RAND_MAX;
    h_C_cuda[i] = h_C[i];
  }
  
  /* Allocate device memory for the matrices */
  cudaMalloc( (void**)&d_A, n2 * sizeof(double) );
  cudaMalloc( (void**)&d_B, n2 * sizeof(double) );
  cudaMalloc( (void**)&d_C, n2 * sizeof(double) );

  size = n2 * sizeof(double);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

  printf("\tTesting CUDA implementation of dgemm function.\n");
  gettimeofday(&tv1, NULL);
  /* Performs operation using cuda code */
  //cuda_dgemm <<< 1, N>>>(N, alpha, d_A, d_B, beta, d_C);
  cuda_dgemm <<< 1, N>>>(N, d_A, d_B, d_C);
  cudaMemcpy(h_C_cuda, d_C, size, cudaMemcpyDeviceToHost);
  gettimeofday(&tv2, NULL);
  printf("\t\tdone...\n");
  printf("\t\tExecution time (in millisec): %.2f\n",
	 (double)(tv2.tv_usec-tv1.tv_usec)/1000 + 
	 (double)(tv2.tv_sec -tv1.tv_sec )*1000);

  print_rst(N, h_C_cuda, h_C);

  
  /* Memory clean up */
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(h_A); free(h_B); free(h_C); free(h_C_cuda);

  return(0);
}
#endif
