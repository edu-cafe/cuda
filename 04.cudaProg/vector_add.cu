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
void vectorAdd(int n, double *a, double *b, double *c) {
  int i;
  for(i=0; i<n; i++) {
    c[i] = a[i] + b[i];
  }
}

void print_rst(int n, double *h_c_cuda, double *h_c)
{
  int i;
  for (i = 0; i < n; i++){
    if(i%10 == 0) printf("--->i:%d\n", i);
    printf("%.1f(%.1f) ", h_c_cuda[i], h_c[i]);
  }
}

int main(int argc, char **argv)
{
  double *h_A, *h_B, *h_C, *h_C_simple;
  int n2, N;
  int i;
  int size=0;

  struct timeval tv1, tv2;


  /* get the size of the matrix from the command line */
  if (argc <2 ) N= 1024*10;
  else N = atoi(argv[1]);
  n2 = N;
  size = n2 * sizeof(double);

  printf("\nRunning dgemm test for %d by %d matricies.\n", N, N);
  
  /* Allocate host memory for the matrices */
  h_A = (double *)malloc(size);
  h_B = (double *)malloc(size);
  h_C = (double *)malloc(size);
  h_C_simple = (double *)malloc(size);

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
  vectorAdd(N, h_A, h_B, h_C_simple);
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

#define	THREADS_PER_BLOCK 128

__constant__ double alpha = 1.0f;
__constant__ double beta = 0.0f;

/* ======================================================= */
/* Cuda implementation of dgemm */
/* ======================================================= */
__global__ void vectorAdd(double *a, double *b, double *c) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  c[i] = a[i] + b[i];
}

void print_rst(int n, double *h_c_cuda, double *h_c)
{
  int i;
  for (i = 0; i < n; i++){
    if(i%10 == 0) printf("--->i:%d\n", i);
    printf("%.1f(%.1f) ", h_c_cuda[i], h_c[i]);
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
  //if (argc <2 ) N= 275;
  if (argc <2 ) N= 1024*10;
  else N = atoi(argv[1]);
  n2 = N;
  size = n2 * sizeof(double);

  printf("\nRunning dgemm test for %d by %d matricies.\n", N, N);

  /* Allocate host memory for the matrices */
  h_A = (double *)malloc(size);
  h_B = (double *)malloc(size);
  h_C = (double *)malloc(size);
  h_C_cuda = (double *)malloc(size);

  /* Fill the matrices with test data */
  for (i = 0; i < n2; i++){
    h_A[i] = rand() / (double)RAND_MAX;
    h_B[i] = rand() / (double)RAND_MAX;
    h_C[i] = rand() / (double)RAND_MAX;
    h_C_cuda[i] = h_C[i];
  }
  
  /* Allocate device memory for the matrices */
  cudaMalloc( (void**)&d_A, size );
  cudaMalloc( (void**)&d_B, size );
  cudaMalloc( (void**)&d_C, size );

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

  printf("\tTesting CUDA implementation of dgemm function.\n");
  gettimeofday(&tv1, NULL);
  /* Performs operation using cuda code */
  //cuda_dgemm <<< 1, N>>>(N, alpha, d_A, d_B, beta, d_C);
  vectorAdd <<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_A, d_B, d_C);
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
