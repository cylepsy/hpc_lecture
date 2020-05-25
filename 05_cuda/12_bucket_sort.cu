#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_init(int *bucket, int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >=range) return;
  bucket[i] = 0;
  __syncthreads();
}

__global__ void bucket_find(int *bucket, int *key, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //bucket[key[i]]++;
  if (i >=n) return;
  atomicAdd(&bucket[key[i]],1);
  __syncthreads();
}

__global__ void bucket_sort(int *bucket, int *key, int range, int n){
  //for (int i=0, j=0; i<range; i++) {
   // for (; bucket[i]>0; bucket[i]--) {
    //  key[j++] = i;
    //}
  //}
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int (i >= range) return;
  for (int a=0, b=0; a <= i; b++){
    key[i] = b;
    a += bucket[b];
  }
  __syncthreads();

}

int main() {
  int n = 50;
  int range = 5;
  //std::vector<int> key(n);
  int *key;
  cudaMallocManaged(&key, n * sizeof(int));
  
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

/*
  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
*/

  int *bucket;
  //allocate memory for bucket array
  cudaMallocManaged(&bucket, range * sizeof(int));
  // initialize bucket array
  bucket_init<<<1, range>>>(bucket, range);
  cudaDeviceSynchronize();

/*
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
*/
  bucket_find<<<1,n>>>(bucket, key, n);
  cudaDeviceSynchronize();

/*
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
*/
  bucket_sort<<<1,range>>>(bucket, key, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
