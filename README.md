# CUDA Programming Resources

## Concepts:
- Heterogeneous Computing
- Blocks
- Threads
- Indexing
- Shared memory
- __syncThreads()
- Asynchronous operation
- Handling errors
- Managing devices

## Terminology
- *Host*	The CPU and its memory (host memory)
- *Device*	The GPU and its memory (device memory)

## Heterogeneous Computing
```
parallel_fn(){
	//...
	//parallel code
	//...
}
code(){
	//serial code
	//parallel code
	//serial code
}
```

### Simple Processing Flow
1. Copy input data from CPU memory to GPU memory
2. Load GPU program and execute, caching data on chip for performance
3. Copy results from GPU memory to CPU memory


### Hello World
```
int main(void){
	printf("Hello World!\n");
	return 0;
}
```

- Standard C code
- nvcc can be used to compile
```
$ nvcc HelloWorld.cu
$ ./a.out
```

### Hello World with Device Code
```
#include <stdio.h>
__global__ void mykernel(void){}

int main(void){
	mykernel<<<1,1>>>();
	printf("Hello World!\n");
	return 0;
}
```

Kernel code is processed by NVIDIA compiler
`__global__ void mykernel(void)`

Host code is processed by host compiler (e.g. gcc)

Triple angle brackets indicate call from host to device code
`mykernel<<<1,1>>>();`

### Memory Management
- *Device pointers* point to *GPU* memory
	- May be passed to/from host code
	- May not be dereferenced in host code
- *Host pointers* point to *CPU* memory
	- May be passed to/from host code
	- May not be dereferenced in host code

```
cudaMalloc(); // == malloc()
cudaFree(); // == free()
cudaMemcpy(): // == memcpy()
```

### Addition on the Device
```
__global__ void add(int *a, int *b, int *c){
	*c = *a + *b;
}

int main(void){
	int a, b, c; //host copies of a, b, c
	int *d_a, *d_b, *d_c; //device copies of a, b, c
	int size = sizeof(int);
	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	// Setup input values
	a = 2;
	b = 7;
	//Copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
	//Launc add() kernel on GPU
	add<<<1,1>>>(d_a,d_b,d_c);
	//Copy result back to host
	cudaMemcpy(d_b, &b, size, cudaMemcpyDeviceToHost);
	//Cleanup
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}
```

### Blocks
```
add<<<1,1>>>(); //execute once
add<<<N,1>>>(); //execute N times in parallel
```

#### Vector Addition
```
__global__ void add(int *a, int *b, int *c){
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
#define N 512

int main(void){
	int a, b, c; //host copies of a, b, c
	int *d_a, *d_b, *d_c; //device copies of a, b, c
	int size = N * sizeof(int);
	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	// Setup input values
	a = (int *)malloc(size); random_ints(a, N);
	b = (int *)malloc(size); random_ints(b, N);
	c = (int *)malloc(size);
	//Copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
	//Launc add() kernel on GPU with N blocks
	add<<<N,1>>>(d_a,d_b,d_c);
	//Copy result back to host
	cudaMemcpy(d_b, &b, size, cudaMemcpyDeviceToHost);
	//Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}
```

- Each parallel invocation of add() is referred to as a block
- The set of blocks is referred to as a grid
- Each invocation can refer to its block index using blockId.x
- By using blockIdx.x to index into the array, each block handles a different index

Parallel calls, arbitrary schedule
*block 0* c[0] = a[0] + b[0]
*block 1* c[1] = a[1] + b[1]
*block 2* c[2] = a[2] + b[2]
...
*block n* c[n] = a[n] + b[n]


```
#include <iostream>
#include <math.h>
__global__
void add(int n, float *x, float *y)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	for(int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}
int main(void)
{
	int N = 1<<20;
	float *x, *y;
	int size = N*sizeof(float);
	// allocate Unified Memory -- accessible from CPU or GPU
	cudaMallocManaged(&x,size);
	cudaMallocManaged(&y,size);
	// initialize x and y arrays on the host
	for(int i = 0; i < N; i++){
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
	// RUN kernel on 1M elements on the GPU
	add<<<1,256>>>(N,x,y);
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	// Check for errors (all values should be 3.0f)
	for(int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	std::cout << "Max error: " << maxError << std::endl;
	// Free memory
	cudaFree(x);
	cudaFree(y);
	return 0;
}
```

