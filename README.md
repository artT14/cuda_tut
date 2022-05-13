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

### Heterogeneous Computing
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