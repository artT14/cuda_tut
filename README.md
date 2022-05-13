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

``
$ nvcc HelloWorld.cu
$ ./a.out
``

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