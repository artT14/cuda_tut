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
`parallel_fn(){
	//...
	//parallel code
	//...
}
code(){
	//serial code
	//parallel code
	//serial code
}`