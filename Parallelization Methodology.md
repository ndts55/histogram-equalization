## Definitions
- Decomposition
	- Decompose computation into tasks
- Assignments
	- Assign tasks to processes
- Orchestration
	- Orchestrate data access, communication, and synchronization
- Mapping
	- Map processes to processors
## Methodology
### Decomposition
- we recognize these broad steps in the histogram equalization
	1. reading the input image
		1. ➿ convert ppm to hsl or yuv, if applicable
	2. ➿ calculating the histogram of the image
	3. ➿ calculating the cumulative distribution of the histogram
	4. ➿ calculating the lookup table
	5. ➿ map the input image to the equalized output image
	6. write the equalized output image
		1. ➿ convert to ppm first, if applicable
	- Note that each of these steps has be completed before the next step can begin
	- steps marked with ➿ rely on a loop
- most for-loops can be decomposed into tasks
	- ✅ conversion between formats relies on a loop where each iteration is independent of another -> parallelizable
	- ✅ histogram calculation relies on a loop where each iteration is independent of another -> parallelizable
	- ❌ calculating the cumulative distribution also relies on a for-loop but the iterations are **not** independent -> not parallelizable
		- only rank 0 calculates the cumulative distribution and scatters the data afterwards
	- ✅ calculating the lookup table relies on a loop where each iteration is independent of another -> parallelizable
	- ✅ mapping to the output image relies on a loop where each iteration is independent of another -> parallelizable
- the number of tasks in each step is equal to the number of iterations in the corresponding loop
	- this gives us lots of tasks to keep the processes busy
- for the mpi+omp version tasks were further split to utilize all available omp threads
### Assignments
- let `n` be the number of processes
- we divide all tasks into `n` similarly sized chunks
	- if `n` does not evenly divide the number of tasks one process will be assigned the remaining tasks
	- gives each process a similarly sized workload -> load balance
	- reduced communication and synchronization to single scattering of the required data
### Orchestration
- let `n` be the number of processes
- the basic pattern to parallelize the parallelizable loops (identified above) is:
	1. calculate the chunk sizes and offets to create `n` chunks of tasks
	2. scatter data to `n` processes
	3. do the partial calculations in each process
	4. gather or reduce data from `n` processes
- conversion between formats
	- the number of tasks is equal to the image size
	- split tasks into `n` chunks by scattering image data
		- requires three communication calls because each image format has three arrays that make up the image data index-wise
	- gather converted image data in rank 0
		- also requires three communication calls for the same reason
- histogram calculation
	- the number of tasks is equal to the image size
	- split tasks into `n` chunks by scattering image data
	- reduce partial histograms using `+`
		- we decided to reduce to all processes (`MPI_Allreduce`) because the full histogram is needed later for the calculatioin of the lookup table
- calculating the cumulative distribution
	- because each iteration depends on the previous iteration there is really only one big task
	- executed by rank 0
	- data is scattered in calculation of lookup table because that does not require the entire cumulative distribution
- calculating the lookup table
	- the number of tasks is equal to the number of possible values in the image data
		- in this case `256`
	- split tasks into `n` chunks by scattering the cumulative distribution
	- gather partial lookup tables into complete lookup table
		- we decided to gather to all processes (`MPI_Allgahterv`) because the full lookup table is needed when mapping the input image to the equalized output image
- mapping to the output image
	- the number of tasks is equal to the image size
	- split tasks into `n` chunks by scattering image data
	- gather partial output image data into complete image data
- another format conversion, if applicable
	- same as above
### Mapping
- 1 process = 1 processor for mpi-only
- 1 process = x processors for mpi+omp?