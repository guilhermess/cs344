//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <iostream>
#include <device_launch_parameters.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void exclusive_scan_add( const unsigned int * const values, unsigned int * const output, unsigned int size )
{
  extern __shared__ unsigned int exclusive_scan_smem[];
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if ( id < size )
    exclusive_scan_smem[tid] = values[id];
  else
    exclusive_scan_smem[tid] = 0;
  __syncthreads();

  //Reduce
  for ( unsigned int s = 1; s < blockDim.x; s <<= 1 )
  {
    if ( (tid+1) % s == 0 && ((tid+1)/s) % 2 == 0)
      exclusive_scan_smem[tid] = exclusive_scan_smem[tid] + exclusive_scan_smem[tid-s];
    __syncthreads();
  }

  if (tid == (blockDim.x - 1))
    exclusive_scan_smem[tid] = 0;
  __syncthreads();

  //Downsweep
  for ( unsigned int s = blockDim.x; s >= 1; s >>= 1 )
  {
    if ( (tid+1) % s == 0 && ((tid+1)/s) % 2 == 0)
    {
      unsigned int tmp = exclusive_scan_smem[tid];
      exclusive_scan_smem[tid] = exclusive_scan_smem[tid] + exclusive_scan_smem[tid - s];
      exclusive_scan_smem[tid-s] = tmp;
    }
    __syncthreads();
  }

  if ( tid < size )
    output[tid] = exclusive_scan_smem[tid] + values[tid];
}

__global__ void exclusive_scan_add_reduce( const unsigned int * const values,
                                           unsigned int *output,
                                           unsigned int *output_reduced,
                                           unsigned int size)
{
  extern __shared__ unsigned int smem[];
  int tid = threadIdx.x;
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  if ( id < size )
    smem[tid] = values[id];
  else
    smem[tid] = 0;

  __syncthreads();

  //Reduce
  for ( unsigned int s = 1; s < blockDim.x; s <<= 1 )
  {
    if ( (tid+1) % s == 0 && ((tid+1)/s) % 2 == 0) {
      smem[tid] = smem[tid] + smem[tid - s];
    }
    __syncthreads();
  }

  if (tid == blockDim.x - 1) {
    smem[tid] = 0;
  }
  __syncthreads();

  //Downsweep
  for ( unsigned int s = blockDim.x; s >= 1; s >>= 1 )
  {
    if ( (tid+1) % s == 0 && ((tid+1)/s) % 2 == 0)
    {
      unsigned int tmp = smem[tid];
      smem[tid] = smem[tid] + smem[tid - s];
      smem[tid - s] = tmp;
    }
    __syncthreads();
  }

  if ( id < size )
    output[id] = smem[tid];

  if ( tid == blockDim.x - 1 )
  {
    if ( id < size ) {
      output_reduced[blockIdx.x] = smem[tid] + values[id];
    }
    else {
      output_reduced[blockIdx.x] = smem[tid];
    }
  }
}

__global__ void exclusive_scan_add_merge( unsigned int * scan,
                                          unsigned int * reduce,
                                          unsigned int size )
{
  if ( blockIdx.x > 0 )
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid < size )
      scan[tid] = scan[tid] + reduce[blockIdx.x - 1];
  }
}

__global__ void sort_update_position( unsigned int *predicate,
                                      unsigned int *address,
                                      unsigned int *input_values,
                                      unsigned int *input_pos,
                                      unsigned int *output_values,
                                      unsigned int *output_pos,
                                      unsigned int *offset,
                                      unsigned int num_items,
                                      bool reset_offset )
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if ( tid < num_items )
  {
    if ( predicate[tid] == 1 ) {
      if ( !reset_offset ) {
        output_values[address[tid]] = input_values[tid];
        output_pos[address[tid]] = input_pos[tid];
      }
      else
      {
        output_values[address[tid] + *offset] = input_values[tid];
        output_pos[address[tid] + *offset] = input_pos[tid];
      }
    }
  }

  __syncthreads();
  if (tid == num_items - 1 ) {
    if ( !reset_offset ) {
      *offset = address[tid] + predicate[tid];
    }
    else
      *offset = 0;
  }

}

__global__ void predicate_zero_or_one_at_position( bool is_one,
                                                   unsigned int position,
                                                   unsigned int *const values,
                                                   unsigned int *output_predicate,
                                                   unsigned int num_items)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if ( tid < num_items ) {
    output_predicate[tid] = ((is_one && (values[tid] & (1 << position))) ||
                             (!is_one && !(values[tid] & (1 << position)))) ? 1 : 0;
  }
  __syncthreads();
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  size_t num_threads = 1024;
  size_t num_blocks = static_cast<size_t>(ceil(static_cast<float>(numElems)/static_cast<float>(num_threads)));

  unsigned int * d_predicate;
  checkCudaErrors(cudaMalloc(&d_predicate, sizeof(unsigned int) * numElems));

  unsigned int * d_reduce;
  checkCudaErrors(cudaMalloc(&d_reduce, sizeof(unsigned int) * num_blocks));

  unsigned int * d_scan_reduce;
  checkCudaErrors(cudaMalloc(&d_scan_reduce, sizeof(unsigned int) * num_blocks));

  unsigned int * d_scan;
  checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int) * numElems));

  unsigned int * d_offset;
  checkCudaErrors(cudaMalloc(&d_offset, sizeof(unsigned int)));

  bool input_is_input = true;
  for ( unsigned int bit = 0; bit < sizeof(unsigned int) * 8; ++bit )
  {
    unsigned int *const d_input_values = (input_is_input) ? d_inputVals : d_outputVals;
    unsigned int *const d_input_pos = (input_is_input) ? d_inputPos : d_outputPos;

    unsigned int *const d_output_values = (!input_is_input) ? d_inputVals : d_outputVals;
    unsigned int *const d_output_pos = (!input_is_input) ? d_inputPos : d_outputPos;

    //Bits Zero
    predicate_zero_or_one_at_position<<<num_blocks,num_threads>>>(false, bit, d_input_values, d_predicate, numElems);
    exclusive_scan_add_reduce<<<num_blocks, num_threads, num_threads * sizeof(unsigned int)>>>(d_predicate, d_scan, d_reduce, numElems);
    exclusive_scan_add<<<1, num_threads, num_threads* sizeof(unsigned int)>>>(d_reduce, d_scan_reduce, num_blocks);
    exclusive_scan_add_merge<<<num_blocks, num_threads>>>(d_scan, d_scan_reduce, numElems);
    sort_update_position<<<num_blocks, num_threads>>>(d_predicate, d_scan, d_input_values, d_input_pos, d_output_values, d_output_pos, d_offset, numElems, false);

    //Bits One
    predicate_zero_or_one_at_position<<<num_blocks,num_threads>>>(true, bit, d_input_values, d_predicate, numElems);
    exclusive_scan_add_reduce<<<num_blocks, num_threads, num_threads * sizeof(unsigned int)>>>(d_predicate, d_scan, d_reduce, numElems);
    exclusive_scan_add<<<1, num_threads, num_threads* sizeof(unsigned int)>>>(d_reduce, d_scan_reduce, num_blocks);
    exclusive_scan_add_merge<<<num_blocks, num_threads>>>(d_scan, d_scan_reduce, numElems);
    sort_update_position<<<num_blocks, num_threads>>>(d_predicate, d_scan, d_input_values, d_input_pos, d_output_values, d_output_pos, d_offset, numElems, true);

    input_is_input = !input_is_input;
  }

  if ( input_is_input ) {
    cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice);
  }

  checkCudaErrors(cudaFree(d_predicate));
  checkCudaErrors(cudaFree(d_reduce));
  checkCudaErrors(cudaFree(d_scan_reduce));
  checkCudaErrors(cudaFree(d_scan));
  checkCudaErrors(cudaFree(d_offset));


}
