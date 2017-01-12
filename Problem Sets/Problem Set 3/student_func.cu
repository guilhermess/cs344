/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include "utils.h"

__global__ void reduce_min( const float * const values, size_t array_size, float *reduced)
{
  extern __shared__ float smem[];
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if ( id < array_size )
    smem[tid] = values[id];
  __syncthreads();

  for ( unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if ( tid < s && id+s < array_size )
      smem[tid] = (smem[tid] - smem[tid+s] > -1e-15) ? smem[tid+s] : smem[tid];
    else if ( tid < s )
      smem[tid] = smem[tid];
    __syncthreads();
  }

  if ( tid == 0 )
    reduced[blockIdx.x] = smem[0];
}

__global__ void reduce_max( const float * const values, size_t size, float *reduced)
{
  extern __shared__ float smem[];
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if ( id < size )
    smem[tid] = values[id];
  __syncthreads();

  for ( unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if ( tid < s && id+s < size )
      smem[tid] = (smem[tid] - smem[tid+s] > 1e-15) ? smem[tid] : smem[tid+s];
    else if ( tid < s )
      smem[tid] = smem[tid];
    __syncthreads();
  }

  if ( tid == 0 )
    reduced[blockIdx.x] = smem[0];
}

__global__ void histogram( const float *values, size_t size, unsigned int *bins, unsigned int num_bins, float min_val, float range )
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if ( id < num_bins )
    bins[id] = 0;
  __syncthreads();

  if ( id < size ) {
    unsigned int bin = static_cast<unsigned int>((values[id] - min_val) / range * num_bins);
    bin = (num_bins - 1 < bin ) ? num_bins - 1 : bin;
    atomicAdd(&bins[bin], 1);
  }
}

__global__ void exclusive_scan_add( const unsigned int * const values, unsigned int size, unsigned int * const output )
{
  extern __shared__ unsigned int exclusive_scan_smem[];
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  if ( id < size )
    exclusive_scan_smem[tid] = values[id];
  __syncthreads();

  //Reduce
  for ( unsigned int s = 1; s < size; s <<= 1 )
  {
    if ( (tid+1) % s == 0 && ((tid+1)/s) % 2 == 0)
      exclusive_scan_smem[tid] = exclusive_scan_smem[tid] + exclusive_scan_smem[tid-s];
    __syncthreads();
  }

  if (tid == size-1)
    exclusive_scan_smem[tid] = 0;
  __syncthreads();

  //Downsweep
  for ( unsigned int s = size; s >= 1; s >>= 1 )
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
    output[tid] = exclusive_scan_smem[tid];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  size_t numRows,
                                  size_t numCols,
                                  size_t numBins)
{
  size_t luminance_size = numRows * numCols;
  size_t num_threads = 1024;
  size_t num_blocks = static_cast<size_t>(ceil(static_cast<float>(luminance_size)/static_cast<float>(num_threads)));

  float *d_partial_reduce;
  checkCudaErrors(cudaMalloc(&d_partial_reduce, sizeof(float) * num_blocks));

  float *d_reduce_output;
  checkCudaErrors(cudaMalloc(&d_reduce_output, sizeof(float)));

  reduce_min<<<num_blocks,num_threads, num_threads* sizeof(float)>>>(d_logLuminance, luminance_size, d_partial_reduce);
  reduce_min<<<1,num_blocks, num_blocks*sizeof(float)>>>(d_partial_reduce, num_blocks, d_reduce_output);
  checkCudaErrors(cudaMemcpy(&min_logLum, d_reduce_output, sizeof(float), cudaMemcpyDeviceToHost));

  reduce_max<<<num_blocks,num_threads, num_threads* sizeof(float)>>>(d_logLuminance, luminance_size, d_partial_reduce);
  reduce_max<<<1,num_blocks, num_blocks*sizeof(float)>>>(d_partial_reduce, num_blocks, d_reduce_output);
  checkCudaErrors(cudaMemcpy(&max_logLum, d_reduce_output, sizeof(float), cudaMemcpyDeviceToHost));

  float range = max_logLum - min_logLum;

  unsigned int *d_bins;
  checkCudaErrors(cudaMalloc(&d_bins, numBins*sizeof(unsigned int)));
  histogram<<<num_blocks, num_threads>>>(d_logLuminance, luminance_size, d_bins, numBins, min_logLum, range);
  exclusive_scan_add<<<1, numBins, numBins * sizeof( unsigned int)>>>(d_bins, numBins, d_cdf);
}
