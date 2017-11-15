/*
   159735 Parallel Programming Assignment 5

   To compile: nvcc -o hyperSpace hyperSpace.cu
   To run: ./hyperSpace [nTrails]
   nTrails is 20 by default
   For example:
   "./hyperSpace 50" will generate a hyper sphere for 50 times,
   and count the number of integer coordinate points inside every sphere,
   by both cuda algorithm and sequential algorithm.

 */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <cuda.h>
#include <vector>
#include <ctime>
#include <stdio.h>

using namespace std;

// Output latest cuda error
#define cudaCheckError() {                                                                 \
 cudaError_t e=cudaGetLastError();                                                         \
 if(e!=cudaSuccess) {                                                                      \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                                         \
}


/*
 *  In my test, if the dimensions number is larger than 7, the time consuming
 *  of sequential algorithm will be very large, so set maximum of dimension 8.
 */
const long MAXDIM = 8;
const double RMIN = 2.0;
const double RMAX = 8.0;

// Used to calculate the total used time for different functions
double diffclock(clock_t clock1,clock_t clock2)
{
  double diffticks = clock1 - clock2;
  double diffms = (diffticks * 1000) / CLOCKS_PER_SEC;
  return diffms; // Time difference in milliseconds
}

/* Evaluate n**k where both are long integers */
long powlong(long n, long k)
{
  long p = 1;
  for (long i = 0; i < k; ++i) p *= n;
  return p;
}

/* Query device about threads limitation */
bool getCudaDeviceInfo(int &maxThreadsPerBlock, int maxDimensionPerGrid[])
{
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp, 0); // In our environment, there is only one GPU.
   maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
   // We'd like to use two-dimensions because it is enough to resolve this assignment
   maxDimensionPerGrid[0] = deviceProp.maxGridSize[0];
   maxDimensionPerGrid[1] = deviceProp.maxGridSize[1];
   return true;
}


/*----------------------------CUDA Version--------------------------------*/

/*
 * Cuda function runs on GPU, convert a number into a new based number,
 * check if the related point is in the hyper sphere or not
 */
__global__ void compute(long ntotal, long base, long ndim, double rsquare, long halfb, unsigned long long int *count)
{

   int row = blockIdx.x * blockDim.x + threadIdx.x;
   int col = blockIdx.y * blockDim.y + threadIdx.y;
   int n = col + row * gridDim.y*blockDim.y;
   //int n = blockIdx.x*blockDim.x + threadIdx.x*threadDim.x + ;
   if(n >= ntotal)
   {
      return;
   }
   // Convert decimal number to a number in a new base
   long index[MAXDIM];
   for (long i = 0; i < MAXDIM; ++i) index[i] = 0;
   long idx = 0;
   while (n != 0) {
      long rem = n % base;
      n = n / base;
      index[idx] = rem;
      ++idx;
   }
   // Check if the point is in the hypersphere
   double rtestsq = 0;
   for (long k = 0; k < ndim; ++k) {
      double xk = index[k] - halfb;
      rtestsq += xk * xk;
   }
   if (rtestsq < rsquare )
      atomicAdd((unsigned long long int *)count, 1);
}


/*
   CUDA version of the algorithm. Given:
   ndim   -> number of dimensions of the hypersphere
   radius -> radius of the hypersphere
   count the number of integer points that lie wholly within the
   hypersphere, assuming it is centred on the origin.
*/

long count_in_cuda(long ndim, double radius)
{

  const long halfb = static_cast<long>(floor(radius));
  const long base = 2 * halfb + 1;
  const double rsquare = radius * radius;

  // This is the total number of points we will need to test.
  const long ntotal = powlong(base, ndim);

  long *h_count = new long[1];
  unsigned long long int *d_count;
  cudaMalloc(&d_count, sizeof(unsigned long long int));
  cudaMemset(d_count, 0, sizeof(unsigned long long int));

  /*
   * In the assignment, the threads we need to create may be very large,
   * use two dimensions for blocks in case the blocks are not enough.
   */
  int threadsPerBlock;
  int maxDimensionPerGrid[2];
  getCudaDeviceInfo(threadsPerBlock, maxDimensionPerGrid);
  int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;
  dim3 numBlocks(blocksPerGrid, 1);
  // If one dimensions is not enough, use two dimensions. In this assignment, two dimensions is enough.
  if(blocksPerGrid > maxDimensionPerGrid[0])
  {
      int dimensionPerGridX = (blocksPerGrid +  maxDimensionPerGrid[1] - 1) /  maxDimensionPerGrid[1];
      int dimensionPerGridY = maxDimensionPerGrid[1];
      numBlocks = dim3(dimensionPerGridX, dimensionPerGridY);
  }
  compute<<<numBlocks, threadsPerBlock>>>(ntotal, base, ndim, rsquare, halfb, d_count);
  // Check if there is any error reported by CUDA, for debug.
  cudaCheckError();
  cudaMemcpy(h_count, d_count, sizeof(long), cudaMemcpyDeviceToHost);
  unsigned long long int count = *h_count;

  // Release memory
  delete[] h_count;
  cudaFree(d_count);
  return count;
}


/*-----------------------------Sequential Version-------------------------------*/

/* Convert a decimal number into another base system - the individual
   digits in the new base are stored in the index array. */
void convert(long num, long base, std::vector<long>& index)
{
   const long ndim = index.size();
   for (long i = 0; i < ndim; ++i)
      index[i] = 0;
   long idx = 0;
   while (num != 0) {
      long rem = num % base;
      num = num / base;
      index[idx] = rem;
      idx ++;
   }
}

/*
   Sequential version of the algorithm. Given:
   ndim   -> number of dimensions of the hypersphere
   radius -> radius of the hypersphere
   count the number of integer points that lie wholly within the
   hypersphere, assuming it is centred on the origin.
*/
long count_in_sequential(long ndim, double radius)
{
   const long halfb = static_cast<long>(floor(radius));
   const long base = 2 * halfb + 1;
   const double rsquare = radius * radius;

   // This is the total number of points we will need to check.
   const long ntotal = powlong(base, ndim);
   long count = 0;
   std::vector<long> index(ndim, 0);

   // Loop over the total number of points. For each visit of the loop,
   // we covert n to its equivalent in a number system of given "base".
   for (long n = 0; n < ntotal; ++n) {
      convert(n, base, index);
      double rtestsq = 0;
      for (long k = 0; k < ndim; ++k) {
         double xk = index[k] - halfb;
         rtestsq += xk * xk;
      }
      if (rtestsq < rsquare) ++count;
   }
   return count ;
}

/*-----------------------------Main Fuction-------------------------------*/

int main(int argc, char* argv[])
{
   int ntrials = 20;
   if(argc >=2)
      ntrials = atoi(argv[1]);
   // Check whether user input is legal
   if(ntrials <= 0)
      ntrials = 20;
   double mscuda = 0.0;
   double msSequential = 0.0;
   for (long n = 0; n < ntrials; ++n) {
      // Get a random value for the hypersphere radius between the two limits
      const double r = drand48() * (RMAX - RMIN) + RMIN;
      // Get a random value for the number of dimensions between 1 and
      // MAXDIM inclusive
      long  nd = lrand48() % (MAXDIM - 1) + 1;
      cout << "### trial: " << n << ", radius: " << r << ", dimensions: " << nd << " ... " << endl;
      clock_t tStartCuda = clock();
      const long numCuda = count_in_cuda(nd, r);
      clock_t tEndCuda = clock();
      mscuda += diffclock(tEndCuda, tStartCuda);
      clock_t tStartSequential = clock();
      const long numSequential = count_in_sequential(nd, r);
      clock_t tEndSequential = clock();
      msSequential += diffclock(tEndSequential, tStartSequential);
      cout << "CUDA result: " << numCuda
           << " ==> sequential result:" << numSequential << endl;
  }
  cout << "Totally "<< ntrials << " trials," << "Cuda version used: " << mscuda
       << " ms, cpu version costs " << msSequential << " ms" << endl;
}

