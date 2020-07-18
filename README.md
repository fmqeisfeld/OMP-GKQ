# OMP-threaded Gauss-Lobatto-Kronrod quadrature 

## Description
The title pretty much tells it: This is an omp-parallelized integration scheme based on the Gauss-Lobatto method with a Kronrod extension.
The method is described in the wonderful numerical recipes collection.
A serial example code can be found in the webnote [here](http://numerical.recipes/webnotes/nr3web4.pdf).
The code compares the performance of the integration of some arbitrary function against GSL's *gsl_integration_qag*.
Note that this rather unoptimzed code performs better than  GSL only if the function to be integrated is computationally expensive (e.g. if it needs to compute another inner integral itself).

## Usage
```bash
gcc ompquad.c -lgsl -lgslcblas -fopenmp -lm
./a.out
