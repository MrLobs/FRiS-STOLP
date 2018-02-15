# FRiS-STOLP

## Summary

This repository contains a C++ implementation and a Python wrapper for FRiS-STOLP classifier.

The algorithm's main idea is to find a subset of representatives in each class of the training set. These representatives are then used to approximate the borderline between the classes using the function of rival similarity (FRiS).

[More info here.](https://link.springer.com/article/10.1134%2FS105466180801001X)

## Dependencies

   * Boost.uBLAS
   * Boost.Python
   
## Implemented
  
  * Binary classifier
  * Memoized and non-memoized distance calculation
  * Python wrapper
  * sklearn adapter without parameter importance estimation
  
## To do

  * Full sklearn compatibility
  * Automatic installation
