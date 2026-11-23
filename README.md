# Parallelized Genetic Algorithm using OpenMP(opm) 
run parallel version using the command below :
```
g++ .\genetic_paralleled_using_omp.cpp -o .\parallel.exe -fopenmp
.\parallel.exe
```
or run unparalleled version istead :
```
g++ .\genetic_no_parallelism.cpp -o .\unparalleled.exe -fopenmp
.\unparalleled.exe
```
---
- Note that these results are based on ``population.txt ``which is a sample population (you can generate your own population using the ``rnad_population_generator.py`` python script )