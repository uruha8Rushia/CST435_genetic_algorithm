# CST435 Assignment 1: Parallel Genetic Algorithm with OpenMP

A genetic algorithm (GA), implemented sequentially and in parallel with OpenMP,
together with a script that measures speedup and efficiency.

## Credit

This project builds on
[genetic_algorithm_implementation_in_c](https://github.com/smhhoseinee/genetic_algorithm_implementation_in_c)
by Mohamad Hosein Hoseinee, Vahid Ramezani and Soroosh (2022). We reuse it with our
lecturer's permission; the original authors' commits are kept in this repository's history.

Our group's changes:
- Both versions now run the same algorithm with the same random numbers, so they print
  identical results and the measured speedup comes from parallelism alone. (Before, the
  sequential version ran an O(n²) bubble sort every generation that the parallel version
  had removed, which inflated the speedup.)
- A counter-based random-number generator: each chromosome's random numbers depend on the
  seed, generation and index, never on which thread processes it. Threads never share
  random-number state.
- Crossover writes children into a separate array, so every child can be made in parallel.
- Population size, generations, thread count, seed and input file are command-line options.
- Wall-clock timing (`std::chrono::steady_clock`) in both versions, covering the evolution
  loop only.
- The population is allocated on the heap, so large populations work.
- Fitness statistics are printed and also written to `initial_fitness.txt` and `final_fitness.txt`.
- `bench.py` runs the scalability experiments.
- Removed the Windows-only `conio.h` include, so the code builds on Windows, Linux and macOS.

## Build

```
g++ -O2 genetic_no_parallelism.cpp -o sequential
g++ -O2 -fopenmp genetic_paralleled_using_omp.cpp -o parallel
```
On Windows (MinGW) add `.exe` to the output names.

## Run

```
sequential [-p population] [-i iterations] [-s seed] [-f population_file] [-q]
parallel   [-p population] [-i iterations] [-t threads] [-s seed] [-f population_file] [-q]
```
| Option | Meaning | Default |
|---|---|---|
| `-p` | population size (chromosomes read from the input file) | 1000 |
| `-i` | number of generations | 500 |
| `-t` | OpenMP threads (parallel version only) | number of CPU cores |
| `-s` | random seed | 12345 |
| `-f` | population file | `population.txt` |
| `-q` | don't write the fitness files | off |

## Input format

`population.txt` has one chromosome per line: a string of 10 characters, each `0` or `1`.
The programs read the first `-p` lines. The provided file has 100,000 chromosomes; make a
new one with `python rnad_population_generator.py`.

Gene *i* (the *i*-th character, counting from 0) has weight 2^i, and
fitness = (sum of the weights of the 1-genes) mod 101, which gives a value from 0 to 100.

## Output format

The console shows total, average, maximum and minimum fitness before and after evolution,
then the parameters, thread count and elapsed time in seconds, and finally one
machine-readable line:
```
RESULT,<version>,<population>,<iterations>,<threads>,<seconds>,<final total fitness>
```
`initial_fitness.txt` and `final_fitness.txt` contain the same statistics followed by every
chromosome and its fitness.

## Algorithm (per generation)

1. **Selection** (roulette wheel): each of the n picks scans the cumulative probabilities,
   O(n²). The picks are independent, so they are split between threads. This is where most
   of the time goes.
2. **Crossover** (one-point): child j = genes before a random cut from parent j + genes after
   it from parent j+1 (wrapping at the end). Each child is independent, so this runs in parallel.
3. **Fitness and probabilities**: recomputed in parallel, with an OpenMP reduction for the
   total fitness.
4. **Mutation**: with 5% chance, one random gene of one random chromosome flips. This step is
   O(1), so it stays sequential.

## Scalability experiments

```
python bench.py --seq ./sequential --par ./parallel --pops 1000 2000 4000 --threads 1 2 4 8 16
```
Each configuration runs 3 times (`--runs`) and the median is used. The script checks that
every parallel result matches the sequential one, then writes `results.csv`, `results.md`
and, if matplotlib is installed, `speedup.png` and `efficiency.png`.
Speedup S(p) = T_sequential / T_parallel(p); efficiency E(p) = S(p) / p.

For reliable timings, close other programs, keep the laptop plugged in, and don't use more
threads than the CPU has logical cores.
