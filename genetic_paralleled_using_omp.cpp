// =============================================================================
// CST435 Assignment 1 - Genetic Algorithm, PARALLEL (OpenMP) version
//
// Based on "genetic_algorithm_implementation_in_c" by Mohamad Hosein Hoseinee,
// Vahid Ramezani and Soroosh (2022):
//   https://github.com/smhhoseinee/genetic_algorithm_implementation_in_c
// Reused with the lecturer's permission and modified by our group (see README).
//
// This file and genetic_no_parallelism.cpp run the SAME algorithm with the
// SAME random numbers. The only differences are the OpenMP pragmas and the
// thread-count option, so the speedup measured between them comes from
// parallelism alone. Given the same input and seed, both print identical results.
//
// Build:  g++ -O2 -fopenmp genetic_paralleled_using_omp.cpp -o parallel
// Run:    parallel [-p population] [-i iterations] [-t threads] [-s seed] [-f file] [-q]
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <chrono>
#include <omp.h>

#define GENE_COUNT 10                 // bits per chromosome
#define FITNESS_MOD 101               // fitness = (binary value of the bits) mod 101
#define MUTATION_FAC 5                // % chance per generation that one gene flips
#define INITIAL_FITNESS_FILE "initial_fitness.txt"
#define FINAL_FITNESS_FILE "final_fitness.txt"

// Random-number streams, so selection, crossover and mutation never share numbers
enum { STREAM_SELECT = 1, STREAM_CROSS = 2, STREAM_MUTATE = 3 };

typedef struct Chrom
{
    unsigned char bit[GENE_COUNT]; // bit[i] has weight 2^i
    int fit;                       // fitness value, 0..100
    double prob;                   // selection probability = fit / total fitness
} chrom;

typedef struct Params
{
    int population;
    int iterations;
    int threads;
    uint64_t seed;
    const char *input_file;
    int quiet; // 1 = don't write the fitness files (used by the benchmark script)
} params;

static uint64_t g_seed = 12345;

// -------------------------------------------------------------------- random --
// Counter-based generator: the random number for (generation, index, stream) is
// a hash of those values and the seed. Each chromosome gets its own number no
// matter which thread handles it, so results do not depend on the thread count,
// and threads never share random-number state.
static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static inline uint64_t rand_u64(int gen, int index, int stream)
{
    uint64_t key = ((uint64_t)(uint32_t)gen << 32) ^ ((uint64_t)stream << 28) ^ (uint32_t)index;
    return splitmix64(g_seed ^ splitmix64(key));
}

static inline double rand01(int gen, int index, int stream) // uniform in [0, 1)
{
    return (double)(rand_u64(gen, index, stream) >> 11) * (1.0 / 9007199254740992.0);
}

static inline int rand_int(int gen, int index, int stream, int n) // uniform in [0, n)
{
    return (int)(rand_u64(gen, index, stream) % (uint64_t)n);
}

// ------------------------------------------------------------------- fitness --
int fitness(const chrom *c)
{
    int sum = 0;
    for (int b = 0; b < GENE_COUNT; b++)
    {
        sum += c->bit[b] << b; // bit b is worth 2^b
    }
    return sum % FITNESS_MOD;
}

// Recompute every chromosome's selection probability. Returns total fitness.
long long update_probabilities(chrom *pop, int n)
{
    long long total_fitness = 0;
#pragma omp parallel for reduction(+ : total_fitness)
    for (int i = 0; i < n; i++)
    {
        total_fitness += pop[i].fit;
    }
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        pop[i].prob = (total_fitness == 0) ? 1.0 / n : (double)pop[i].fit / total_fitness;
    }
    return total_fitness;
}

// --------------------------------------------------------------------- input --
// Reads the first n chromosomes from the input file: one string of 0s and 1s
// per line; only the first GENE_COUNT characters are used.
void init_pop(chrom *pop, int n, const char *file_name)
{
    FILE *file = fopen(file_name, "rt");
    if (file == NULL)
    {
        printf("Error!\nCouldn't open population file '%s'.\n", file_name);
        exit(EXIT_FAILURE);
    }
    char line[64];
    for (int i = 0; i < n; i++)
    {
        if (fscanf(file, "%63s", line) != 1)
        {
            fclose(file);
            printf("Error!\n'%s' contains fewer than %d chromosomes.\n", file_name, n);
            exit(EXIT_FAILURE);
        }
        if ((int)strlen(line) < GENE_COUNT)
        {
            fclose(file);
            printf("Error!\nLine %d is shorter than %d genes.\n", i + 1, GENE_COUNT);
            exit(EXIT_FAILURE);
        }
        for (int b = 0; b < GENE_COUNT; b++)
        {
            if (line[b] != '0' && line[b] != '1')
            {
                fclose(file);
                printf("Error!\nLine %d is not a string of 0s and 1s.\n", i + 1);
                exit(EXIT_FAILURE);
            }
            pop[i].bit[b] = (unsigned char)(line[b] - '0');
        }
    }
    fclose(file);

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        pop[i].fit = fitness(&pop[i]);
    }
    update_probabilities(pop, n);
}

// ---------------------------------------------------------- genetic operators --
// Roulette-wheel selection: chromosome i is picked with probability prob[i].
// Each of the n picks scans the cumulative probabilities, so this step is
// O(n^2) and is where most of the running time goes. The n picks are
// independent, so they are shared out between the threads.
void select_pop(const chrom *current, chrom *selected, int n, int gen)
{
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++)
    {
        double r = rand01(gen, j, STREAM_SELECT);
        double cumulative = 0.0;
        int pick = n - 1; // fallback in case rounding leaves r >= the total
        for (int i = 0; i < n; i++)
        {
            cumulative += current[i].prob;
            if (r < cumulative)
            {
                pick = i;
                break;
            }
        }
        selected[j] = current[pick];
    }
}

// One-point crossover: child j takes the genes before a random cut point from
// parent j and the genes after it from parent j+1 (wrapping around at the end).
// Parents are read from 'selected' and children written to 'next', so every
// child can be made independently.
void crossover(const chrom *selected, chrom *next, int n, int gen)
{
#pragma omp parallel for
    for (int j = 0; j < n; j++)
    {
        const chrom *a = &selected[j];
        const chrom *b = &selected[(j + 1) % n];
        int cut = 1 + rand_int(gen, j, STREAM_CROSS, GENE_COUNT - 1); // 1..GENE_COUNT-1
        for (int g = 0; g < GENE_COUNT; g++)
        {
            next[j].bit[g] = (g < cut) ? a->bit[g] : b->bit[g];
        }
        next[j].fit = fitness(&next[j]);
    }
}

// With MUTATION_FAC % chance, flip one random gene of one random chromosome.
// Returns 1 if a mutation happened. O(1), so it stays sequential in both versions.
int mutation(chrom *pop, int n, int gen)
{
    if (rand_int(gen, 0, STREAM_MUTATE, 100) >= MUTATION_FAC)
    {
        return 0;
    }
    int c = rand_int(gen, 1, STREAM_MUTATE, n);
    int g = rand_int(gen, 2, STREAM_MUTATE, GENE_COUNT);
    pop[c].bit[g] ^= 1;
    pop[c].fit = fitness(&pop[c]);
    return 1;
}

// -------------------------------------------------------------------- output --
void print_pop(const chrom *pop, int n, const char *stage, const char *output_file, int quiet)
{
    long long total_fitness = 0;
    int max_fit = 0;
    int min_fit = FITNESS_MOD;
#pragma omp parallel for reduction(+ : total_fitness) reduction(max : max_fit) reduction(min : min_fit)
    for (int i = 0; i < n; i++)
    {
        total_fitness += pop[i].fit;
        if (pop[i].fit > max_fit)
            max_fit = pop[i].fit;
        if (pop[i].fit < min_fit)
            min_fit = pop[i].fit;
    }
    printf("%s\n", stage);
    printf("--------------------\n");
    printf("Total Fitness is:   %lld\n", total_fitness);
    printf("Average Fitness is: %f\n", total_fitness / (double)n);
    printf("Maximum Fitness is: %d\n", max_fit);
    printf("Minimum Fitness is: %d\n\n", min_fit);

    if (quiet)
        return;
    FILE *file = fopen(output_file, "wt");
    if (file == NULL)
    {
        printf("Error!\nCouldn't open fitness output file '%s'.\n", output_file);
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%s\n", stage);
    fprintf(file, "Total Fitness: %lld\n", total_fitness);
    fprintf(file, "Average Fitness: %f\n", total_fitness / (double)n);
    fprintf(file, "Maximum Fitness: %d\n", max_fit);
    fprintf(file, "Minimum Fitness: %d\n\n", min_fit);
    for (int i = 0; i < n; i++)
    {
        for (int b = 0; b < GENE_COUNT; b++)
            fprintf(file, "%d", pop[i].bit[b]);
        fprintf(file, " %d\n", pop[i].fit);
    }
    fclose(file);
}

// --------------------------------------------------------------- arguments --
static void usage(const char *prog)
{
    printf("Usage: %s [-p population] [-i iterations] [-t threads] [-s seed] [-f population_file] [-q]\n", prog);
    exit(EXIT_FAILURE);
}

params parse_args(int argc, char **argv)
{
    params p = {1000, 500, omp_get_num_procs(), 12345, "population.txt", 0};
    for (int a = 1; a < argc; a++)
    {
        if (strcmp(argv[a], "-q") == 0)
        {
            p.quiet = 1;
            continue;
        }
        if (a + 1 >= argc)
            usage(argv[0]);
        if (strcmp(argv[a], "-p") == 0)
            p.population = atoi(argv[++a]);
        else if (strcmp(argv[a], "-i") == 0)
            p.iterations = atoi(argv[++a]);
        else if (strcmp(argv[a], "-t") == 0)
            p.threads = atoi(argv[++a]);
        else if (strcmp(argv[a], "-s") == 0)
            p.seed = strtoull(argv[++a], NULL, 10);
        else if (strcmp(argv[a], "-f") == 0)
            p.input_file = argv[++a];
        else
            usage(argv[0]);
    }
    if (p.population < 2 || p.iterations < 1 || p.threads < 1)
        usage(argv[0]);
    return p;
}

// ---------------------------------------------------------------------- main --
int main(int argc, char **argv)
{
    params p = parse_args(argc, argv);
    g_seed = p.seed;
    omp_set_dynamic(0); // use exactly the requested number of threads
    omp_set_num_threads(p.threads);

    // Heap allocation, so large populations don't overflow the stack
    chrom *pop = (chrom *)malloc(sizeof(chrom) * p.population);
    chrom *selected = (chrom *)malloc(sizeof(chrom) * p.population);
    if (pop == NULL || selected == NULL)
    {
        printf("Error!\nNot enough memory for %d chromosomes.\n", p.population);
        return EXIT_FAILURE;
    }

    init_pop(pop, p.population, p.input_file);
    print_pop(pop, p.population, "Before evolution", INITIAL_FITNESS_FILE, p.quiet);

    // Only the evolution loop is timed (file input/output is excluded)
    auto start = std::chrono::steady_clock::now();
    for (int gen = 0; gen < p.iterations; gen++)
    {
        select_pop(pop, selected, p.population, gen);
        crossover(selected, pop, p.population, gen);
        update_probabilities(pop, p.population);
        if (mutation(pop, p.population, gen))
            update_probabilities(pop, p.population);
    }
    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    print_pop(pop, p.population, "After evolution", FINAL_FITNESS_FILE, p.quiet);

    long long total = 0;
    for (int i = 0; i < p.population; i++)
        total += pop[i].fit;
    printf("Version: parallel (OpenMP)\n");
    printf("Population %d, iterations %d, seed %llu\n", p.population, p.iterations,
           (unsigned long long)p.seed);
    printf("Number of threads: %d\n", p.threads);
    printf("Elapsed time is: %f(s)\n", seconds);
    // Machine-readable line for bench.py
    printf("RESULT,parallel,%d,%d,%d,%.6f,%lld\n", p.population, p.iterations, p.threads, seconds,
           total);

    free(pop);
    free(selected);
    return 0;
}
