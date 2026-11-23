#include <stdio.h>  //to use the printf function
#include <conio.h>  //to use the getche function
#include <stdlib.h> //to use the rand function
#include <time.h>   //to use the time function
#include <math.h>   //to use the pow function
#include <omp.h>    //to use the openmp functionalities
#include <stdint.h>

#define GENE_COUNT 10
#define POPULATION 1000
#define ITER 500
#define STATIC_POP_FILE "population.txt"
#define INITIAL_FITNESS_FILE "initial_fitness.txt"
#define FINAL_FITNESS_FILE "final_fitness.txt"
#define MUTATION_FAC 5

typedef struct Chrom
{ // creating the chrom structure
    short int bit[GENE_COUNT];
    int fit;
    double prob;
} chrom;

void init_pop(chrom popcurrent[POPULATION]); // defining the functions that we will use
int fitness(chrom chrom);
void select(chrom popcurrent[POPULATION], chrom selected[POPULATION]);
void crossover(chrom popnext[POPULATION]);
void mutation(chrom popnext[POPULATION]);
void sort(chrom popcurrent[POPULATION]);
void print_pop(chrom pop[POPULATION], const char *stage, const char *output_file);

int main()
{
    omp_set_num_threads(16);
    omp_set_dynamic(1);
    chrom popcurrent[POPULATION];
    chrom popnext[POPULATION];
    clock_t elapsed_time;

    init_pop(popcurrent);
    printf("Initial Fitness Statistics:\n");
    printf("---------------\n");
    print_pop(popcurrent, "Before evolution", INITIAL_FITNESS_FILE);

    elapsed_time = clock();
    for (int i = 0; i < ITER; i++)
    {
        select(popcurrent, popnext);
        crossover(popnext);
        mutation(popnext);
#pragma omp parallel for
        for (int j = 0; j < POPULATION; j++)
        {
            popcurrent[j] = popnext[j];
        }
    }
    elapsed_time = clock() - elapsed_time;
    printf("Final Fitness Statistics:\n");
    printf("---------------\n");
    print_pop(popcurrent, "After evolution", FINAL_FITNESS_FILE);

    printf("Population %d and iteration %d \n\n", POPULATION, ITER);

    printf("number of threads : %i\n",omp_get_max_threads());
    printf("Elapsed time is: %f(s)\n",(((double)elapsed_time)/CLOCKS_PER_SEC));
    // printf("omp_get_num_threads() : %i\n",omp_get_num_threads());
    printf("is dynamic : %i\n",omp_get_dynamic());
    
    return 0;
}

void print_pop(chrom pop[POPULATION], const char *stage, const char *output_file)
{ // prints population fitness statistics
    long long int total_fitness = 0;
    int max_fit = 0;
    int min_fit = 101;
#pragma omp parallel for reduction(+ : total_fitness) reduction(max : max_fit) reduction(min : min_fit)
    for (int i = 0; i < POPULATION; i++)
    {
        total_fitness += pop[i].fit;
        max_fit = (pop[i].fit > max_fit) ? pop[i].fit : max_fit;
        min_fit = (pop[i].fit < min_fit) ? pop[i].fit : min_fit;
    }
    printf("Total Fitness is: %lld\n", total_fitness);
    printf("Average Fitness is: %f\n", total_fitness / (double)POPULATION);
    printf("Maximum Fitness is: %d\n", max_fit);
    printf("Minimum Fitness is: %d\n", min_fit);
    printf("\n");
    for (int i = 0; i < 20; i++)
    {
        printf("-");
    }
    printf("\n\n");

    FILE *fitness_file = fopen(output_file, "wt");
    if (fitness_file == NULL)
    {
        printf("Error!\nCouldn't open fitness output file '%s'.\n", output_file);
        exit(EXIT_FAILURE);
    }
    fprintf(fitness_file, "%s\n", stage);
    for (int i = 0; i < POPULATION; i++)
    {
        fprintf(fitness_file, "%d ", pop[i].fit);
        for (int j = 0; j < GENE_COUNT; j++)
        {
            fprintf(fitness_file, "%d", pop[i].bit[j]);
        }
        fprintf(fitness_file, "\n");
    }
    fclose(fitness_file);
}

chrom convert_str_2_chrom(char *chrom_str)
{                 // Accepts a string of 1s & 0s to the length of GENE_COUNT.
    chrom chromo; // Will return the corresponding chromosome.
    for (int i = 0; i < GENE_COUNT; i++)
    {
        if (chrom_str[i] != '0' && chrom_str[i] != '1')
        {
            printf("Error!\nInvalid string as a chromosome.\n");
            exit(EXIT_FAILURE);
        }
        chromo.bit[i] = chrom_str[i] - '0';
    }
    return chromo;
}

void init_pop(chrom popcurrent[POPULATION])
{ // Read Initial population from a static file. STATIC_POP_FILE
    FILE *population;

    population = fopen(STATIC_POP_FILE, "rt");

    if (population == NULL)
    {
        printf("Error!\nCouldn't open population file.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < POPULATION; i++)
    {
        char chrom_str[GENE_COUNT + 2];
        if (fscanf(population, "%s", chrom_str) != 1)
        {
            fclose(population);
            printf("Error!\npopulation.txt contains fewer than %d chromosomes.\n", POPULATION);
            exit(EXIT_FAILURE);
        }
        chrom_str[GENE_COUNT] = 0;
        popcurrent[i] = convert_str_2_chrom(chrom_str);
    }
    fclose(population);
    long long int total_fitness = 0;
#pragma omp parallel for reduction(+ : total_fitness)
    for (int i = 0; i < POPULATION; i++)
    {
        popcurrent[i].fit = fitness(popcurrent[i]);
        total_fitness += popcurrent[i].fit;
    }
#pragma omp parallel for
    for (int i = 0; i < POPULATION; i++)
    {
        popcurrent[i].prob = (total_fitness == 0)
                                 ? 1.0 / POPULATION
                                 : (double)popcurrent[i].fit / total_fitness;
    }
}

int fitness(chrom chrom)
{
    int bit_number = 0;
    int sum = 0;
    while (bit_number < GENE_COUNT)
    {
        sum += chrom.bit[bit_number] * (1 << bit_number);
        bit_number++;
    }
    sum = sum % 101;
    return (sum); // return the value of sum
}

void select(chrom popcurrent[POPULATION], chrom selected[POPULATION])
{ // perfom selection on the curr pop considering prob.
#pragma omp parallel for
    for (int j = 0; j < POPULATION; j++)
    {
        uint32_t seed = (uint32_t)(j + 1) * 1664525u + 1013904223u;
        seed ^= (uint32_t)omp_get_thread_num() * 0x9e3779b9u;
        seed = seed * 1664525u + 1013904223u;
        double random = (double)(seed >> 8) / (double)(1u << 24);
        double cumulative_prob = 0;
        for (int i = 0; i < POPULATION; i++)
        {
            cumulative_prob += popcurrent[i].prob;
            if (random < cumulative_prob)
            {
                selected[j] = popcurrent[i];
                break;
            }
        }
    }
}

void sort(chrom popcurrent[POPULATION])
{ // sort pop by their prob
    chrom temp;
    for (int i = 0; i < POPULATION; i++)
    {
        for (int j = 0; j < POPULATION - 1; j++)
        {
            if (popcurrent[j].fit > popcurrent[j + 1].fit)
            {
                temp = popcurrent[j + 1];
                popcurrent[j + 1] = popcurrent[j];
                popcurrent[j] = temp;
            }
        }
    }
}

void crossover(chrom popnext[POPULATION])
{ // crossover function takes a pointer to array of chromes
    uint32_t random_state = (uint32_t)time(NULL) ^ 0xA5A5A5A5u;
    random_state = random_state * 1664525u + 1013904223u;
    int random = (int)((random_state >> 8) % (GENE_COUNT - 1)) + 1;
    chrom temp_child;
    for (int i = 0; i < random; i++)
    {
        temp_child.bit[i] = popnext[0].bit[i];
    }
    for (int i = random; i < GENE_COUNT; i++)
    {
        temp_child.bit[i] = popnext[POPULATION - 1].bit[i];
    }
    for (int chrom_counter = 0; chrom_counter < POPULATION; chrom_counter++)
    {
        random_state = random_state * 1664525u + 1013904223u;
        random = (int)((random_state >> 8) % (GENE_COUNT - 1)) + 1;
        for (int i = random; i < GENE_COUNT; i++)
        {
            if (chrom_counter + 1 < POPULATION)
            {
                popnext[chrom_counter].bit[i] = popnext[chrom_counter + 1].bit[i];
            }
        }
    }
    popnext[POPULATION - 1] = temp_child; // the last popnext chrom now becomes the new child

    long long int total_fitness = 0;
#pragma omp parallel for reduction(+ : total_fitness)
    for (int i = 0; i < POPULATION; i++)
    {
        popnext[i].fit = fitness(popnext[i]);
        total_fitness += popnext[i].fit;
    }
#pragma omp parallel for
    for (int i = 0; i < POPULATION; i++)
    {
        popnext[i].prob = (total_fitness == 0)
                              ? 1.0 / POPULATION
                              : (double)popnext[i].fit / total_fitness;
    }
}

void mutation(chrom popnext[POPULATION])
{
    uint32_t state = (uint32_t)time(NULL) ^ 0xC001D00Du;
    state = state * 1664525u + 1013904223u;
    int prob = (int)((state >> 8) % 101);
    if (prob <= MUTATION_FAC)
    {
        state = state * 1664525u + 1013904223u;
        int chrom_index = (int)((state >> 8) % POPULATION);
        state = state * 1664525u + 1013904223u;
        int gene = (int)((state >> 8) % GENE_COUNT);
        popnext[chrom_index].bit[gene] = (popnext[chrom_index].bit[gene] == 1) ? 0 : 1;
        popnext[chrom_index].fit = fitness(popnext[chrom_index]);

        long long int total_fitness = 0;
#pragma omp parallel for reduction(+ : total_fitness)
        for (int i = 0; i < POPULATION; i++)
        {
            total_fitness += popnext[i].fit;
        }
#pragma omp parallel for
        for (int i = 0; i < POPULATION; i++)
        {
            popnext[i].prob = (total_fitness == 0)
                                  ? 1.0 / POPULATION
                                  : (double)popnext[i].fit / total_fitness;
        }
    }
}