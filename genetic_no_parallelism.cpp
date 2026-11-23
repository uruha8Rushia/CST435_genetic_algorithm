#include <stdio.h>  //to use the printf function
#include <conio.h>  //to use the getche function
#include <stdlib.h> //to use the rand fuFnction
#include <time.h>   //to use the time function
#include <math.h>   //to use the pow function

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
        for (int j = 0; j < POPULATION; j++)
        {
            popcurrent[j] = popnext[j];
        }
    }
    elapsed_time = clock() - elapsed_time;
    printf("Final Fitness Statistics:\n");
    printf("---------------\n");
    print_pop(popcurrent, "After evolution", FINAL_FITNESS_FILE);
    printf("Population %d and iteration %d\n\n", POPULATION, ITER);
    printf("Elapsed time is: %f(s)\n", ((double)elapsed_time) / CLOCKS_PER_SEC);
    return 0;
}

void print_pop(chrom pop[POPULATION], const char *stage, const char *output_file)
{ // prints population fitness statistics
    long long int total_fitness = 0;
    int max_fit = 0;
    int min_fit = 101;
    for (int i = 0; i < POPULATION; i++)
    {
        total_fitness += pop[i].fit;
        if (pop[i].fit > max_fit)
        {
            max_fit = pop[i].fit;
        }
        if (pop[i].fit < min_fit)
        {
            min_fit = pop[i].fit;
        }
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
    fprintf(fitness_file, "Total Fitness: %lld\n", total_fitness);
    fprintf(fitness_file, "Average Fitness: %f\n", total_fitness / (double)POPULATION);
    fprintf(fitness_file, "Maximum Fitness: %d\n", max_fit);
    fprintf(fitness_file, "Minimum Fitness: %d\n", min_fit);
    fprintf(fitness_file, "\n");
    for (int i = 0; i < POPULATION; i++)
    {
        for (int j = 0; j < GENE_COUNT; j++)
        {
            fprintf(fitness_file, "%d", pop[i].bit[j]);
        }
        fprintf(fitness_file, " %d\n", pop[i].fit);
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
    for (int i = 0; i < POPULATION; i++)
    {
        popcurrent[i].fit = fitness(popcurrent[i]);
        total_fitness += popcurrent[i].fit;
    }
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
        sum += (chrom.bit[bit_number] * ((int)pow(2, bit_number)));
        bit_number++;
    }
    sum = sum % 101;
    return (sum); // return the value of sum
}

void select(chrom popcurrent[POPULATION], chrom selected[POPULATION])
{ // perfom selection on the curr pop considering prob.
    sort(popcurrent);
    srand((unsigned)time(NULL));
    for (int j = 0; j < POPULATION; j++)
    {
        double random = (double)rand() / RAND_MAX;
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
    int random;
    random = rand();
    chrom temp_child = {};
    random = ((random % (GENE_COUNT - 1)) + 1); // random cross over for first child (child of first popnext chrom and the last one)
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
        random = rand();
        random = ((random % (GENE_COUNT - 1)) + 1);
        if (chrom_counter + 1 >= POPULATION)
        {
            continue;
        }
        for (int i = random; i < GENE_COUNT; i++)
        {
            popnext[chrom_counter].bit[i] = popnext[chrom_counter + 1].bit[i];
        }
    }
    popnext[POPULATION - 1] = temp_child; // the last popnext chrom now becomes the new child

    long long int total_fitness = 0;
    for (int i = 0; i < POPULATION; i++)
    {
        popnext[i].fit = fitness(popnext[i]);
        total_fitness += popnext[i].fit;
    }
    for (int i = 0; i < POPULATION; i++)
    {
        popnext[i].prob = (total_fitness == 0)
                              ? 1.0 / POPULATION
                              : (double)popnext[i].fit / total_fitness;
    }
}

void mutation(chrom popnext[POPULATION])
{
    srand((unsigned)time(NULL));
    int prob = rand() % 101;
    if (prob <= MUTATION_FAC)
    {
        int chrom = rand() % (POPULATION);
        int gene = rand() % (GENE_COUNT);
        popnext[chrom].bit[gene] = (popnext[chrom].bit[gene] == 1) ? 0 : 1;
        popnext[chrom].fit = fitness(popnext[chrom]);

        long long int total_fitness = 0;
        for (int i = 0; i < POPULATION; i++)
        {
            total_fitness += popnext[i].fit;
        }
        for (int i = 0; i < POPULATION; i++)
        {
            popnext[i].prob = (total_fitness == 0)
                                  ? 1.0 / POPULATION
                                  : (double)popnext[i].fit / total_fitness;
        }
    }
}