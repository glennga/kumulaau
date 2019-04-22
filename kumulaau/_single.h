#include <pthread.h>
#include <stdlib.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// Comment out to not see the debug messages.
//#define _DEBUGGING_POPULATION_ENABLED_

#define MAX_NUM_THREADS 20

// MAX and MIN aren't defined in the standard library. O:
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/**
 * Our population parameters, equivalent to "BaseParameters" in "population.py". Note that if any changes are made
 * to the quantity or type of parameters here, they MUST be changed in "population.py" as well.
 */
typedef struct PopulationParametersStruct {
    int n; ///< Population size, used for determining the number of generations between events.
    float f; ///< Scaling factor for the total mutation rate. Smaller = shorter time to coalescence.
    float c; ///< Constant bias for the upward mutation rate.
    float d; ///< Linear bias for the downward mutation rate.
    int kappa; ///< Lower bound of repeat lengths.
    int omega; ///< Upper bound of repeat lengths.
} PopulationParameters;

/**
 * The population and the associated parameters we are evolving with. Note that if any changes are made
 * to the quantity or type of parameters here, they MUST be changed in "population.py" as well.
 */
typedef struct CoalescentStruct {
    PopulationParameters theta; ///< Mutation model parameters.
    int *coalescent_tree; ///< Pointer to our tree, stored as an array.
    int offset; ///< We generalize to include 1+ ancestors. Determine the offset for array representation of tree.
    gsl_rng *r; ///< Pointer to our RNG. This is preserved across the trace and evolve steps.
} PopulationTree;

/**
 * Triangle number generator. Given 'a', return a choose 2.
 *
 * @param a: Which triangle number to return.
 * @return: The a'th triangle number.
 */
int _triangle (int a) { return (int) (a * (a + 1) / 2.0); }

/**
 * Round a floating point number.
 *
 * @param a: The number to round.
 * @return: The rounded number.
 */
int _round_num (int a) { return (a < 0) ? (int) (a - 0.5) : (int) (a + 0.5); }

/**
 * Signature for a mutation function. This is meant to separate generation vs. coalescent event based evolution.
 */
typedef int (*_mutate_f) (int, int, float, float, int, int, const gsl_rng *);

/**
 * Given a repeat length 'ell', we mutate this repeat length up or down dependent on our parameters c (upward
 * constant bias) and d (downward linear bias). The mutation model gives us the following focal bias:
 *
 * \hat{L} = \frac{-c}{-d}.
 *
 * We repeat this for t generations. This could be done in parallel, but is NOT. ): Not of high importance right now.
 *
 * @param t: The number of generations to mutate for.
 * @param ell: The current repeat length to mutate.
 * @param c: Constant bias for the upward mutation rate.
 * @param d: Linear bias for the downward mutation rate.
 * @param kappa: Lower bound of our repeat length space. If 'ell = kappa', we do not mutate.
 * @param omega: Upper bound of our repeat length space.
 * @param r: GSL random number generator.
 * @return: A mutated repeat length.
 */
int _mutate_generation (int t, int ell, float c, float d, int kappa, int omega, const gsl_rng *r) {
    for (int k = 0; k < t; k++) {
        // Compute our upward mutation rate. We are bounded by omega.
        ell = (gsl_ran_flat(r, 0, 1) < c) ? MIN(omega, ell + 1) : ell;

        // Compute our downward mutation rate. We are bounded by kappa.
        ell = (gsl_ran_flat(r, 0, 1) < ell * d) ? MAX(kappa, ell - 1) : ell;
    }

    // Return the result of mutating over several generations.
    return ell;
}

/**
 * Given a repeat length 'ell', we mutate this repeat length up or down dependent on our parameters c (upward
 * constant bias) and d (downward linear bias). The mutation model gives us the following focal bias:
 *
 * \hat{L} = \frac{-c}{-d}.
 *
 * As opposed to the "_mutate_generation" function, we draw the number of upward mutations and downward mutations,
 * then return the sum. We perform this in constant time as opposed to $\Theta (t)$.
 *
 * @param t: The number of generations to mutate for.
 * @param ell: The current repeat length to mutate.
 * @param c: Constant bias for the upward mutation rate.
 * @param d: Linear bias for the downward mutation rate.
 * @param kappa: Lower bound of our repeat length space. If 'ell = kappa', we do not mutate.
 * @param omega: Upper bound of our repeat length space.
 * @param r: GSL random number generator.
 * @return: A mutated repeat length.
 */
int _mutate_draw (int t, int ell, float c, float d, int kappa, int omega, const gsl_rng *r) {
    return MIN((unsigned) omega, MAX((unsigned) kappa, ell + gsl_ran_poisson(r, t * c) -
                                                       gsl_ran_poisson(r, t * ell * d)));
}

/**
 * Determine an ancestor for the descendant generations. This portion is deterministic, is not meant to simulate a
 * coalescent event. Consequently, the first descendant (i.e. the coalescence event) is not determined.
 *
 * @param coalescent_tree: Pointer to the **entire** coalescent tree.
 * @param tau_0: Start of the ancestor indices to choose from.
 * @param tau_1: Start of the descendant indices to save to.
 * @param k: The specific ancestor to save / descendant to save to.
 */
void _trace_no_branch_nodes (int *coalescent_tree, int tau_0, int tau_1, int k) {
    coalescent_tree[tau_1 + k + 1] = tau_0 + k;
}

/**
 * Create a random evolutionary tree. The result is a 1D array, indexed by _triangle(n). Repeat length determination
 * does not occur at this stage, rather we determine which ancestor belongs to who.
 *
 * @param coalescent_tree: Pointer to the **entire** coalescent tree.
 * @param n: The number of end individuals to choose from after evolution (**diploid**).
 * @param r: RNG to use.
 */
void _trace_tree (int *coalescent_tree, int n, const gsl_rng *r) {
    for (int tau = 0; tau < 2 * n - 1; tau++) {
        int tau_0 = _triangle(tau), tau_1 = _triangle(tau + 1); // Start at 2nd coalescence.

        // We save the indices of our ancestors to our descendants.
        for (int k = 0; k < tau_1 - tau_0; k++) {
            _trace_no_branch_nodes(coalescent_tree, tau_0, tau_1, k);
#ifdef _DEBUGGING_POPULATION_ENABLED_
            // Verify the indices assigned with each non-coalescent event.
            printf("\nCoalescence event [%d], individual [%d] of tree trace, no shuffle: %d",
                   tau + 2, k + 2, coalescent_tree[tau_1 + k + 1]);
#endif
        }

        // Determine the individual whose frequency increases.
        gsl_ran_choose(r, &coalescent_tree[tau_1], 1, &coalescent_tree[tau_1 + 1], tau_1 - tau_0, sizeof(int));

        // Finally, shuffle the individuals in this generation.
        gsl_ran_shuffle(r, &coalescent_tree[tau_1], tau_1 - tau_0 + 1, sizeof(int));
#ifdef _DEBUGGING_POPULATION_ENABLED_
        // Verify that the coalescent event has occurred and that our tree has been randomized.
        for (int k = 0; k < tau_1 - tau_0 + 1; k++) {
            printf("\nCoalescence event [%d], individual [%d] of tree trace, shuffled: %d",
                   tau + 2, k + 1, coalescent_tree[tau_1 + k]);
            if (tau == 2 * n - 1) printf("\n");
        }
#endif
    }
}

/**
 * Given the index to a list of descendants (a subarray inside "coalescence_tree"), a function to mutate between events,
 * and the population structure itself, determine the repeat length of a single individual.
 *
 * @param k: Index of the specific descendant to evolve, currently holding the index of its ancestor.
 * @param descendants: Pointer to an array of descendants, which is a subarray inside p->coalescence_tree.
 * @param mutate: Pointer to the mutation function to use in order to evolve the referenced individual.
 * @param p: Pointer to the PopulationTree object holding the coalescent tree and the associated parameters.
 */
void _evolve_individual (int k, int *descendants, int t_coalescence, _mutate_f mutate, PopulationTree *p) {
    int *descendant_to_evolve = &(p->coalescent_tree[descendants[k]]);

    // Evolve each ancestor according to the average time to coalescence and the scaling factor f.
    *descendant_to_evolve = (*mutate)(t_coalescence, *descendant_to_evolve, p->theta.c, p->theta.d,
                                      p->theta.kappa, p->theta.omega, p->r);

    // Save our descendant state.
    descendants[k] = *descendant_to_evolve;
}

/**
 * We assumed our tree has been traced. Given the population tree structure and a pointer to a mutation function,
 * determine the repeat length of all individuals in our tree. We do so by iterating through each coalescent event.
 *
 * @param p: Pointer to the PopulationTree object holding the coalescent tree and the associated parameters.
 * @param mutate: Pointer to the mutation function to use in order to evolve each individual in our tree.
 */
void _evolve_event (PopulationTree *p, _mutate_f mutate) {
    int descendants_size, expected_time, t_coalescence;
    int *descendants = NULL;

    for (int tau = p->offset; tau < 2 * p->theta.n - 1; tau++) {
        int tau_1 = _triangle(tau + 1), tau_2 = _triangle(tau + 2);

        // We define our descendants for the tau'th coalescent.
        descendants = &(p->coalescent_tree[tau_1]);
        descendants_size = tau_2 - tau_1;

        // Determine time to coalescence. This is exponentially distributed, but the mean stays the same. Scale by f.
        expected_time = (int) (p->theta.f * 2 * p->theta.n / (float) tau_1);
        t_coalescence = MAX(1, _round_num(gsl_ran_exponential(p->r, expected_time)));

        // Iterate through each of the descendants (currently indices) and determine each ancestor.
        for (int k = 0; k < descendants_size; k++) {
            _evolve_individual(k, descendants, t_coalescence, mutate, p);
#ifdef _DEBUGGING_POPULATION_ENABLED_
            // Verify that evolution has occured.
            printf("\nCoalescence event [%d], individual [%d] of evolution: %d",
                   tau + 2, k + 1, descendants[k]);
            if (tau == descendants_size) printf("\n");
#endif
        }
    }
}

/**
 * We assumed our tree has been traced. Given an array of our ancestors and the population tree structure, determine
 * the repeat length of all individuals in our tree. Here, we determine the offset and seed our ancestors before passing
 * the heavy work to "_evolve_event". We also define which mutation function to use here.
 *
 * @param i_0: Pointer to the array holding our seed lengths.
 * @param i_0_size: Size of the array holding our seed lengths.
 * @param p: Pointer to the PopulationTree object holding the coalescent tree and the associated parameters.
 */
void _evolve (int *i_0, int i_0_size, PopulationTree *p) {
    // Determine our offset, and seed our ancestors for the tree.
    p->offset = i_0_size - 1;
    for (int k = 0; k < i_0_size; k++) {
        p->coalescent_tree[k] = i_0[k];
    }
#ifdef _DEBUGGING_POPULATION_ENABLED_
    // Verify that are ancestors have been properly placed.
    printf("\nContents of first |ancestors| in tree array: ");
    for (int k = 0; k < i_0_size; k++) {
        printf("[%d]", p->coalescent_tree[k]);
    }
    printf("\n");
#endif
    // From our common ancestors, descend forward in time and populate our tree with repeat lengths.
    _evolve_event(p, _mutate_draw);
}

/**
 * Release the memory incurred for the coalescent tree array and our RNG. I am positive that I need to do more here,
 * but... I'm too lazy to go searching.
 *
 * @param p: Pointer to the PopulationTree object holding the coalescent tree and the RNG.
 */
void _cleanup (PopulationTree *p) {
    free(p->coalescent_tree);
    gsl_rng_free(p->r);
}
