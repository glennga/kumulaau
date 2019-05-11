#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// MAX and MIN aren't defined in the standard library. O:
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef struct PopulationParametersStruct {
    int n; ///< Population size, used for determining the number of generations between events.
    float f; ///< Scaling factor for the total mutation rate. Smaller = shorter time to coalescence.
    float c; ///< Constant bias for the upward mutation rate.
    float d; ///< Linear bias for the downward mutation rate.
    int kappa; ///< Lower bound of repeat lengths.
    int omega; ///< Upper bound of repeat lengths.
} PopulationParameters;

typedef struct CoalescentStruct {
    PopulationParameters theta; ///< Mutation model parameters.
    int *coalescent_tree; ///< Pointer to our tree, stored as an array.
    int offset; ///< We generalize to include 1+ ancestors. Determine the offset for array representation of tree.
    gsl_rng *r; ///< Pointer to our RNG. This is preserved across the trace and evolve steps.
} PopulationTree;

int _triangle (int a) { return (int) (a * (a + 1) / 2.0); }
int _round_num (int a) { return (a < 0) ? (int) (a - 0.5) : (int) (a + 0.5); }
typedef int (*_mutate_f) (int, int, float, float, int, int, const gsl_rng *);

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

int _mutate_draw (int t, int ell, float c, float d, int kappa, int omega, const gsl_rng *r) {
    return MIN((unsigned) omega, MAX((unsigned) kappa, ell + gsl_ran_poisson(r, t * c) -
                                                       gsl_ran_poisson(r, t * ell * d)));
}

void _trace_tree (int *coalescent_tree, int n, const gsl_rng *r) {
    for (int tau = 0; tau < 2 * n - 1; tau++) { // Diploid!
        int tau_0 = _triangle(tau + 0), tau_1 = _triangle(tau + 1); // Start at 2nd coalescence.

        // We save the indices of our ancestors to our descendants.
        for (int k = 0; k < tau_1 - tau_0; k++) {
        	coalescent_tree[tau_1 + k + 1] = tau_0 + k;
        }

        // Determine the individual whose frequency increases.
        gsl_ran_choose(r, &coalescent_tree[tau_1], 1, &coalescent_tree[tau_1 + 1], tau_1 - tau_0, sizeof(int));

        // Finally, shuffle the individuals in this generation.
        gsl_ran_shuffle(r, &coalescent_tree[tau_1], tau_1 - tau_0 + 1, sizeof(int));
    }
}

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
 */
void _evolve_event (PopulationTree *p, _mutate_f mutate) {
    int descendants_size, expected_time, t_coalescence;
    int *descendants = NULL;

    for (int tau = p->offset; tau < 2 * p->theta.n - 1; tau++) {
        // We define our descendants for the tau'th coalescent.
        descendants = &(p->coalescent_tree[_triangle(tau + 1)]);
        descendants_size = _triangle(tau + 2) - _triangle(tau + 1);

        // Determine time to coalescence. This is exponentially distributed, but the mean stays the same. Scale by f.
        expected_time = (int) (p->theta.f * 2 * p->theta.n / (float) _triangle(tau + 1));
        t_coalescence = MAX(1, _round_num(gsl_ran_exponential(p->r, expected_time)));

        // Iterate through each of the descendants (currently indices) and determine each ancestor.
        for (int k = 0; k < descendants_size; k++) {
            _evolve_individual(k, descendants, t_coalescence, mutate, p);
        }
    }
}

void _evolve (int *i_0, int i_0_size, PopulationTree *p) {
    // Determine our offset, and seed our ancestors for the tree.
    p->offset = i_0_size - 1;
    for (int k = 0; k < i_0_size; k++) {
        p->coalescent_tree[k + _triangle(p->offset)] = i_0[k];
    }

    // From our common ancestors, descend forward in time and populate our tree with repeat lengths.
    _evolve_event(p, _mutate_draw);
}

void _cleanup (PopulationTree *p) {
    free(p->coalescent_tree);
    gsl_rng_free(p->r);
}
