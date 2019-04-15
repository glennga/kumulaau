#include <Python.h>
#include "_single.h"

/**
 * The tree tracing method, to be called directly from Python. We accept 6 parameters here, all a part of the
 * "BaseParameters" class in "population.py":
 *
 * 1. n -- (int) Population size, used for determining the number of generations between events.
 * 2. f -- (float) Scaling factor for the total mutation rate. Smaller = shorter time to coalescence.
 * 3. c -- (float) Constant bias for the upward mutation rate.
 * 4. d -- (float) Linear bias for the downward mutation rate.
 * 5. kappa -- (int) Lower bound of repeat lengths.
 * 6. omega -- (int) Upper bound of repeat lengths.
 *
 * We also seed our RNG here using the current time of day and allocate memory for our coalescent tree.
 *
 * @param self: Unused, but required in signature I guess.
 * @param args: Arguments from the Python call. See list above.
 * @return: A pointer to the population structure holding the given parameters, the generated coalescent tree, and RNG.
 */
static PyObject *trace (PyObject *self, PyObject *args) {
    // Put our population object on the heap, to be shared between the trace and evolve steps.
    PopulationTree *p = (PopulationTree *) malloc(sizeof(PopulationTree));

    // Parse our arguments.
    if (!PyArg_ParseTuple(args, "ifffii", &p->theta.n, &p->theta.f, &p->theta.c, &p->theta.d,
                          &p->theta.kappa, &p->theta.omega))
        return NULL;

    // Generate our random seed based on the current time.
    gsl_rng_env_setup();
    struct timeval tv;
    gettimeofday(&tv, 0);
    unsigned long s = tv.tv_sec + tv.tv_usec;

    // Setup our generator.
    const gsl_rng_type *T = gsl_rng_taus2;
    p->r = gsl_rng_alloc(T);
    gsl_rng_set(p->r, s);

    // Reserve space for our ancestor chain.
    p->coalescent_tree = (int *) malloc(_triangle(2 * p->theta.n) * sizeof(int));
    memset(p->coalescent_tree, 0, _triangle(2 * p->theta.n) * sizeof(int));

    // Trace our tree. We do not perform repeat length determination at this step.
    _trace_tree(p->coalescent_tree, p->theta.n, p->r);
#ifdef _DEBUGGING_POPULATION_ENABLED_
    // Verify that our tree has been generated. Look at the indices of our array representation.
    printf("\nContents of tree trace: ");
    for (int k = 0; k < _triangle(2 * p->theta.n); k++) {
        printf("[%d]", p->coalescent_tree[k]);
    }
    printf("\n");
#endif
    // Return our pointer to our tree to evolve. This is meant to be passed to the call of "evolve".
    return PyCapsule_New(p, NULL, NULL);
}

/**
 * The repeat length determination method, to be called directly from Python. We accept 2 parameters here:
 *
 * 1. p -- (PopulationTree) The population structure generated from a "trace" call.
 * 2. i_0 -- (list of ints) A Python list of integers holding the seed lengths.
 *
 * NOTE: No error checking occurs to see if the number of seeds passed fill in all slots of a given coalescent event.
 * For instance you may pass in 2 lengths, which leaves the individuals of the 2nd coalescent event to be both
 * determined and not determined.
 *
 * After evolving our entire tree, we release the memory incurred by our RNG and the entire tree. The evolved
 * individuals are saved to a Python list and returned.
 *
 * @param self: Unused, but required in signature I guess.
 * @param args: Arguments from the Python call. See list above.
 * @return: A Python list containing 2n evolved individuals. The remaining individuals are currently discarded.
 */
static PyObject *evolve (PyObject *self, PyObject *args) {
    PyObject *i_0_list, *p_capsule = NULL;
    PopulationTree *p;

    // Parse our arguments.
    if (!PyArg_ParseTuple(args, "OO", &p_capsule, &i_0_list))
        return NULL;

    // Parse the population object generated from the trace call.
    if (!(p = (PopulationTree *) PyCapsule_GetPointer(p_capsule, NULL)))
        return NULL;
#ifdef _DEBUGGING_POPULATION_ENABLED_
    // Check if our tree still exists.
    printf("\nContents of tree import: ");
    for (int k = 0; k < _triangle(2 * p->theta.n); k++) {
        printf ("[%d]", p->coalescent_tree[k]);
    }
    printf("\n");
#endif
    // Verify that our list is not empty.
    int i_0_size = PyObject_Length(i_0_list);
    if (i_0_size < 0) return NULL;

    // Reserve space for our seed array.
    int *i_0 = (int *) malloc(i_0_size * sizeof(int));

    // Parse our seed array. We assume that our array are integers.
    for (int k = 0; k < i_0_size; k++) {
        i_0[k] = PyLong_AsLong(PyList_GetItem(i_0_list, k));
    }
#ifdef _DEBUGGING_POPULATION_ENABLED_
    // Verify that our seed lengths were properly imported.
    printf("\nContents of seed length argument: ");
    for (int k = 0; k < i_0_size; k++) {
        printf ("[%d]", i_0[k]);
    }
    printf("\n");
#endif
    // Evolve our population.
    _evolve(i_0, i_0_size, p);
    int *i_evolved = &(p->coalescent_tree[_triangle(2 * p->theta.n) - 2 * p->theta.n]);
#ifdef _DEBUGGING_POPULATION_ENABLED_
    // Peer into the contents of the repeat length determination step.
    printf("\nResults of evolution: ");
    for (int k = 0; k < _triangle(2 * p->theta.n); k++) {
        printf ("[%d]", p->coalescent_tree[k]);
    }
    printf("\n");
#endif
    // Store our evolved generation of ancestors in a Python list.
    PyObject *i_evolved_list = PyList_New(2 * p->theta.n);
    for (size_t k = 0; k != (size_t) 2 * p->theta.n; ++k) {
        PyList_SET_ITEM(i_evolved_list, k, PyLong_FromLong(i_evolved[k]));
    }

    // Cleanup, and return our result.
    _cleanup(p);
    return i_evolved_list;
}

/**
 * Array of Python methods available for use in this module. A trace method and an evolve method.
 */
static PyMethodDef popMethods[] = {
        {"trace",  trace,  METH_VARARGS, "Creates an evolutionary tree."},
        {"evolve", evolve, METH_VARARGS, "Evolves a given evolutionary tree."},
        {NULL,     NULL,   0,            NULL}
};

/**
 * A structure describing the module itself.
 */
static struct PyModuleDef popModule = {
        PyModuleDef_HEAD_INIT,
        "pop",
        "Python module to create evolutionary trees and evolve them in C.",
        -1,
        popMethods
};

/**
 * Our initialization function for the module.
 */
PyMODINIT_FUNC PyInit_pop (void) {
    return PyModule_Create(&popModule);
}
