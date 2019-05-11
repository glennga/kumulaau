# Coalescent Based Microsatellite Simulator + Analysis for Different Demographic Models
This repository holds research toward the modeling and analysis of microsatellite mutations, and a package for microsatellite demographic model inference. The name _kumulaau_ comes from the Hawaiian word for tree, which serves as the backbone for our speedy simulations.

## Getting Started with Kumulaau
1. Clone this repository!
```bash
git clone https://github.com/glennga/kumulaau.git
```

2. This project uses Python 3.7, Matplotlib, Numpy, Numba, and GSL. An `environment.yml` file is provided to build a Conda clone of the Python environment used to develop this:
```bash
conda env create -f environment.yml
conda activate kumulaau
conda list
```

4. Install GSL. For instructions on how to do so, follow here: http://www2.lawrence.edu/fast/GREGGJ/CMSC210/gsl/gsl.html

5. Install our module. This will build our C extensions.
```bash
conda activate kumulaau
python3 setup.py build
python3 setup.py install
```

6. Populate the observed database `observed.db` by running the ALFRED script:
```bash
kumulaau/data/alfred/alfred.sh ${OBSERVED_DATABASE}
```

7. Run one of the examples listed in the `models` folder:
```bash
kumulaau/models/abc1t0s0i/abc1t0s0i.sh ${RESULTS_DATABASE} ${OBSERVED_DATABASE}
```

## ABC-MCMC Single Population Example
### Usage of `kumulaau.Parameter`
The `Parameter` class is an ABC which holds all parameters associated with a given model. Two methods must be defined: the constructor and `validity`. The former defines all parameters a model requires, while the latter defines what a valid parameter set is (returns `True` if the parameter set is valid, `False` otherwise). When defining the child constructor, the base constructor must use keyword arguments to pass the model specific parameters.
```python
from kumulaau import Parameter

class MyParameter(Parameter):
    def __init__(self, n: int, f: float, c: float, d: float, kappa: int, omega: int):
        # Requirement: Call of base constructor uses keyword arguments.
        super().__init__(n=n, f=f, c=c, d=d, kappa=kappa, omega=omega)  
		
    def validity(self): 
        return self.n * self.c > 0 and self.f * self.d >= 0 and 0 < self.kappa < self.omega
```

Aside from holding the parameters, an implementation of `Parameter` also allows one to construct an instance of this implementation given a namespace and some transformation function. This allows one to use a package like `argparse` with ease:
```python
parser = ArgumentParser()
parser.add_argument('-n', type=int)
parser.add_argument('-f', type=float)
parser.add_argument('-c', type=float)
parser.add_argument('-d', type=float)
parser.add_argument('-kappa', type=int)
parser.add_argument('-omega', type=int)
theta = MyParameter.from_namespace(parser.parse_args())
```
The *transform* argument to this function allows one to instantiate a `Parameter` implementation using names with a suffix or prefix to the parameters themselves:
```python
parser = ArgumentParser()
parser.add_argument('-n_sp', type=int)
parser.add_argument('-f_sp', type=float)
parser.add_argument('-c_sp', type=float)
parser.add_argument('-d_sp', type=float)
parser.add_argument('-kappa_sp', type=int)
parser.add_argument('-omega_sp', type=int)
theta = MyParameter.from_namespace(parser.parse_args(), lambda a: a + '_sp')
```

For posterior walk functions (i.e. generating a new point given a current point and a description of its randomness), a decorator is provided: `@Parameter.walkfunction`. This will utilize the `validity` function to ensure that a valid parameter set is always generated.


### Usage of `kumulaau.model`
The `model` module holds all functions required to simulate the evolution of a single population. There are two functions available here: `trace` and `evolve`. The former generates the topology associated with a single population and returns a pointer to be passed to the latter. There are six parameter associated with the `trace` function:

| Parameter          | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| n                  | Starting population size.                                    |
| f                  | Scaling factor for total mutation rate.                      |
| c                  | Constant bias for the upward mutation rate.                  |
| d                  | Linear bias for the downward mutation rate.                  |
| kappa              | Lower bound of repeat lengths.                               |
| omega              | Upper bound of repeat lengths.                               |

The second function, `evolve` accepts two parameters: the first is the result of the `trace` call and the second is an iterable of seed lengths (i.e. ancestors). The length of this iterable must be a triangle number, to avoid misshapen topologies.

### Usage of `kumulaau.observed`

The `observed` module holds all functions associated with interacting with the database generated by the ALFRED script, as well as all functions associated with transforming the base representation of our observations, `List[List[Tuple(int, float)]]`, to other forms. The outer list specifies different population samples while the inner list specifies (repeat length, frequency) tuples for specific populations. See below for an example. Note that it is entirely possible to avoid using the ALFRED script for posterior inference, you just need to specify your own observed distributions in the base representation.

Below is the SQLite DDL associated with the ALFRED database:

```sqlite
CREATE TABLE OBSERVED_ELL (
    TIME_R TIMESTAMP,  -- entryDate in ALFRED TSV --
    POP_NAME TEXT,  -- popName in ALFRED TSV --
    POP_UID TEXT,  -- popUId in ALFRED TSV --
    SAMPLE_UID TEXT,  -- sampleUId in ALFRED TSV --
    SAMPLE_SIZE INT,  -- 2N in ALFRED TSV --
    LOCUS TEXT,  -- locusSymbol in ALFRED TSV --
    ELL TEXT,  -- alleleSymbol in ALFRED TSV --
    ELL_FREQ FLOAT  -- frequency in ALFRED TSV --
);
```

To distinguish population samples here, one must specify two items: `SAMPLE_UID` and `LOCUS`. We can now delve into the call required to extract observations in our base representation: `extract_alfred_tuples`. The first argument `uid_loci` is a sequence of tuples, whose first point is a `SAMPLE_UID` entry while the second is a `LOCUS` entry. 

```python3
>>> uid_loci = [('SA001097R', 'D16S539'), ('SA001098S', 'D16S539')]
>>> observed.extract_alfred_tuples(uid_loci)
[[(12, 0.23), (14, 0.02), (10, 0.16), (8, 0.02), (13, 0.13), (9, 0.2), (11, 0.25)], [(13, 0.14), (8, 0.02), (9, 0.12), (12, 0.27), (10, 0.18), (11, 0.28)]]
```

### Usage of `kumulaau.distance`

The `distance` module holds all functions associated with describing the different between two populations of microsatellites. We offer three different statistics: `mean`, `deviation`, and `frequency`. The latter statistic describes the frequency of the mean repeat length. The method of importance here is `summary_factory`, which creates a compiled function to compute a vector of summary statistics to use to compare two populations.

```python
from kumulaau import distance

summarizer = distance.summary_factory(['mean', 'deviation', 'frequency'], [3, 30])
```

*The following paragraphs describe the internals behind finding the likelihood of some parameter set $\theta$ given observations.* This is split into two phases: a shape definition phase and a matrix population phase. 

The shape definition phase is specified by `generate_hdo`, which returns a binary match matrix $H$ (rows = `kumulaau.evolve` result, columns = observed distribution, enter 1 if the computed distance falls below some $\epsilon$, otherwise 0), a distance matrix $D$ (rows = `kumulaau.evolve` result, columns = observed distribution, enter distance between a simulation and observation), and a sparse observation matrix $O$ (rows = repeat length, columns = observation, enter frequency of repeat length associated with that observation). In this phase, only the $O$ matrix is populated while the $H$ and $D$ matrices's shape is defined. This call accepts our observations in base representation, the number of `kumulaau.evolve` instances desired (*simulation_n*), and the bounds of our repeat length space (*bounds*). This call returns a namespace holding all matrices, as well as the observations and bounds used to create these matrices.

The matrix population phase fills in the entries of the $H​$ and $D​$ matrices, and is specified by the `populate_hd` call. There are five parameters required here: the result from our `generate_hdo` call, a function that returns the result of a `kumulaau.evolve` call given some parameter set $\theta​$ (*sample*) and common ancestor length $\ell​$, a function that computes the distance between an observed distribution and a `kumulaau.evolve` call (*delta*), the parameter set $\theta​$ of interest (*theta_proposed*), and an $\epsilon​$ term that defines what a match is. What follows is:

1. We collect the result of calling $sample(\theta_{proposed}, \ell)$ *simulation_n* times. $\ell$ is treated as a nuisance parameter and is sampled randomly from the lengths in our observations.
2. We iterate through each generated sample and observed sample, compute the distance to save to $D$, and fill our $H$ matrix accordingly (1 if $D$ entry $< \epsilon​$, 0 otherwise).
3. The expected distance (mean of all entries in $D$), the $D$ matrix itself, and the $H$ matrix itself are returned in a namespace.

Likelihood determination varies on usage of ABC or ELE.

### Usage of `kumulaau.abc`

There exists only one method associated with this module: `run`, which defines an ABC-MCMC approach to inferring the likelihood of different parameter sets. There are eight parameters that must be defined here:

| Parameter     | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| *walk*        | Function that accepts some parameter set and returns another parameter set. This is used to jump between different parameter sets, and the sensitivity is to be specified by the user. This must have a signature of `walk(theta)`. |
| *sample*      | Function that produces a `kumulaau.evolve` result, given a parameter set $\theta$ and some common ancestor $\ell$. This must have a signature of `sample(theta, i_0: Sequence)`. |
| *summarize*       | Function that computes a summary vector of some population. |
| *log_handler* | Function that handles what occurs with the Markov chain and results at a specific iteration $i$. This must have a signature of `log_handler(x: List, i: int)`. |
| *theta\_0*    | Initial starting point to use with MCMC.                     |
| *observed*    | Observations in base representation.                         |
| *epsilon*     | Maximum acceptance value for distance between [0, 1].        |
| *boundaries*  | Starting and ending iteration for this specific MCMC run. Used to continue MCMC runs and specify when to stop the MCMC. |

Below represents a MWE to utilize the `run` method.

```python
from kumulaau import *

class MyParameter(Parameter):
    def __init__(self, n: int, f: float, c: float, d: float, kappa: int, omega: int):
        # Requirement: Call of base constructor uses keyword arguments.
        super().__init__(n=n, f=f, c=c, d=d, kappa=kappa, omega=omega)

    def validity(self):
        return self.n * self.c > 0 and self.f * self.d >= 0 and 0 < self.kappa < self.omega
    
def sample(theta, i_0):
    topology = model.trace(theta.n, theta.f, theta.c, theta.d, theta.kappa, theta.omega)
    return model.evolve(topology, i_0)  # Must return the result of evolve call.

@MyParameter.walkfunction
def walk(theta):
    from numpy.random import normal
    from numpy import nextafter

    return MyParameter(n=theta.n,  # No change in N.
                       f=theta.f,  # No change in F.
                       c=max(normal(theta.c, 0.0003), nextafter(0, 1)),
                       d=max(normal(theta.d, 5.5e-5), 0),
                       kappa=theta.kappa,  # No change in kappa.
                       omega=theta.omega)  # No change in omega.

def log_handler(x, i):
    [print(a) for a in x[1:]]  # Print every element but first.
    x[:] = [x[-1]]  # Remove all elements but last.
    
# Define the UIDs and Loci associated with frequency entries in `data/observed.ell`.
uid = ['SA001097R', 'SA001098S', 'SA001538R', 'SA001539S', 'SA001540K']
loci = ['D16S539' for _ in uid]

# Collect our observations in base representation.
observations = observed.extract_alfred_tuples(zip(uid, loci))

# Define our starting point.
theta_0 = MyParameter(n=100, f=100, c=0.001, d=0.0001, kappa=3, omega=30)

# Create our summarizer using the mean and deviation.
summarize = kumulaau.distance.summary_factory(['mean', 'deviation'], [3, 30])

# Run our MCMC!
abc.run(walk=walk, sample=sample, summarize=summarize, log_handler=log_handler,
        theta_0=theta_0, observed=observations, epsilon=0.4, boundaries=[0, 1000])
```

Continuing from the `kumulaau.distance` section, to determine the likelihood is to use the function `likelihood_from_h`, passing the $H$ result from `populated_hd` as a parameter. The average of each column (i.e. observation) is computed, representing the probability of a model matching this specific observation. To compute the likelihood is to take the product of each average. We assume that each probability is independent.

### Usage of `kumulaau.ele`

There exists only one method associated with this module: `run`, which defines a weighted logarithmic regression approach to determine likelihood instead of using a cutoff $\epsilon$ in `kumulaau.abc`. There are nine parameters that must be defined here:

| Parameter     | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| *walk*        | Function that accepts some parameter set and returns another parameter set. This is used to jump between different parameter sets, and the sensitivity is to be specified by the user. This must have a signature of `walk(theta)`. |
| *sample*      | Function that produces a `kumulaau.evolve` result, given a parameter set $\theta$ and some common ancestor $\ell$. This must have a signature of `sample(theta, i_0: Sequence)`. |
| *summarize*       | Function that computes a summary vector of some population. |
| *log_handler* | Function that handles what occurs with the Markov chain and results at a specific iteration $i$. This must have a signature of `log_handler(x: List, i: int)`. |
| *theta\_0*    | Initial starting point to use with MCMC.                     |
| *observed*    | Observations in base representation.                         |
| *r*     | Exponential decay rate for weight vector used in regression.         |
| *bin_n*     | Number of bins used to construct CDF.         |
| *boundaries*  | Starting and ending iteration for this specific MCMC run. Used to continue MCMC runs and specify when to stop the MCMC. |

```python
.
.
.

# Run our MCMC!
ele.run(walk=walk, sample=sample, summarize=summarize, log_handler=log_handler,
        theta_0=theta_0, observed=observations, r=0.4, bin_n=500, boundaries=[0, 1000])
```

### Usage of `kumulaau.RecordSQLite`

The `RecordSQLite` class is a convenience class to record the results of a `run` method in the `posterior` module. Instead of the basic print-to-console `log_handler` defined in the *Usage of `kumulaau.abc`* section, we pass the handler specified in a `RecordSQLite` instance.

There exists three tables we log to here: a `_OBSERVED` table which holds all observations associated with a specific posterior run, a `_MODEL` table which holds the sequence of our parameters, and a `_RESULTS` which holds the posterior specific results associated with the parameter sequence. All tables are keyed by `RUN_R`, a randomly generated 10-digit alphanumeric string that distinguishes different posteriors runs from one another. The `_MODEL` and `_RESULTS` share a composite key of `(RUN_R, TIME_R)`, with `TIME_R` being the datetime associated with a specific parameter-result set.

To use this class, one must specify the following:

| Parameter        | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| *filename*       | Location of the SQLite database to save to.                  |
| *model\_name*    | Prefix to append to all tables associated with this specific posterior run. |
| *model_schema*   | Schema of the _MODEL table, does not include `RUN_R, TIME_R`. |
| *is_new_run*     | Flag which indicates if the current run to be logged is new or not. This determines if we should query for old `RUN_R` entries or if we should generate a new one. |

Given that the observations associated with a specific posterior run will never change, there exists a separate method to record these separate from the posterior results themselves: `record_observed`. This accepts observations in our base representation and, optionally, a list of IDs to attach to each population sample in our observations. If the second argument is not specified, then each population is enumerated from 1 to `len(observations)`.

It is advised to use this class with a context manager as such, to ensure database consistency:

```python
MODEL_SQL = "N INT, F FLOAT, C FLOAT, D FLOAT, KAPPA INT, OMEGA INT"
MODEL_NAME = "ABC1T0S0I"

with RecordSQLite('data/results', MODEL_NAME, MODEL_SQL, False) as log:
    # Record our observations.
    log.record_observed(observations)

    # In order to match the signature of a log function, we use a lambda:
    handler = lambda a, b: log.handler(a, b, 100)  # This records every 100 iterations.
    
    .
    .
    .
    
    # Run our MCMC!
    abc.run(..., log_handler=handler, ...)
```
