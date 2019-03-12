# Coalescent Based Microsatellite Simulator + Analysis for Different Demographic Models
This repository holds research toward the modeling and analysis of microsatellite mutations, and a package for microsatellite demographic model inference. The name _kumulaau_ comes from the Hawaiian word for tree, which serves as the backbone for our speedy simulations.

## Getting Started with Kumulaau
1. Clone this repository!
```bash
git clone https://github.com/glennga/kumulaau.git
```

2. This project uses Python 3.7, Matplotlib, Numpy, Numba, Scipy, and GSL. An `environment.yml` file is provided to build a Conda clone of the Python environment used to develop this:
```bash
conda env create -f environment.yml
conda activate kumulaau
conda list
```

4. Install GSL. For instructions on how to do so, follow here: http://www2.lawrence.edu/fast/GREGGJ/CMSC210/gsl/gsl.html

5. Install our module. This will build our C extensions.
```bash
conda activate kumulaau
pip install kumulaau
```

6. Populate the observed database `observed.db` by running the ALFRED script:
```bash
kumulaau/data/alfred/alfred.sh
```

7. Run one of the examples listed in the `models` folder:
```bash
# If working without the HPC:
kumulaau/models/ma1t0s0i/ma1t0s0i.sh ${RESULTS_DATABASE}

# If you are on the UH HPC w/ SLURM (runs three separate instances):
sbatch ./script/ma1t0s0i/ma1t0s0i.slurm --array=0-3
```

## ABC-MCMC Single Population Example
### Usage of `kumulaau.Parameter`
The `Parameter` class is an ABC which holds all parameters associated with a given model. Two methods must be defined: the constructor and `_validity`. The former defines all parameters a model requires, while the latter defines what a valid parameter set is (returns `True` if the parameter set is valid, `False` otherwise). When defining the child constructor, the base constructor must use keyword arguments to pass the model specific parameters.
```python
from kumulaau import Parameter

class ParameterExample(Parameter):
    def __init__(self, n: int, f: float, c: float, d: float, kappa: int, omega: int):
        # Requirement: Call of base constructor uses keyword arguments.
        super().__init__(n=n, f=f, c=c, d=d, kappa=kappa, omega=omega)  
		
    def _validity(self): 
        return self.n * self.c > 0 and self.f * self.d >= 0 and 0 < self.kappa < self.omega
```

### Usage of `kumulaau.Model`
The `Model` class is an ABC which holds the actions of the demographic model itself. Two methods must be defined: `_generate_topology` and `_resolve_lengths`. The former defines all populations that are associated with a model, while the latter defines how the populations interact with each other. 

The first method, `_generate_topology`, requires that (a) a parameter set `theta` be passed (the one defined in previous section) and (b) a list of `self.pop_trace()` results be returned. There are six parameters associated with this call: 

| Parameter          | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| n                  | Starting population size.                                    |
| f                  | Scaling factor for total mutation rate.                      |
| c                  | Constant bias for the upward mutation rate.                  |
| d                  | Linear bias for the downward mutation rate.                  |
| kappa              | Lower bound of repeat lengths.                               |
| omega              | Upper bound of repeat lengths.                               |

The second method, `_resolve_lengths`, requires that (a) an iterable of seed lengths (ancestors) be passed and (b) the result of a single `self.pop_evolve()` call be returned (this is the final population to compare with observations). The first parameter to this call is one of the `_generate_topology` results in `self.generate_topology_results`, while the second is the iterable of seed lengths. The length of this iterable must be a triangle number, to avoid misshapen topologies.

```python
from kumulaau import Model

class ModelExample(Model):
    def _generate_topology(self, theta):
        # Requirement: Return a list of self.pop_trace() calls.
        return [self.pop_trace(theta.n, theta.f, theta.c, theta.d, theta.kappa, theta.omega)]
	
    def _resolve_lengths(self, i_0):
        # Requirements: (a) Utilize the topologies from self.generate_topology_results.
        #               (b) i_0 must be triangle number.
        #               (c) The result of a single self.pop_evolve() call be returned.
        return self.pop_evolve(self.generate_topology_results[0], i_0)
```

### Usage of `kumulaau.Distance`
The `Distance` class is an ABC which holds all functions associated with comparing an observed sample in `kumulaau/data/observed.db` and the results of a `kumulaau.Model`. Unlike the previous two classes, there exists two preexisting child classes of `kumulaau.Distance` that can be used without defining a new one: `kumulaau.Cosine`, the angular distance between the repeat length frequencies of the observed and generated data, and `kumulaau.Euclidean`, the Euclidean distance between the repeat length frequencies of the observed and generated data.

To define your own `Distance` child, one method must be defined: the `_delta` method. This is given three parameters: (a) the generated sample vector, a NumPy array of the repeat lengths from a `kumulaau.Model`'s `evolve()` call, (b) an observed frequency sample vector, a NumPy array which holds the frequency of an observation as a sparse vector indexed by repeat length, and (c) the bounds vector, holding the upper and lower bounds of the repeat length space respectively. The result returned should exist between 0 and 1, with 0 = identical and 1 = maximally dissimilar.

### Usage of `kumulaau.Posterior`
The `Posterior` class is an ABC which holds all data and functions associated with an MCMC variant. There exists only one working implementation of this at the moment, which is the ABC-MCMC approach, the `MCMCA` class. Five properties and one method must be defined: `MODEL_NAME`, `MODEL_SCHEME_SQL`, `POPULATION_CLASS`, `PARAMETER_CLASS`, `DISTANCE_CLASS` and `_walk`. The former five are given in the table below, while the latter defines how one moves between states in an MCMC run.

| Property           | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `MODEL_NAME`       | String of the name of the model, to use with the tables in the recorded database. |
| `MODEL_SCHEME_SQL` | String of the SQL schema associated with the model, **in the same order as the defined Parameters  constructor**. Parameters should be seperated by commas, and should specify their SQLite type after the their name. |
| `MODEL_CLASS`      | Model associated with this MCMC. A child of the `Model` class. |
| `PARAMETER_CLASS`  | Parameters associated with the model defined above. A child of the `Parameter` class. |
| `DISTANCE_CLASS`   | Distance function to use with this MCMC. A child of the `Distance` class. |

The static `_walk` method requires that a `Parameter` object be passed in with a distribution parameter collection, and that another `Parameter` object be returned. An example of walking with a multivariate normal distribution centered around `theta` with deviation specified in `walk_params` is given below.

```python
from kumulaau import MCMCA, Cosine
class MCMCAExample(MCMCA):
    MODEL_NAME = "EXAMPLE_MODEL"
    MODEL_SCHEME_SQL = "N INT, F FLOAT, C FLOAT, D FLOAT, KAPPA INT, OMEGA INT"
    MODEL_CLASS = ModelExample
    PARAMETER_CLASS = ParameterExample
    DISTANCE_CLASS = Cosine
	
    @staticmethod
    def _walk(theta, walk_params):
        # Requirements: (a) Parameter instance is passed in.
        #               (b) Walk parameters are passed in.
        #               (c) A different Parameter instance is returned.
        #               (d) Resulting Parameter instance is valid.
        from numpy.random import normal
        from numpy import nextafter

        return ParameterExample(
            n=max(round(normal(theta.n, walk_params.n)), 0),
            f=max(normal(theta.f, walk_params.f), 0),
            c=max(normal(theta.c, walk_params.c), nextafter(0, 1)),
            d=max(normal(theta.d, walk_params.d), 0),
            kappa=max(round(normal(theta.kappa, walk_params.kappa)), 0),
            omega=max(round(normal(theta.omega, walk_params.omega)), theta.kappa))
```

### Running the MCMC for a Custom Model
To run the MCMC for the model you have now created, create an MCMC instance and call the `.run()` method. This will save the results of the MCMC run in three tables: a `(MODEL_NAME)_OBSERVED` table, a `(MODEL_NAME)_MODEL` table, and a `(MODEL_NAME)_RESULTS` table. The first table holds the observed data used for an MCMC run, the second holds the parameters associated with some posterior sample, and the third holds the results associated with that posterior sample. All results share a primary key `TIME_R`, which is the time the recorded was recorded.

| Parameter        | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| connection_m     | SQLite connection to the database to log to.                 |
| connection_o     | Connection to `data/observed.ell`.                           |
| uid_observed     | ID of the observed samples to compare to.                    |
| locus_observed   | Loci of observed samples (must match with uids).             |
| simulation_n     | Number of simulations to use to obtain a distance.           |
| iterations_n     | Number of iterations to run MCMC for.                        |
| flush_n          | Number of iterations to run MCMC before flushing to disk.    |
| epsilon          | Maximum acceptance value for distance between [0, 1].        |
| walk_params      | Parameters associated with your walk function.               |
| theta_0 (optional)         | Initial state (Parameter object) to use with the MCMC. *If this is not specified, the MCMC will assume that `connection_m` has already run and will pull the last recorded model parameters.* |

```python
from types import SimpleNamespace
from sqlite3 import connect

# Connect to our databases.
connection_m = connect('data/myexampleresults.db')
connection_o = connect('data/observed.db')

# Define our starting point.
theta_0 = ParameterExample(n=100, f=100, c=0.001, d=0.0001, kappa=3, omega=30)

# Define our deviation parameters as a SimpleNamespace.
walk_params = SimpleNamespace(n=0.0, f=0.0, c=0.0003, d=5.5e-5, kappa=0.0, omega=0.0)

# Define the UIDs and Loci associated with frequency entries in `data/observed.ell`.
uid_observed = ['SA001097R', 'SA001098S', 'SA001538R', 'SA001539S', 'SA001540K']
locus_observed = ['D16S539' for _ in uid_observed]

# Run our MCMC!
MCMCAExample(connection_m=connection_m, 
             connection_o=connection_o,
             theta_0=theta_0,
             walk_params=walk_params,
             uid_observed=uid_observed,
             locus_observed=locus_observed,
             simulation_n=100,
             iterations_n=100,
             flush_n=100,
             epsilon=0.4).run()

# View the results of our MCMC by querying our results database.
results = connection_m.execute(f"""
    SELECT A.N, A.F, A.C, A.D, A.KAPPA, A.OMEGA, B.WAITING_TIME, B.PROPOSED_TIME, B.DISTANCE
    FROM {MCMCAExample.MODEL_NAME}_MODEL A
    INNER JOIN {MCMCAExample.MODEL_NAME}_RESULTS B USING (TIME_R)
""").fetchall()
[print(result) for result in results]

# If desired, one can continue the previous MCMC run by creating another MCMCAExample instance w/o theta_0.
MCMCAExample(connection_m=connection_m, 
             connection_o=connection_o,
             walk_params=walk_params,
             uid_observed=uid_observed,
             locus_observed=locus_observed,
             simulation_n=100,
             iterations_n=100,
             flush_n=100,
             epsilon=0.4).run()

# Close our connections when we are finished.
connection_m.commit(), connection_o.close(), connection_m.close()
```