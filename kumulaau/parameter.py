#!/usr/bin/env python3
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable


class Parameter(ABC):
    def __init__(self, c: float, d: float, kappa: int, omega: int, **kwargs):
        """ Every model involves the same mutation model (for now). This involves the parameters c, d, and our bounds
        [kappa, omega]. Model specific parameters are specified in the kwargs argument. Call of base constructor must
        use keyword arguments.

        :param c: Constant bias for the upward mutation rate.
        :param d: Linear bias for the downward mutation rate.
        :param kappa: Lower bound of repeat lengths.
        :param omega: Upper bound of repeat lengths.
        """
        # Set our mutation model parameters.
        self.c, self.d, self.kappa, self.omega = c, d, kappa, omega

        # Set our model specific parameters.
        self.__dict__.update(kwargs)

    def __iter__(self):
        """ Return each our of parameters in the constructor order.

        :return: Iterator for all of our parameters.
        """
        for parameter in self.__dict__:
            yield parameter

    def __len__(self):
        """ The number of parameters that exist here.

        :return: The number of parameters we have.
        """
        return len(self.__dict__)

    @classmethod
    def from_namespace(cls, arguments, transform: Callable = lambda a: a):
        """ Given a namespace, return a Parameters object with the appropriate parameters. Transform each attribute
        (e.g. add a suffix or prefix) if desired.

        :param arguments: Arguments from some namespace.
        :param transform: Function to transform each attribute, given a string and returning a string.
        :return: New Parameters object with the parsed in arguments.
        """
        from inspect import getfullargspec

        return cls(*list(map(lambda a: getattr(arguments, transform(a)), getfullargspec(cls.__init__).args[1:])))

    @abstractmethod
    def validity(self) -> bool:
        """ Determine if a current parameter set is valid.

        :return: True if valid. False otherwise.
        """
        raise NotImplementedError

    @classmethod
    def walkfunction(cls, func: Callable) -> Callable:
        """ Decorator to apply validity constraints to a given walk function (generating a new point given a current
        point and variables describing it's randomness.

        :param func: Walk function.
        :return: Function that will generate new points that are valid.
        """
        @wraps(func)
        def _walkfunction(theta, walk_params):
            while True:
                theta_proposed = cls(**func(theta, walk_params).__dict__)

                if theta_proposed.validity():  # Only return if the parameter set is valid.
                    return theta_proposed

        return _walkfunction
