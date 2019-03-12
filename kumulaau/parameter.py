#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Namespace
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
    def from_namespace(cls, arguments: Namespace, transform: Callable = lambda a: a):
        """ Given a namespace, return a Parameters object with the appropriate parameters. Transform each attribute
        (e.g. add a suffix or prefix) if desired.

        :param arguments: Arguments from some namespace.
        :param transform: Function to transform each attribute, given a string and returning a string.
        :return: New Parameters object with the parsed in arguments.
        """
        return cls(*list(map(lambda a: getattr(arguments, transform(a)), cls.__dict__)))

    @abstractmethod
    def _validity(self) -> bool:
        """ Determine if a current parameter set is valid.

        :return: True if valid. False otherwise.
        """
        raise NotImplementedError

    @classmethod
    def from_walk(cls, theta, pi_sigma, walk: Callable):
        """ Generate a new point from some walk function. We apply this walk function to each dimension, using the
        walking parameters specified in 'pi_sigma'. 'walk' must accept two variables, with the first being
        the point to walk from and second being the parameter to walk with. We must be within bounds.

        :param theta: Current point in our model space. The point we are currently walking from.
        :param pi_sigma: Walking parameters. These are commonly deviations.
        :param walk: For some point theta, generate a new one with the corresponding pi_sigma.
        :return: A new Parameters (point).
        """
        while True:
            theta_proposed = cls(**walk(theta, pi_sigma).__dict__)

            if theta_proposed._validity():  # Only return if the parameter set is valid.
                return theta_proposed
