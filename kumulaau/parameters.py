#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List
from argparse import Namespace


class Parameters(ABC):
    def __init__(self, c: float, d: float, kappa: int, omega: int):
        """ Every model involves the same mutation model (for now). This involves the parameters c, d, and our bounds
        [kappa, omega].

        :param c: Constant bias for the upward mutation rate.
        :param d: Linear bias for the downward mutation rate.
        :param kappa: Lower bound of repeat lengths.
        :param omega: Upper bound of repeat lengths.
        """
        self.c, self.d, self.kappa, self.omega = c, d, kappa, omega

    @property
    @abstractmethod
    def PARAMETER_COUNT(self):
        """ Enforce the definition of a parameter count.

        :return: None.
        """
        raise NotImplementedError

    def __iter__(self):
        """ Return each our of parameters in some standard order (based off _from_namespace).

        :return: Iterator for all of our parameters.
        """
        for parameter in self._from_namespace(self):
            yield parameter

    def __len__(self):
        """ The number of parameters that exist here.

        :return: The number of parameters we have.
        """
        return self.PARAMETER_COUNT

    @staticmethod
    @abstractmethod
    def _from_namespace(p) -> List:
        """ Return a list from a namespace in the same order of __iter__.

        :param p: Arguments from some namespace.
        :return: List from namespace.
        """
        raise NotImplementedError

    @classmethod
    def from_args(cls, arguments: Namespace):
        """ Given a namespace, return a Parameters object with the appropriate parameters.

        :param arguments: Arguments from some namespace.
        :return: New Parameters object with the parsed in arguments.
        """
        return cls(*cls._from_namespace(arguments))

    @classmethod
    def from_args_sigma(cls, arguments: Namespace):
        """ TODO

        :param arguments:
        :return:
        """
        return cls(*[vars(arguments)[i] for i in list(filter(lambda a: '_sigma' in a, vars(arguments)))])

    @abstractmethod
    def _walk_criteria(self) -> bool:
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
            theta_proposed = cls(*walk(theta, pi_sigma))

            if theta_proposed._walk_criteria():  # Only return if the parameter set is valid.
                return theta_proposed
