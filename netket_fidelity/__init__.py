from ._version import version as __version__  # noqa: F401

from . import operator

from . import driver

from .infidelity import InfidelityOperator
from .renyi2 import Renyi2EntanglementEntropy

from netket.utils import _hide_submodules

_hide_submodules(__name__, hide_folder=["renyi2", "infidelity"])
