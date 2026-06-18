# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Custom model entry point.

Importing this package imports ``resnet`` so all builders register themselves.
"""

from .registry import available_models, create_model
from . import resnet as _resnet  # noqa: F401

__all__ = ["available_models", "create_model"]
