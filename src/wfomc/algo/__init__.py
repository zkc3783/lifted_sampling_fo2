from enum import Enum

from .IncrementalWFOMC3 import incremental_wfomc3
from .IncrementalWFOMC3 import incremental_wfoms3
from .IncrementalWFOMC3 import analyze_all_sample
__all__ = [

    "incremental_wfomc3",
    "incremental_wfoms3",
    "analyze_all_sample"
]


class Algo(Enum):
  
    INCREMENTAL3 = 'incremental3'

    def __str__(self):
        return self.value
