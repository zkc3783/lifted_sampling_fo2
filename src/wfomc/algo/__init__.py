from enum import Enum

from .IncrementalWFOMC3 import incremental_wfomc3
from .IncrementalWFOMC3 import incremental_wfoms3
from .IncrementalWFOMC3 import analyze_all_sample
from .IncrementalWFOMC32 import incremental_wfomc32
from .IncrementalWFOMC32 import incremental_wfoms32
__all__ = [

    "incremental_wfomc3",
    "incremental_wfoms3",
    "incremental_wfomc32",
    "incremental_wfoms32",
    "analyze_all_sample",
]


class Algo(Enum):
  
    INCREMENTAL3 = 'incremental3'
    INCREMENTAL32 = 'incremental32'

    def __str__(self):
        return self.value
