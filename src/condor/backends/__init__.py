from dataclasses import dataclass, field
import numpy as np

"""
Need to keep in a separate file so backend.get_symbol_data (required by fields which
generate the symbols) can return filled dataclass without causing a circular import
"""

@dataclass
class BackendSymbolData:
    shape: tuple # TODO: tuple of ints
    symmetric: bool
    diagonal: bool
    size: int = field(init=False)
    # TODO: mark size as computed? or replace with @property? can that be trivially cached?
    # currently computed by actual backend...

    def __post_init__(self, *args, **kwargs):
        n = self.shape[0]
        size = np.prod(self.shape)
        if self.symmetric:
            size = int(n*(n+1)/2)
        if self.diagonal:
            size = n

        if self.diagonal:
            assert size == self.shape[0]
        elif self.symmetric:
            assert len(shape) == 2
            assert self.shape[0] == self.shape[1]
        self.size = size


