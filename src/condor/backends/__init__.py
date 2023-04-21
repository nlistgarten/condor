from dataclasses import dataclass, field
import numpy as np

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


