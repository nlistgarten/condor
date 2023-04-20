from dataclasses import dataclass

@dataclass
class BackendSymbolData:
    shape: tuple # TODO: tuple of ints
    symmetric: bool
    diagonal: bool
    size: int
