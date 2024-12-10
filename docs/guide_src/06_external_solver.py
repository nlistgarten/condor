"""
=========================
6. External Solvers
=========================
"""

import condor
from condor import operators as ops

# %%
# :class:`ExternalSolverWrapper`
# ==============================
#
# External solver systems are not really mathematically distinct from explicit systems.
# They may be thought of as a collection of explicit functions. The distinction is in
# usage -- an external solver system provides a convenient method of wrapping existing
# computational routines and providing derivatives if available.

# TODO maybe we should just forgo the code sample here

import subprocess
import sys


class ExternalSolverWrapper(condor.ExternalSolverWrapper):

    def __init__(self):
        self.input(name="x1")
        self.input(name="x2")
        self.output(name="y1")
        self.output(name="y2")

    def function(self, x1, x2):
        out = subprocess.run(
            [sys.executable, "-c", f"print(2*{x1}); print({x2}**3)"],
            capture_output=True,
            text=True,
        )
        y1, y2 = map(float, out.stdout.split())
        return y1, y2

    def jacobian(self, x1, x2):
        # may provide jacobian dy/dx if available for use within a parent solver
        ...


ExtSys = ExternalSolverWrapper()
ext_out = ExtSys(1, 2)
print(ext_out.output)
