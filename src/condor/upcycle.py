import os
import re

from condor.backends.scipy import sympy2casadi, SymbolicArray, TableLookup, Solver, assignment_cse

import casadi
import numpy as np
import sympy as sym
import pandas as pd
from openmdao.solvers.solver import NonlinearSolver
from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data
from openmdao.components.balance_comp import BalanceComp
from openmdao.components.meta_model_structured_comp import \
    MetaModelStructuredComp
from openmdao.core.problem import Problem
import openmdao.api as om
from openmdao.vectors.default_vector import DefaultVector
from pycycle.elements.US1976 import USatm1976Comp, USatm1976Data

import warnings


os.environ["OPENMDAO_REPORTS"] = "none"

def sympify_problem(prob):
    """Set up and run `prob` with symbolic inputs

    `prob` should *not* have had ``final_setup`` called.
    """
    prob.setup(local_vector_class=SymbolicVector, derivatives=False)
    prob.final_setup()
    # run_apply_nonlinear gives better matching to om but makes 1.0*sym everywhere
    prob.model.run_apply_nonlinear()

    root_vecs = prob.model._get_root_vectors()

    out_syms = root_vecs["output"]["nonlinear"].syms

    res_exprs = [
        rexpr
        for rarray in prob.model._residuals.values()
        for rexpr in sym.flatten(rarray)
        if not isinstance(rexpr, sym.core.numbers.Zero)
    ]
    res_mat = sym.Matrix(res_exprs)

    return prob, res_mat, out_syms


"""
These three "Upcycle" systems are maps from OpenMDAO constructs
UpcycleSolver also maps to a generic Rootfinding / Algebraic System of Equations model
in condor; not a sub-class but provides the same interface:
"""

class UpcycleSystem:
    """
    Base class for Implicit and Explic
    """
    def __init__(self, path, parent):
        self.path = path

        # external inputs
        self.inputs = []

        # explicit: output name -> RHS expr
        # implicit: output n ame -> resid expr
        self.outputs = {}

        # output name -> output symbol
        self.out_syms = {}
        self.parent = parent

        self.output_symbols = {}

    def __repr__(self):
        return self.path


class UpcycleImplicitSystem(UpcycleSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.declaration_string = ""
        self.residual_string = ""


class UpcycleExplicitSystem(UpcycleSystem):
    def to_implicit(self):
        self.__class__ =  UpcycleImplicitSystem
        for output in self.outputs:
            self.outputs[output] = self.outputs[output] - self.output_symbols[output]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_string = ""


class UpcycleSolver:
    def __init__(self, path="", parent=None, om_equiv=None, ):
        self.path = path
        self.parent = parent
        self.om_equiv=om_equiv

        self.inputs = []  # external inputs (actually src names)

        self.internal_loop = False

        self.children = []

        self.solver_name = sanitize_variable_name(repr(self))
        self.param_string = ""

    def __repr__(self):
        if self.path == "":
            return "root"
        return self.path

    def iter_solvers(self, include_self=True):
        if include_self:
            yield self

        for child in self.children:
            if isinstance(child, UpcycleSolver):
                yield from child.iter_solvers(include_self=True)

    def iter_systems(self):
        for child in self.children:
            if isinstance(child, UpcycleSolver):
                yield from child.iter_systems()
            else:
                yield child

    def add_child(self, child):
        child.parent = self
        # explicit/implicit systems have solvers not parents, unless they get reparented? 
        # was that causing problems?

        if self.internal_loop and isinstance(child, UpcycleExplicitSystem):
            child.to_implicit()

        for input_ in child.inputs:
            if input_ not in self.inputs and input_ not in self.outputs:
                self.inputs.append(input_)

        for output in child.outputs:
            if output in self.inputs:
                if not isinstance(child, UpcycleImplicitSystem):
                    self.internal_loop = True

                    if isinstance(child, UpcycleExplicitSystem):
                    # only the last explicit system should get turned into an implicit
                    # system? or go-back and turn all children into implicit?
                        child.to_implicit()
                        for sibling in self.children:
                            if isinstance(sibling, UpcycleExplicitSystem):
                                sibling.to_implicit()
                    else:
                        # Upsolver -- this shouldn't happen, especially with cyclic io
                        # check
                        breakpoint()

                self.inputs.remove(output)
            if output in self.outputs:
                breakpoint() # conflicting outputs

        self.children.append(child)

    def add_inputs(self, inputs):
        for om_input_name in inputs:
            input_ = sanitize_variable_name(om_input_name)
            if input_ in self.outputs:
                # shouldn't happen if this is only getting called for ivcs?
                breakpoint()

            if input_ not in self.inputs:
                self.inputs.append(input_)
                self.param_string += f"\n    {input_} = parameter(shape={inputs[om_input_name]['shape']})"


    @property
    def outputs(self):
        return self.implicit_outputs + self.explicit_outputs

    @property
    def implicit_outputs(self):
        return [o for s in self.children
                if isinstance(s, UpcycleImplicitSystem)
                for o in s.outputs]

    @property
    def explicit_outputs(self):
        return [o for s in self.children
                if not isinstance(s, UpcycleImplicitSystem) # solvers or explicit
                for o in s.outputs]

    @property
    def implicit_output_declaration_string(self):
        return "".join([
            sys.declaration_string
            for sys in self.children
            if isinstance(sys, UpcycleImplicitSystem)
        ])

    @property
    def implicit_output_residual_string(self):
        return "".join([
            sys.residual_string
            for sys in self.children
            if isinstance(sys, UpcycleImplicitSystem)
        ])


    @property
    def explicit_output_computation_string(self):
        return "".join([
            s.output_string
            for s in self.children
            if isinstance(s, (UpcycleExplicitSystem, UpcycleSolver))
        ])

    @property
    def explicit_output_assignment_string(self):
        return "".join([
            f"\n    explicit_output.{o} = {o}"
            for s in self.children
            if isinstance(s, (UpcycleExplicitSystem, UpcycleSolver))
            for o in s.outputs
        ])



    def _set_run(self, func):
        def wrapper(*inputs):
            return np.array(func(*inputs)).squeeze()
        self.run = wrapper


def get_sources(conn_df, sys_path, parent_path):
    sys_conns = conn_df[conn_df["tgt"].str.startswith(sys_path + ".")]
    return sys_conns["src"]

def pop_upsolver(syspath, upsolver):
    while not syspath.startswith(upsolver.path):
        # special case: solver in the om system has nothing to solve
        # remove self from parent, re-parent children, propagate up inputs
        cyclic_io = set(upsolver.inputs) & set(upsolver.outputs)
        #I don't think cyclic_io should happen anymore?
        if cyclic_io:
            breakpoint()
        if (len(upsolver.implicit_outputs) == 0) or cyclic_io:
            # TODO: delete cyclic?

            for child in upsolver.children:
                upsolver.parent.add_child(child)

        else:
            upsolver.parent.add_child(upsolver)

        upsolver = upsolver.parent
    return upsolver


def upcycle_problem(make_problem, warm_start=False):
    prob = make_problem()
    up_prob = make_problem()

    prob.final_setup()
    up_prob, _, _ = sympify_problem(up_prob)

    top_upsolver = UpcycleSolver(om_equiv=prob.model)
    upsolver = top_upsolver

    vdat = _get_viewer_data(prob)
    conn_df = pd.DataFrame(vdat["connections_list"])

    all_om_systems = list(
            up_prob.model.system_iter(include_self=False, recurse=True),
    )

    for omsys, numeric_omsys in zip(
            up_prob.model.system_iter(include_self=False, recurse=True),
            prob.model.system_iter(include_self=False, recurse=True),
    ):
        syspath = omsys.pathname


        upsolver = pop_upsolver(syspath, upsolver)

        nls = omsys.nonlinear_solver
        if nls is not None and not isinstance(nls, om.NonlinearRunOnce):
            upsolver = UpcycleSolver(path=syspath, parent=upsolver,
                                     om_equiv=numeric_omsys)

        if isinstance(omsys, om.Group):
            continue

        if isinstance(omsys, om.IndepVarComp):
            print("indepvarcomp?")


            if upsolver is not top_upsolver:
                warnings.warn("IVC not under top level model. Adding " +
                              "\n".join(flat_varnames) + " to " + upsolver.path 
                              + " as well as top level.")
                # TODO: maybe don't need this with add_child
                top_upsolver.add_inputs(omsys._var_abs2meta["output"])
            upsolver.add_inputs(omsys._var_abs2meta["output"])
            continue

        # propagate inputs external to self up to parent solver
        sys_input_srcs = get_sources(conn_df, syspath, upsolver.path)

        out_meta = omsys._var_abs2meta["output"]
        all_outputs = flatten_varnames(out_meta)
        # cleaning variable names here
        all_outputs = sanitize_variable_names(all_outputs)

        # don't need to flatten anymore
        clean_all_sys_input_srcs = sanitize_variable_names(sys_input_srcs)


        # TODO: can be DRY'd up, 
        if isinstance(omsys, om.ExplicitComponent) and not upsolver.internal_loop:
            upsys = UpcycleExplicitSystem(syspath, upsolver)
            upsys.inputs.extend(clean_all_sys_input_srcs)
            for output_name, symbol, expr in zip(all_outputs, omsys._outputs.asarray(), omsys._residuals.asarray()):
                # TODO why does this happen?
                if isinstance(symbol, SymbolicArray):
                    symbol, expr = symbol[0], expr[0]
                if expr == 0:
                    raise ValueError("null explicit system")
                upsys.output_symbols[output_name] = symbol
                upsys.outputs[output_name] = (symbol + expr).expand()
                upsys.output_string += f"\n    {output_name} = {upsys.outputs[output_name]}"

        else:
            upsys = UpcycleImplicitSystem(syspath, upsolver)
            upsys.inputs.extend(clean_all_sys_input_srcs)

            for om_name, symbol, expr in zip(out_meta, omsys._outputs.asarray(), omsys._residuals.asarray()):
                output_name = sanitize_variable_name(om_name)
                meta = out_meta[om_name]
                declare_string = f"\n    {output_name} = implicit_output("
                declare_string +=  f"\n        shape={meta['shape']},"
                declare_string += f"\n        initializer={prob.model._outputs[om_name]},"
                lb = meta["lower"]
                if lb is not None:
                    declare_string += f"\n        lower_bound={lb},"
                ub = meta["upper"]
                if ub is not None:
                    declare_string += f"\n        upper_bound={ub},"
                declare_string += "\n    )"
                upsys.declaration_string += declare_string

                if isinstance(symbol, SymbolicArray):
                    symbol, expr = symbol[0], expr[0]
                upsys.output_symbols[output_name] = symbol
                upsys.outputs[output_name] = expr
                upsys.residual_string += f"\n    residual.{output_name} = {expr}"



        upsys.x0 = np.zeros(len(omsys._outputs))
        upsys.lbx = np.full_like(upsys.x0, -np.inf)
        upsys.ubx = np.full_like(upsys.x0, np.inf)
        count = 0
        for absname, meta in out_meta.items():
            size = meta["size"]

            upsys.x0[count : count + size] = prob.model._outputs[absname].flatten()

            lb = meta["lower"]
            ub = meta["upper"]
            if lb is not None:
                upsys.lbx[count : count + size] = lb
            if ub is not None:
                upsys.ubx[count : count + size] = ub

            count += size

        upsolver.add_child(upsys)

    upsolver = pop_upsolver(top_upsolver.path, upsolver)

    upsolver = get_nlp_for_rootfinder(top_upsolver, prob,)

    if prob.driver._objs:# and False:
        updriver = UpcycleSolver(path="optimizer")
        updriver.add_child(upsolver)
        updriver = get_nlp_for_optimizer(updriver, prob)
        return updriver, prob
    else:
        return upsolver, prob

# TODO: generalize get_nlp_for... methods? currently its a mix of sympy and casadi
# backends.

def get_nlp_for_optimizer(upsolver, prob):

    cons_meta = prob.driver._cons
    obj_meta = prob.driver._objs
    dv_meta = prob.driver._designvars

    declare_variable_string = "".join([
          f"\n    {sanitize_variable_name(name)} = variable("
        + f"\n        shape={meta.get('shape', (1,))},"
        + f"\n        upper_bound={meta['upper']},"
        + f"\n        lower_bound={meta['lower']},"
        + f"\n        initializer={get_val(prob,meta['source'])},"
        +  "\n    )"
        for name, meta in dv_meta.items()
    ])


    constraint_string = "".join([
           "\n    constraint("
        + f"\n        {sanitize_variable_name(name)},"
        + f"\n        upper_bound={meta['upper']},"
        + f"\n        lower_bound={meta['lower']},"
        +  "\n    )"
        for name, meta in cons_meta.items()
    ])

    upsolver.model_string = "\n".join([
        f"class {upsolver.solver_name}(co.OptimizationProblem):"
        + declare_variable_string,
        upsolver.param_string,
        upsolver.explicit_output_computation_string,
        constraint_string,
        f"\n    objective = {sanitize_variable_name(list(obj_meta.keys())[0])}"
    ])
    breakpoint()
    return upsolver


# solver dict of all child explicit system output exprs, keys are symbols
def get_nlp_for_rootfinder(upsolver, prob,):

    upsolver.model_string = "\n".join([
        f"class {upsolver.solver_name}(co.AlgebraicSystem):"
        + upsolver.param_string,
        upsolver.implicit_output_declaration_string,
        upsolver.explicit_output_computation_string,
        upsolver.implicit_output_residual_string,
        upsolver.explicit_output_assignment_string,
    ])

    assigned_vars = ", ".join(upsolver.outputs)
    call_vars = ", ".join(upsolver.inputs)
    upsolver.output_string = f"\n    ({assigned_vars}) = {upsolver.solver_name}({call_vars})"
    Solver.registry[upsolver.solver_name] = upsolver

    return upsolver


orig_add_balance = BalanceComp.add_balance


def add_balance(self, *args, **kwargs):
    kwargs["normalize"] = False
    # kwargs["use_mult"] = False
    orig_add_balance(self, *args, **kwargs)


BalanceComp.add_balance = add_balance


orig_mm_compute = MetaModelStructuredComp.compute


def make_interp_wrapper(interp):
    def wrapper(*args):
        return interp(casadi.vertcat(*args[0]))

    return wrapper


def mmsc_compute(self, inputs, outputs):
    if isinstance(inputs, SymbolicVector):
        for output_name in outputs:
            name = f"{self.pathname}.{output_name}_table"
            name = sanitize_variable_name(name)

            if name not in TableLookup.registry:
                ca_interp = casadi.interpolant(
                    name,
                    "linear",
                    self.inputs,
                    self.training_outputs[output_name].ravel(order="F"),
                )
                TableLookup.registry[name] = make_interp_wrapper(ca_interp)

            # TODO flatten inputs?
            # TODO: the (symbolic) table should "own" the callable, data, etc.
            f = TableLookup(name)(sym.Array(sym.flatten(inputs.values())))
            outputs[output_name] = f
    else:
        orig_mm_compute(self, inputs, outputs)


MetaModelStructuredComp.compute = mmsc_compute

orig_usatm1976_compute = USatm1976Comp.compute


def usatm1976_compute(self, inputs, outputs):
    if isinstance(inputs, SymbolicVector):
        for output_name in outputs:
            fname = f"{self.name}_{output_name}"
            attr_name = output_name[:-1]
            ca_interp = casadi.interpolant(
                fname,
                "linear",
                [USatm1976Data.alt],
                getattr(USatm1976Data, attr_name),
            )
            ca_interp_wrapped = make_interp_wrapper(ca_interp)
            TableLookup.registry[fname] = ca_interp_wrapped
            f = TableLookup(fname)(sym.Array(sym.flatten(inputs.values())))
            outputs[output_name] = f
    else:
        orig_usatm1976_compute(self, inputs, outputs)


USatm1976Comp.compute = usatm1976_compute



def flatten_varnames(abs2meta, varpaths=None):
    names = []
    if varpaths is None:
        varpaths = abs2meta.keys()
    # TODO: I want a list comprehension?
    for path in varpaths:
        names.extend(expand_varname(path, abs2meta[path]["size"]))
    return names

def expand_varname(name, size=1):
    if size == 1:
        yield name
    else:
        for i in range(size):
            yield f"{name}[{i}]"


def sanitize_variable_name(path):
    # TODO: replace with regex, ensure no clashes?
    return path.replace(".", "_dot_").replace(':', '_colon_')

def sanitize_variable_names(paths):
    return [ sanitize_variable_name(path) for path in paths ]

_VARNAME_PATTERN = re.compile(r"(.*)\[(\d)+\]$")

def contaminate_variable_name(clean_name):
    return clean_name.replace('_dot_', '.').replace('_colon_', ':')

def get_val(prob, name, sanitized=True, **kwargs):
    if sanitized: # re-contaminate variable names to interact with om
        name = contaminate_variable_name(name)
    try:
        return prob.get_val(name, **kwargs)[0]
    except KeyError:
        match = _VARNAME_PATTERN.match(name)
        base_name = match[1]
        idx = int(match[2])
        return prob.get_val(base_name, **kwargs)[idx]

# SYMPY BACKEND FOR UPCYCLE -- I think casadi backend doesn't need getitem? easiest
# thing is just build a numpy array of MX objects
class SymbolicVector(DefaultVector):
    """OpenMDAO Vector implementation backed by SymbolicArray"""

    def __getitem__(self, name):
        """Ensure accessing items always gives an array (as opposed to a scalar)"""
        return np.atleast_1d(super().__getitem__(name)).view(SymbolicArray)

    # override
    def _create_data(self):
        """Replace root Vector's ndarray allocation with SymbolicArray"""
        system = self._system()
        # om uses this and relies on ordering when building views, should be ok
        names = flatten_varnames(system._var_abs2meta[self._typ])
        names = sanitize_variable_names(names)
        syms = np.array([sym.Symbol(name) for name in names]).view(SymbolicArray)
        self.syms = syms
        return syms

    def set_var(self, name, val, idxs=None, flat=False, var_name=None):
        """Disable setting values"""
        pass
