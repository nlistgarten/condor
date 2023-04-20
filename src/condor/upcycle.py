import os
import re

from condor import SymbolicArray, TableLookup, Solver
from casadi_backend import sympy2casadi, SolverWithWarmStart

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
    pass

class UpcycleExplicitSystem(UpcycleSystem):
    def to_implicit(self):
        self.__class__ =  UpcycleImplicitSystem
        for output in self.outputs:
            self.outputs[output] = self.outputs[output] - self.output_symbols[output]


class UpcycleSolver:
    def __init__(self, path="", parent=None, om_equiv=None):
        self.path = path
        self.parent = parent
        self.om_equiv=om_equiv

        self.inputs = []  # external inputs (actually src names)

        self.internal_loop = False

        self.children = []

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
        for input_ in inputs:
            if input_ in self.outputs:
                # shouldn't happen if this is only getting called for ivcs?
                breakpoint()

            if input_ not in self.inputs:
                self.inputs.append(input_)

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


    def _set_run(self, func):
        def wrapper(*inputs):
            return np.array(func(*inputs)).squeeze()
        self.run = wrapper


    def __call__(self, *args):
        is_casadi_symbolic = isinstance(args, (list, tuple)) and isinstance(args[0], (casadi.MX, casadi.SX))

        if str(self.casadi_imp).endswith('IpoptInterface') or str(self.casadi_imp).endswith('Qrsqp'): # no wrapper

            if self.implicit_outputs: # rootfinder, no wrapper
                if is_casadi_symbolic:
                    solver_out = self.casadi_imp(
                        x0=self.x0, p=casadi.vertcat(*args), lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg
                    )
                    return casadi.vertsplit(solver_out["x"]) + casadi.vertsplit(solver_out["g"][len(self.ubx):])

                solver_out = self.casadi_imp(
                    x0=self.x0, p=args, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg
                )
                return [np.concatenate([
                    solver_out["x"].toarray().reshape(-1),
                    solver_out["g"].toarray().reshape(-1)[len(self.ubx):]
                ])]
            else: # optimizer, no wrapper -- always numeric? for now...
                args = np.array(args)
                return self.casadi_imp(
                    x0=args[self.design_var_indices],
                    p=args[self.parameter_indices],
                    lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg
                )

        elif str(self.casadi_imp).endswith('CallbackInternal'): # SolverWithWarmstart
            if is_casadi_symbolic:
                out = self.casadi_imp(*args)
                return out
                #return casadi.vertsplit(out)
            else:
                out = self.casadi_imp(*args)
                return np.array(out).squeeze()

        else: # a function,
            return np.array(self.casadi_imp(*args)).squeeze()


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

            flat_varnames = flatten_varnames(omsys._var_abs2meta["output"])
            flat_varnames = sanitize_variable_names(flat_varnames)

            if upsolver is not top_upsolver:
                warnings.warn("IVC not under top level model. Adding " +
                              "\n".join(flat_varnames) + " to " + upsolver.path 
                              + " as well as top level.")
                # TODO: maybe don't need this with add_child
                top_upsolver.add_inputs(flat_varnames)
            upsolver.add_inputs(flat_varnames)
            continue

        # propagate inputs external to self up to parent solver
        sys_input_srcs = get_sources(conn_df, syspath, upsolver.path)
        all_sys_input_srcs = flatten_varnames(prob.model._var_abs2meta["output"], sys_input_srcs)


        out_meta = omsys._var_abs2meta["output"]
        all_outputs = flatten_varnames(out_meta)


        # cleaning variable names here
        all_outputs = sanitize_variable_names(all_outputs)
        clean_all_sys_input_srcs = sanitize_variable_names(all_sys_input_srcs)


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

        else:
            upsys = UpcycleImplicitSystem(syspath, upsolver)
            upsys.inputs.extend(clean_all_sys_input_srcs)
            for output_name, symbol, expr in zip(all_outputs, omsys._outputs.asarray(), omsys._residuals.asarray()):
                if isinstance(symbol, SymbolicArray):
                    symbol, expr = symbol[0], expr[0]
                upsys.output_symbols[output_name] = symbol
                upsys.outputs[output_name] = expr

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

    upsolver = get_nlp_for_rootfinder(top_upsolver, prob, warm_start=warm_start)

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
    output_assignments = {}
    for child in upsolver.children:

        if isinstance(child, UpcycleExplicitSystem):
            output_assignments.update({
                sym.Symbol(k): v for k, v in child.outputs.items()
            })

        elif isinstance(child, UpcycleSolver):

            sub_solver_name = sanitize_variable_name(repr(child))

            if sub_solver_name not in Solver.registry:
                # TODO: this set of calls gets made at least once more? so refactor?
                # TODO: should this get done at the bottom?
                sub_solver = get_nlp_for_rootfinder(child, prob)
                Solver.registry[sub_solver_name] = sub_solver

            output_assignments.update({
                tuple(
                    [sym.Symbol(k) for k in child.outputs]

                ) :
                Solver(sub_solver_name)(
                    #sym.Array(
                    *tuple(
                        [ sym.Symbol(k) for k in child.inputs]
                    )
                )
            })

    inputs = [sym.Symbol(s) for s in upsolver.inputs]

    exprs = [sym.Symbol(outp) for outp in upsolver.outputs]
    cr, cs, f = sympy2casadi(exprs, inputs, output_assignments, False)

    cons_meta = prob.driver._cons
    obj_meta = prob.driver._objs
    dv_meta = prob.driver._designvars

    upsolver.lbg = np.full(len(upsolver.outputs), -np.inf)
    upsolver.ubg = np.full(len(upsolver.outputs), np.inf)
    upsolver.lambda_func = f

    idx_obj = upsolver.outputs.index(sanitize_variable_name(list(obj_meta.keys())[0]))

    for name, meta in cons_meta.items():
        if meta["size"] != 1:
            raise NotImplementedError(f"Constraint {name} has size > 1")

        idx = upsolver.outputs.index(sanitize_variable_name(name))
        upsolver.lbg[idx] = meta["lower"]
        upsolver.ubg[idx] = meta["upper"]

    upsolver.lbx = []
    upsolver.ubx = []
    upsolver.x0 = []
    upsolver.design_var_indices = []
    for tgtname, meta in dv_meta.items():
        srcname = meta["source"]
        for srcname_expanded in expand_varname(srcname, meta["size"]):
            idx = upsolver.inputs.index(sanitize_variable_name(srcname_expanded))
            upsolver.lbx.append(meta["lower"])
            upsolver.ubx.append(meta["upper"])
            upsolver.x0.append(get_val(prob, srcname_expanded, sanitized=False))
            upsolver.design_var_indices.append(idx)

    upsolver.p0 = []
    upsolver.parameter_indices = []
    for i, name in enumerate(upsolver.inputs):
        if i not in upsolver.design_var_indices:
            upsolver.p0.append(get_val(prob, name))
            upsolver.parameter_indices.append(i)

    x = casadi.vertcat(*[cs[i] for i in upsolver.design_var_indices])
    p = casadi.vertcat(*[cs[i] for i in upsolver.parameter_indices])
    f = cr[idx_obj]

    nlp_args = {"x": x, "p": p, "f": f, "g": casadi.vertcat(*cr)}

    opts = {
        "expand": False,
        "ipopt": {
            "tol": 1e-10,
            #"hessian_approximation": "limited-memory",
        }
    }
    upsolver.casadi_imp = casadi.nlpsol("optimizer", "ipopt", nlp_args, opts)

    return upsolver


# solver dict of all child explicit system output exprs, keys are symbols
def get_nlp_for_rootfinder(upsolver, prob, warm_start=False):
    # TODO: this is really a method on UpcycleSolver,
    # this is creating a casadi numerical implementation


    # upsolver is UpcycleSolver instance
    # prob is (root?) problem to get numeric values... can these get put in in the
    # upsolver and its children?

    # if top solver
    #   if there are constraints, add to implicit_outputs, set appropriate bounds
    #   if there are design variables, update bounds on those inputs
    #   inputs that are not dvs need their value
    #   if any solved outputs or obective, return as nlpsolver
    #   otherwise return as explicit expression

    output_assignments = {}
    for child in upsolver.children:

        if isinstance(child, UpcycleExplicitSystem):
            output_assignments.update({
                sym.Symbol(k): v for k, v in child.outputs.items()
            })

        elif isinstance(child, UpcycleSolver):

            sub_solver_name = sanitize_variable_name(child.path)

            if sub_solver_name not in Solver.registry:
                # TODO: this set of calls gets made at least once more? so refactor?
                # TODO: should this get done at the bottom?
                sub_solver = get_nlp_for_rootfinder(child, prob)
                Solver.registry[sub_solver_name] = sub_solver

            output_assignments.update({
                tuple(
                    [sym.Symbol(k) for k in child.outputs]
                ) :
                Solver(sub_solver_name)(
                    #sym.Array(
                    *tuple(
                        [ sym.Symbol(k) for k in child.inputs]
                    )
                )
            })


    residual_args = [sym.Symbol(s) for s in upsolver.implicit_outputs + upsolver.inputs]

    print(upsolver.path, '...', sep='')
    if upsolver.path == "" and len(upsolver.implicit_outputs) == 0:
        exprs = [sym.Symbol(outp) for outp in upsolver.outputs]
        cr, cs, f = sympy2casadi(exprs, residual_args, output_assignments, False)
        func = casadi.Function("model", casadi.vertsplit(cs), cr)
        upsolver.casadi_imp = func
        return upsolver

    implicit_residuals = [
        v.subs(output_assignments) 
        for s in upsolver.children 
        if isinstance(s, UpcycleImplicitSystem)
        for v in s.outputs.values()
    ]
    cr, cs, f = sympy2casadi(implicit_residuals, residual_args, output_assignments)
    upsolver.lambda_func = f

    cs_split = casadi.vertsplit(cs)
    x = casadi.vertcat(*cs_split[:len(upsolver.implicit_outputs)])
    p = casadi.vertcat(*cs_split[len(upsolver.implicit_outputs):])

    upsolver.x0 = np.hstack([get_val(prob, absname) for absname in upsolver.implicit_outputs])
    upsolver.p0 = np.hstack([get_val(prob, absname) for absname in upsolver.inputs])
    upsolver.lbx = np.hstack([s.lbx for s in upsolver.children if isinstance(s, UpcycleImplicitSystem)])
    upsolver.ubx = np.hstack([s.ubx for s in upsolver.children if isinstance(s, UpcycleImplicitSystem)])

    n_exp = len(upsolver.explicit_outputs)
    n_imp = len(upsolver.implicit_outputs)


    upsolver.lbg = np.hstack([np.zeros(n_imp), np.full(n_exp, -np.inf)])
    upsolver.ubg = np.hstack([np.zeros(n_imp), np.full(n_exp, np.inf)])

    nlp_args = {"x": x, "p": p, "f": 0, "g": casadi.vertcat(*cr)}
    opts =  {}
    #opts["qpsol.error_on_fail"] = "False"
    opts["error_on_fail"] = False
    #opts["ipopt.warm_start_init_point"] = "yes" if warm_start else "no"
    #opts["ipopt.max_iter"] = 100
    if upsolver.path:
        solver_name = sanitize_variable_name(upsolver.path)
    else:
        solver_name = 'solver'

    nlp = upsolver.casadi_imp = SolverWithWarmStart(
        solver_name,
        nlp_args,
        upsolver.x0,
        upsolver.p0,
        upsolver.lbx,
        upsolver.ubx,
        upsolver.lbg,
        upsolver.ubg,
    )
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
