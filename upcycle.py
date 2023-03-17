import os
import re

from condor import SymbolicArray, sympy2casadi, TableLookup, Solver

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


def extract_problem_data(prob):
    out_meta = prob.model._var_abs2meta["output"]
    cons_meta = prob.driver._cons
    obj_meta = prob.driver._objs
    dv_meta = prob.driver._designvars

    n = len(prob.model._outputs)
    x0 = np.zeros(n)
    lbx = np.full(n, -np.inf)
    ubx = np.full(n, np.inf)
    iobj = None
    count = 0
    explicit_idxs = []
    non_indep_idxs = []
    constant_idxs = []
    for name, meta in out_meta.items():
        size = meta["size"]

        # x0[count : count + size] = meta["val"].flatten()
        x0[count : count + size] = prob.model._outputs[name].flatten()

        lb = meta["lower"]
        ub = meta["upper"]
        if lb is not None:
            lbx[count : count + size] = lb
        if ub is not None:
            ubx[count : count + size] = ub

        if name in cons_meta:
            lbx[count : count + size] = cons_meta[name]["lower"]
            ubx[count : count + size] = cons_meta[name]["upper"]

        # TODO make sure explicit indepvarcomp sets this tag
        if "openmdao:indep_var" in meta["tags"]:
            found = False
            for dv_name, dv_meta_loc in dv_meta.items():
                if name == dv_meta_loc["source"]:
                    found = True
                    lbx[count : count + size] = dv_meta_loc["lower"]
                    ubx[count : count + size] = dv_meta_loc["upper"]
            if not found:
                constant_idxs.append(count)

        else:
            non_indep_idxs += range(count, count + size)

        if "openmdao:allow_desvar" not in meta["tags"]:
            explicit_idxs += range(count, count + size)

        if name in obj_meta:
            iobj = count

        count += size

    return x0, lbx, ubx, iobj, non_indep_idxs, explicit_idxs, constant_idxs


class UpcycleSystem:

    def __init__(self, path, solver):
        self.path = path
        self.solver = solver

        # external inputs
        self.inputs = []

        # explicit: output name -> RHS expr
        # implicit: output n ame -> resid expr
        self.outputs = {}

        # output name -> output symbol
        self.out_syms = {}

        self.solver = solver
        self.solver.children.append(self)

    def __repr__(self):
        return self.path


class UpcycleExplicitSystem(UpcycleSystem):
    pass


class UpcycleImplicitSystem(UpcycleSystem):
    pass


class UpcycleSolver:

    def __init__(self, path="", parent=None):
        self.path = path
        self.parent = parent

        self.inputs = []  # external inputs (actually src names)
        self.solved_outputs = []

        self.children = []
        if parent is not None:
            self.parent.children.append(self)

    def __repr__(self):
        return self.path

    def iter_solvers(self, include_self=True):
        if include_self:
            yield self

        for child in self.children:
            if isinstance(child, UpcycleSolver):
                yield child
                child.iter_solvers()

    def iter_systems(self):
        for child in self.children:
            if isinstance(child, UpcycleSolver):
                yield from child.iter_systems()
            else:
                yield child

    def add_inputs(self, inputs):
        new_inputs = [i for i in inputs if i not in self.inputs]
        self.inputs.extend(new_inputs)

    @property
    def outputs(self):
        return [o for s in self.iter_systems() for o in s.outputs]

    def _set_run(self, func):
        def wrapper(*inputs):
            return np.array(func(*inputs)).squeeze()
        self.run = wrapper


def get_sources(conn_df, sys_path, parent_path):
    sys_conns = conn_df[conn_df["tgt"].str.startswith(sys_path + ".")]
    suffix = "." if parent_path else ""
    external_conns = ~sys_conns["src"].str.startswith(parent_path + suffix)
    return sys_conns["src"], external_conns


def upcycle_problem(make_problem):
    prob = make_problem()
    up_prob = make_problem()

    prob.final_setup()
    up_prob, _, _ = sympify_problem(up_prob)

    top_upsolver = UpcycleSolver()
    upsolver = top_upsolver

    vdat = _get_viewer_data(prob)
    conn_df = pd.DataFrame(vdat["connections_list"])

    for omsys in up_prob.model.system_iter(include_self=False, recurse=True):
        syspath = omsys.pathname

        while not syspath.startswith(upsolver.path):
            # special case: solver in the om system has nothing to solve
            # remove self from parent, re-parent children, propagate up inputs
            if len(upsolver.solved_outputs) == 0:
                upsolver.parent.children.pop(-1)

                for child in upsolver.children:
                    child.parent = upsolver.parent
                    upsolver.parent.children.append(child)

                upsolver.parent.add_inputs(upsolver.inputs)

            # propagate inputs external to self up to parent
            all_inputs, ext_mask = get_sources(conn_df, upsolver.path, upsolver.parent.path)
            upsolver.parent.add_inputs(all_inputs[ext_mask])
            upsolver = upsolver.parent

        nls = omsys.nonlinear_solver
        if nls is not None and not isinstance(nls, om.NonlinearRunOnce):
            upsolver = UpcycleSolver(path=syspath, parent=upsolver)

        if isinstance(omsys, om.Group):
            continue

        if isinstance(omsys, om.IndepVarComp):
            if upsolver is top_upsolver:
                upsolver.inputs.extend(flatten_varnames(omsys._var_abs2meta["output"]))
                continue
            else:
                warnings.warn("IVC not under top level model. Can be treated as explicit?")

        # propagate inputs external to self up to parent solver
        sys_input_srcs, ext_mask = get_sources(conn_df, syspath, upsolver.path)
        all_sys_input_srcs = flatten_varnames(prob.model._var_abs2meta["output"], sys_input_srcs)
        ext_sys_input_srcs = flatten_varnames(prob.model._var_abs2meta["output"], sys_input_srcs[ext_mask])
        upsolver.add_inputs(ext_sys_input_srcs)

        out_meta = omsys._var_abs2meta["output"]
        all_outputs = flatten_varnames(out_meta)

        if isinstance(omsys, om.ExplicitComponent):  # TODO and not unintentionally implicit
            upsys = UpcycleExplicitSystem(syspath, upsolver)
            upsys.inputs.extend(all_sys_input_srcs)
            for name, symbol, expr in zip(all_outputs, omsys._outputs.asarray(), omsys._residuals.asarray()):
                # TODO why does this happen?
                if isinstance(symbol, SymbolicArray):
                    symbol, expr = symbol[0], expr[0]
                upsys.outputs[name] = (symbol + expr).expand()

        else:
            upsys = UpcycleImplicitSystem(syspath, upsolver)
            upsys.inputs.extend(all_sys_input_srcs)
            for name, symbol, expr in zip(all_outputs, omsys._outputs.asarray(), omsys._residuals.asarray()):
                if isinstance(symbol, SymbolicArray):
                    symbol, expr = symbol[0], expr[0]
                upsys.outputs[name] = expr
                upsolver.solved_outputs.append(name)

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

    func, f = get_nlp_for_solver(top_upsolver, prob)

    return func, prob


def sanitize_solver_name(path):
    return path.replace(".", "_")


# solver dict of all child explicit system output exprs, keys are symbols
def get_nlp_for_solver(upsolver, prob):
    # upsolver is UpcycleSolver instance
    # prob is (root?) problem to get numeric values... can these get put in in the
    # upsolver and its children?

    # if top solver
    #   if there are constraints, add to solved_outputs, set appropriate bounds
    #   if there are design variables, update bounds on those inputs
    #   inputs that are not dvs need their value
    #   if any solved outputs or obective, return as nlpsolver
    #   otherwise return as explicit expression

    output_assignments = {}
    for child in upsolver.children:

        if isinstance(child, UpcycleExplicitSystem):
            output_assignments.update({sym.Symbol(k): v for k, v in child.outputs.items()})

        elif isinstance(child, UpcycleSolver):
            name = sanitize_solver_name(child.path)

            if name not in Solver.registry:
                sub_nlp, sub_S, sub_kwargs = get_nlp_for_solver(child, prob)
                Solver.registry[name] = make_solver_wrapper(sub_S, sub_kwargs)

            output_assignments.update({
                tuple(
                    [sym.Symbol(k) for k in child.solved_outputs]
                    + [sym.Symbol(k) for k in child.outputs if k not in child.solved_outputs]
                ) :
                Solver(name)(sym.Array([sym.Symbol(k) for k in child.inputs]))
            })

    s = [sym.Symbol(s) for s in upsolver.solved_outputs + upsolver.inputs]

    if upsolver.path == "" and len(upsolver.solved_outputs) == 0:
        exprs = [sym.Symbol(outp) for outp in upsolver.outputs]
        cr, cs, f = sympy2casadi(exprs, s, output_assignments, False)
        func = casadi.Function("model", casadi.vertsplit(cs), cr)
        upsolver._set_run(func)
        return upsolver, prob

    implicit_residuals = [
        v.subs(output_assignments) 
        for s in upsolver.children 
        if isinstance(s, UpcycleImplicitSystem)
        for v in s.outputs.values()
    ]
    cr, cs, f = sympy2casadi(implicit_residuals, s, output_assignments)

    cs_split = casadi.vertsplit(cs)
    x = casadi.vertcat(*cs_split[:len(upsolver.solved_outputs)])
    p = casadi.vertcat(*cs_split[len(upsolver.solved_outputs):])

    x0 = np.hstack([get_val(prob, absname) for absname in upsolver.solved_outputs])
    p0 = np.hstack([get_val(prob, absname) for absname in upsolver.inputs])
    lbx = np.hstack([s.lbx for s in upsolver.children if isinstance(s, UpcycleImplicitSystem)])
    ubx = np.hstack([s.ubx for s in upsolver.children if isinstance(s, UpcycleImplicitSystem)])

    n_explicit = len(cr) - x0.size
    lbg = np.hstack([np.zeros_like(lbx), np.full(n_explicit, -np.inf)])
    ubg = np.hstack([np.zeros_like(ubx), np.full(n_explicit, np.inf)])

    nlp = {"x": x, "p": p, "f": 0, "g": casadi.vertcat(*cr)}
    opts =  {}
    opts["ipopt.warm_start_init_point"] = "no"
    S = casadi.nlpsol("S", "ipopt", nlp, opts)
    kwargs = dict(x0=x0, p=p0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    if upsolver.path == "":
        upsolver._set_run(make_solver_wrapper(S, kwargs))
        return upsolver, prob

    return nlp, S, kwargs



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
            name = f"{self.name}_{output_name}"

            if name not in TableLookup.registry:
                ca_interp = casadi.interpolant(
                    name,
                    "linear",
                    self.inputs,
                    self.training_outputs[output_name].ravel(order="F"),
                )
                TableLookup.registry[name] = make_interp_wrapper(ca_interp)

            # TODO flatten inputs?
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




def make_solver_wrapper(solver, solver_kwargs):
    def wrapper(*args):
        nx = len(solver_kwargs["ubx"])

        symbolic_kwargs = solver_kwargs.copy()
        if isinstance(args[0], list) and isinstance(args[0][0], (casadi.MX, casadi.SX)):
            symbolic_kwargs["p"] = casadi.vertcat(*args[0])
        else:
            # not just for symbolic
            symbolic_kwargs["p"] = np.array(args)
        symbolic_nlp = solver(**symbolic_kwargs)

        return casadi.vertsplit(symbolic_nlp["x"]) + casadi.vertsplit(symbolic_nlp["g"][nx:])

    return wrapper





def flatten_varnames(abs2meta, varpaths=None):
    names = []
    if varpaths is None:
        varpaths = abs2meta.keys()
    for path in varpaths:
        names.extend(expand_varname(path, abs2meta[path]["size"]))
    return names


def expand_varname(name, size=1):
    if size == 1:
        yield name
    else:
        for i in range(size):
            yield f"{name}[{i}]"


_VARNAME_PATTERN = re.compile(r"(.*)\[(\d)+\]$")


def get_val(prob, name, **kwargs):
    try:
        return prob.get_val(name, **kwargs)[0]
    except KeyError:
        match = _VARNAME_PATTERN.match(name)
        base_name = match[1]
        idx = int(match[2])
        return prob.get_val(base_name, **kwargs)[idx]

class SymbolicVector(DefaultVector):
    """OpenMDAO Vector implementation backed by SymbolicArray"""

    def __getitem__(self, name):
        """Ensure accessing items always gives an array (as opposed to a scalar)"""
        return np.atleast_1d(super().__getitem__(name)).view(SymbolicArray)

    # override
    def _create_data(self):
        """Replace root Vector's ndarray allocation with SymbolicArray"""
        system = self._system()
        names = []
        # om uses this and relies on ordering when building views, should be ok
        names = flatten_varnames(system._var_abs2meta[self._typ])
        syms = np.array([sym.Symbol(name) for name in names]).view(SymbolicArray)
        self.syms = syms
        return syms

    def set_var(self, name, val, idxs=None, flat=False, var_name=None):
        """Disable setting values"""
        pass
