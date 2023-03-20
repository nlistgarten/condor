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

import warnings


os.environ["OPENMDAO_REPORTS"] = "none"
# NO_DUMMY -> everything must be pre-sanitized
NO_DUMMY = True
#NO_DUMMY = False
# last attempt with no_dummy had a dummy referenced before assignment



class dummify_assignment_cse:
    def __init__(self, assignment_dict, return_assignments=True):
        self.assignment_dict = assignment_dict
        self.return_assignments = return_assignments
        self.original_arg_to_dummy = {}
        # TODO: 
        self.dummified_assignment_list = {}
        for arg, expr in assignment_dict.items():
            if hasattr(arg, "__len__"):
                new_arg = tuple([sym.Dummy() for a in arg])
                for a, na in zip(arg, new_arg):
                    self.original_arg_to_dummy[a] = na
            else:
                new_arg = sym.Dummy()
                self.original_arg_to_dummy[arg] = new_arg
            self.dummified_assignment_list[new_arg] = expr.subs(
                self.original_arg_to_dummy
            )

    def __call__(self, expr):
        rev_dummified_assignments = {
            v: k for k, v in self.dummified_assignment_list.items()
        }
        rev_dummified_assignments.update(
            {
                kk: vv for k, v in rev_dummified_assignments.items()
                if isinstance(k, tuple) for kk, vv in zip(k, v)
            },
        )
        rev_dummified_assignments.update(self.original_arg_to_dummy)
        if hasattr(expr, 'subs'):
            expr_ = expr.subs(rev_dummified_assignments)
        else:
            expr_ = [
                expr__.subs(rev_dummified_assignments)
                for expr__ in expr
            ]
        return self.assignment_dict.items(), expr

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
        # I think since each solver has its childrens output, we dont need to recurse
        # again
        return [o for s in self.children for o in s.outputs]
        return [o for s in self.iter_systems() for o in s.outputs]

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


def get_sources(conn_df, sys_path, parent_path):
    sys_conns = conn_df[conn_df["tgt"].str.startswith(sys_path + ".")]
    suffix = "." if parent_path else ""
    external_conns = ~sys_conns["src"].str.startswith(parent_path + suffix)
    return sys_conns["src"], external_conns


def upcycle_problem(make_problem, warm_start=False):
    prob = make_problem()
    up_prob = make_problem()

    prob.final_setup()
    up_prob, _, _ = sympify_problem(up_prob)

    top_upsolver = UpcycleSolver(om_equiv=prob.model)
    upsolver = top_upsolver

    vdat = _get_viewer_data(prob)
    conn_df = pd.DataFrame(vdat["connections_list"])

    for omsys in up_prob.model.system_iter(include_self=False, recurse=True):
        syspath = omsys.pathname

        while not syspath.startswith(upsolver.path):
            # special case: solver in the om system has nothing to solve
            # remove self from parent, re-parent children, propagate up inputs
            cyclic_io = set(upsolver.inputs) & set(upsolver.outputs)
            #I don't think cyclic_io should happen anymore?
            if cyclic_io:
                breakpoint()
            if (len(upsolver.implicit_outputs) == 0) or cyclic_io:

                for child in upsolver.children:
                    upsolver.parent.add_child(child)

            else:
                upsolver.parent.add_child(upsolver)

            upsolver = upsolver.parent

        nls = omsys.nonlinear_solver
        if nls is not None and not isinstance(nls, om.NonlinearRunOnce):
            upsolver = UpcycleSolver(path=syspath, parent=upsolver, om_equiv=omsys)

        if isinstance(omsys, om.Group):
            continue

        if isinstance(omsys, om.IndepVarComp):
            print("indepvarcomp?")

            flat_varnames = flatten_varnames(omsys._var_abs2meta["output"])
            if NO_DUMMY:
                flat_varnames = sanitize_variable_names(flat_varnames)

            if upsolver is not top_upsolver:
                warnings.warn("IVC not under top level model. Adding " +
                              "\n".join(flat_varnames) + " to " + upsolver.path 
                              + " as well as top level.")
                top_upsolver.add_inputs(flat_varnames)
            upsolver.add_inputs(flat_varnames)
            continue

        # propagate inputs external to self up to parent solver
        sys_input_srcs, ext_mask = get_sources(conn_df, syspath, upsolver.path)
        all_sys_input_srcs = flatten_varnames(prob.model._var_abs2meta["output"], sys_input_srcs)
        ext_sys_input_srcs = flatten_varnames(prob.model._var_abs2meta["output"], sys_input_srcs[ext_mask])

        #upsolver.add_inputs(ext_sys_input_srcs)

        out_meta = omsys._var_abs2meta["output"]
        all_outputs = flatten_varnames(out_meta)


        # cleaning variable names here
        if NO_DUMMY:
            all_outputs = sanitize_variable_names(all_outputs)
            clean_all_sys_input_srcs = sanitize_variable_names(all_sys_input_srcs)
        else:
            clean_all_sys_input_srcs = all_sys_input_srcs


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



    func, f = get_nlp_for_solver(top_upsolver, prob, warm_start=warm_start)

    return func, prob




# solver dict of all child explicit system output exprs, keys are symbols
def get_nlp_for_solver(upsolver, prob, warm_start=False):
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
            # TODO: I'm sanitizing names too early -- can I go back to dummify but
            # generate names of the form _dummy_<autonum>_<original var name>
            # or even _z<autonum>_<...>

            sub_solver_name = sanitize_variable_name(child.path)

            if sub_solver_name not in Solver.registry:
                # TODO: this set of calls gets made at least once more? so refactor?
                sub_nlp_args, sub_solver, sub_kwargs = get_nlp_for_solver(child, prob)
                Solver.registry[sub_solver_name] = make_solver_wrapper(sub_solver, sub_kwargs)

            output_assignments.update({
                tuple(
                    [sym.Symbol(k) for k in child.implicit_outputs]
                    + [sym.Symbol(k) for k in child.outputs if k not in child.implicit_outputs]
                ) :
                Solver(sub_solver_name)(sym.Array([sym.Symbol(k) for k in child.inputs]))
            })

    residual_args = [sym.Symbol(s) for s in upsolver.implicit_outputs + upsolver.inputs]

    print(upsolver.path, '...', sep='')
    if upsolver.path == "" and len(upsolver.implicit_outputs) == 0:
        exprs = [sym.Symbol(outp) for outp in upsolver.outputs]
        cr, cs, f = sympy2casadi(exprs, residual_args, output_assignments, False)
        func = casadi.Function("model", casadi.vertsplit(cs), cr)
        upsolver._set_run(func)
        return upsolver, prob

    implicit_residuals = [
        v.subs(output_assignments) 
        for s in upsolver.children 
        if isinstance(s, UpcycleImplicitSystem)
        for v in s.outputs.values()
    ]
    cr, cs, f = sympy2casadi(implicit_residuals, residual_args, output_assignments)

    cs_split = casadi.vertsplit(cs)
    x = casadi.vertcat(*cs_split[:len(upsolver.implicit_outputs)])
    p = casadi.vertcat(*cs_split[len(upsolver.implicit_outputs):])

    x0 = np.hstack([get_val(prob, absname) for absname in upsolver.implicit_outputs])
    p0 = np.hstack([get_val(prob, absname) for absname in upsolver.inputs])
    lbx = np.hstack([s.lbx for s in upsolver.children if isinstance(s, UpcycleImplicitSystem)])
    ubx = np.hstack([s.ubx for s in upsolver.children if isinstance(s, UpcycleImplicitSystem)])

    n_explicit = len(cr) - x0.size
    lbg = np.hstack([np.zeros_like(lbx), np.full(n_explicit, -np.inf)])
    ubg = np.hstack([np.zeros_like(ubx), np.full(n_explicit, np.inf)])

    nlp_args = {"x": x, "p": p, "f": 0, "g": casadi.vertcat(*cr)}
    opts =  {}
    opts["ipopt.warm_start_init_point"] = "yes" if warm_start else "no"
    if upsolver.path:
        solver_name = sanitize_variable_name(upsolver.path)
    else:
        solver_name = 'solver'
    solver = casadi.nlpsol(solver_name, "ipopt", nlp_args, opts)
    kwargs = dict(x0=x0, p=p0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    if upsolver.path == "":
        upsolver._set_run(make_solver_wrapper(solver, kwargs))
        return upsolver, prob

    return nlp_args, solver, kwargs



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
    return path.replace(".", "_").replace(':', '_')

def sanitize_variable_names(paths):
    return [ sanitize_variable_name(path) for path in paths ]

_VARNAME_PATTERN = re.compile(r"(.*)\[(\d)+\]$")

def contaminate_variable_name(clean_name):
    return clean_name.replace('_dot_', '.').replace('_colon_', ':')

def get_val(prob, name, **kwargs):
    if NO_DUMMY: # re-contaminate variable names to interact with om
        name = contaminate_variable_name(name)
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
        # om uses this and relies on ordering when building views, should be ok
        names = flatten_varnames(system._var_abs2meta[self._typ])
        if NO_DUMMY:
            names = sanitize_variable_names(names)
        syms = np.array([sym.Symbol(name) for name in names]).view(SymbolicArray)
        self.syms = syms
        return syms

    def set_var(self, name, val, idxs=None, flat=False, var_name=None):
        """Disable setting values"""
        pass
