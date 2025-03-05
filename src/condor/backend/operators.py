from ._get_backend import get_backend
backend_mod = get_backend()
# operators should be...
# ~ array API 
#   algebra and trig binary/unary ops
#   set reduction: (f)min/max, sum??
#   limited manipulation: concat, stack, split?, reshape?
#   concat = backend_mod.concat

#
# ~ calculus
#    - jacobian
#    - jacobian_product? hessp? later
# symbolic operators
#    - if_else
#    - substitute?

#    - NOT callable/expression to operator

# constants
pi = backend_mod.operators.pi
inf = backend_mod.operators.inf

# calculus & symbolic
jacobian = backend_mod.operators.jacobian
recurse_if_else = backend_mod.operators.recurse_if_else
substitute = backend_mod.operators.substitute

# creation functions
zeros = backend_mod.operators.zeros
eye = backend_mod.operators.eye
ones = backend_mod.operators.ones

# "manipulation functions"
concat = backend_mod.operators.concat
#stack?
unstack = backend_mod.operators.unstack

# "element-wise functions"
def wrap(f):
    def _(*args, **kwargs):
        new_args = [getattr(arg, "backend_repr", arg) for arg in args]
        new_kwargs = {k: getattr(v, "backend_repr", v) for k, v in kwargs.items()}
        return f(*new_args, **new_kwargs)
    return _

min = wrap(backend_mod.operators.min)
max = wrap(backend_mod.operators.max)
mod = wrap(backend_mod.operators.mod)

tan = wrap(backend_mod.operators.tan)
atan = wrap(backend_mod.operators.atan)
atan2 = wrap(backend_mod.operators.atan2)
sin = wrap(backend_mod.operators.sin)
cos = wrap(backend_mod.operators.cos)
asin = wrap(backend_mod.operators.asin)
acos = wrap(backend_mod.operators.acos)
exp = wrap(backend_mod.operators.exp)
log = wrap(backend_mod.operators.log)
log10 = wrap(backend_mod.operators.log10)
sqrt = wrap(backend_mod.operators.sqrt)

vector_norm = wrap(backend_mod.operators.vector_norm)

