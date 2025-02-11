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

# "manipulation functions"
concat = backend_mod.operators.concat
#stack?
unstack = backend_mod.operators.unstack

# "element-wise functions"
min = backend_mod.operators.min
max = backend_mod.operators.max
mod = backend_mod.operators.mod

atan = backend_mod.operators.atan
atan2 = backend_mod.operators.atan2
sin = backend_mod.operators.sin
cos = backend_mod.operators.cos
asin = backend_mod.operators.asin
acos = backend_mod.operators.acos
exp = backend_mod.operators.exp
log = backend_mod.operators.log
log10 = backend_mod.operators.log10
sqrt = backend_mod.operators.sqrt

