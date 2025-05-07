from ._get_backend import get_backend
backend_mod = get_backend()

# non-operators -- more utility-like, ~outside scope of array API
symbol_class = backend_mod.symbol_class
symbol_generator = backend_mod.symbol_generator
get_symbol_data = backend_mod.get_symbol_data
symbol_is = backend_mod.symbol_is
BackendSymbolData = backend_mod.BackendSymbolData
callables_to_operator = backend_mod.callables_to_operator
expression_to_operator = backend_mod.expression_to_operator

process_relational_element = backend_mod.process_relational_element
is_constant = backend_mod.is_constant
evalf = backend_mod.evalf
SymbolCompatibleDict = backend_mod.SymbolCompatibleDict
symbols_in = backend_mod.symbols_in
# list of symbols in an expression, casadi.symvar, aesara.graph_inputs, possibly in
# jax.jax_expr stuff?
