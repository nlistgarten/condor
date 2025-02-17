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


