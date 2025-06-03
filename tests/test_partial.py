"""Exercise wrapping different systems overriding some inputs with constants"""

import condor


def test_embedded_explicit():
    class EmbSys(condor.ExplicitSystem):
        a = input()
        b = input()
        output.c = a + b

    class WrapSys(condor.ExplicitSystem):
        a = input()
        out = EmbSys(a=a, b=4)
        output.c = out.c

    assert WrapSys(a=3).c == 7


def test_embedded_algebraic():
    class EmbSys(condor.AlgebraicSystem):
        a = parameter()
        b = variable()
        residual(a**2 == b)

    assert EmbSys(a=4).b == 16

    class WrapSys(condor.ExplicitSystem):
        out = EmbSys(a=3)
        output.b = out.b

    assert WrapSys().b == 9


def test_embedded_optimization():
    class EmbSys(condor.OptimizationProblem):
        a = parameter()
        b = variable()
        objective = (a**2 - b) ** 2

        class Options:
            print_level = 0

    assert EmbSys(a=4).b == 16

    class WrapSys(condor.ExplicitSystem):
        out = EmbSys(a=3)
        output.b = out.b

    assert WrapSys().b == 9
