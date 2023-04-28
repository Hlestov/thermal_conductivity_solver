import sympy as sp
from sympy.simplify.fu import TRpower
from sympy.integrals.meijerint import _mul_as_two_parts


def get_coefficients(expr):
    """
    Возвращает коэффициенты c_i и n_i линейной комбинации
    вида Sum(c_i*(sin(n_i)))
    
    Args: 
        expr (sympy.Function): линейная комбинация вида sum(c_i*sin(n_i*x))

    Returns:
        tuple([c_i], [n_i])
    """

    x = sp.symbols('x')

    # Списки для хранения c_i и n_i
    c = []
    n = []
    
    # Разбиваем функцию на слагаемые
    if isinstance(expr, sp.Add):
        terms = expr.args
    else:
        terms = [expr]

    # Итерируемся по слагаемым, записываем коэффициеты
    for term in terms:
        if isinstance(term, sp.sin):
            c.append(sp.Integer(1))
            n.append(term.args[0] / x)
        elif isinstance(term, sp.Mul):
            c.append(term.args[0])
            sin = term.args[1]
            n.append(sin.args[0]/x)

    return (c, n)


def quotient(a, f):
    """
    Находит частное решение уравнение теплопроводности

    Args:
        a (int): Параметр уравнения
        f (sympy.Function): Неоднородность уравнения

    Returns:
        sympy.Function: Частное решение
    """

    x, t = sp.symbols('x t')
    w = sp.Function('w')(t)

    # Разделяем функцию f(x, t) на f(x, t)=F(t)*G(x)
    f = _mul_as_two_parts(f)[0]
    F, G = (f[0], f[1]) if f[0].free_symbols == {t} else (f[1], f[0])

    # Находим коэффициент k^2
    k = - sp.diff(G, x, 2) / G

    # решаем дифур w'(t) = -a**2 * k**2 * w(t) + F(t)
    diff_eq = sp.Eq(sp.diff(w,t), -a*k*w + F)
    solution = sp.dsolve(diff_eq).rhs

    return solution.subs('C1', 0) * G


def solve_dirichlet(a, phi, f = None):
    """
    Решает задачу Дирихле уравнения теплопроводности

    Args:
        a (int): Параметр уравнения
        phi (sympy.Function): Граничное условие
        f (sympy.Function): Неоднородность уравнения

    Returns:
        sympy.Function: Решение уравнения
    """

    x, t = sp.symbols('x t')

    # Понизим степени тригонометрических функций 
    phi = TRpower(phi)

    # Находим общее решение
    c, n = get_coefficients(phi)
    answer = sp.Integer(0)
    for i in range(len(n)):
        answer += с[i]*sp.exp(-a*(n[i]**2)*t)*sp.sin(n[i]*x)

    # Находим частное решение
    if f != None:
        if f.func == sp.Add:
            for component in f.args:
                answer += quotient(a, component)
        else:
            answer += quotient(a, f)

    return answer
