"""Utility module of reusable algebra routines."""
import math
import jax.numpy as jnp

# import geomstats.backend as gs
gs = jnp

EPSILON = 1e-6
COS_TAYLOR_COEFFS = [
    1.0,
    -1.0 / math.factorial(2),
    +1.0 / math.factorial(4),
    -1.0 / math.factorial(6),
    +1.0 / math.factorial(8),
]
SINC_TAYLOR_COEFFS = [
    1.0,
    -1.0 / math.factorial(3),
    +1.0 / math.factorial(5),
    -1.0 / math.factorial(7),
    +1.0 / math.factorial(9),
]
INV_SINC_TAYLOR_COEFFS = [1, 1.0 / 6.0, 7.0 / 360.0, 31.0 / 15120.0, 127.0 / 604800.0]
INV_TANC_TAYLOR_COEFFS = [1.0, -1.0 / 3.0, -1.0 / 45.0, -2.0 / 945.0, -1.0 / 4725.0]
COSC_TAYLOR_COEFFS = [
    1.0 / 2.0,
    -1.0 / math.factorial(4),
    +1.0 / math.factorial(6),
    -1.0 / math.factorial(8),
    +1.0 / math.factorial(10),
]
VAR_INV_TAN_TAYLOR_COEFFS = [1.0 / 12.0, 1.0 / 720.0, 1.0 / 30240.0, 1.0 / 1209600.0]
SINHC_TAYLOR_COEFFS = [
    1.0,
    1 / math.factorial(3),
    1 / math.factorial(5),
    1 / math.factorial(7),
    1 / math.factorial(9),
]
LOG_SINHC_TAYLOR_COEFFS = [
    0.0,
    1 / 6,
    1 / 180,
    1 / 2835,
    1 / 37800,
]
COSH_TAYLOR_COEFFS = [
    1.0,
    1 / math.factorial(2),
    1 / math.factorial(4),
    1 / math.factorial(6),
    1 / math.factorial(8),
]
INV_SINHC_TAYLOR_COEFFS = [
    1.0,
    -1.0 / 6.0,
    7.0 / 360.0,
    -31.0 / 15120.0,
    127.0 / 604800.0,
]
INV_TANH_TAYLOR_COEFFS = [1.0, 1.0 / 3.0, -1.0 / 45.0, 2.0 / 945.0, -1.0 / 4725.0]
TANH_TAYLOR_COEFFS = [1.0, -1.0 / 3.0, 2.0 / 15.0, -17.0 / 315.0, 62.0 / 2835.0]
LOG_TANH_TAYLOR_COEFFS = [0.0, -1.0 / 3.0, 7.0 / 90.0, -62.0 / 2835.0, 127.0 / 18900.0]
LOG1P_M_TANH_SQ_TAYLOR_COEFFS = [0.0, -1.0, 1.0 / 6.0, -2.0 / 45.0, 17.0 / 1260.0]
ARCTANH_CARD_TAYLOR_COEFFS = [1.0, 1.0 / 3.0, 1.0 / 5.0, 1 / 7.0, 1.0 / 9]


cos_close_0 = {"function": gs.cos, "coefficients": COS_TAYLOR_COEFFS}
sinc_close_0 = {"function": lambda x: gs.sin(x) / x, "coefficients": SINC_TAYLOR_COEFFS}
inv_sinc_close_0 = {
    "function": lambda x: x / gs.sin(x),
    "coefficients": INV_SINC_TAYLOR_COEFFS,
}
inv_tanc_close_0 = {
    "function": lambda x: x / gs.tan(x),
    "coefficients": INV_TANC_TAYLOR_COEFFS,
}
cosc_close_0 = {
    "function": lambda x: (1 - gs.cos(x)) / x**2,
    "coefficients": COSC_TAYLOR_COEFFS,
}
var_sinc_close_0 = {
    "function": lambda x: (x - gs.sin(x)) / x**3,
    "coefficients": [-k for k in SINC_TAYLOR_COEFFS[1:]],
}
var_inv_tanc_close_0 = {
    "function": lambda x: (1 - (x / gs.tan(x))) / x**2,
    "coefficients": VAR_INV_TAN_TAYLOR_COEFFS,
}
sinch_close_0 = {
    "function": lambda x: gs.sinh(x) / x,
    "coefficients": SINHC_TAYLOR_COEFFS,
}
log_sinch_close_0 = {
    "function": lambda x: gs.log(gs.sinh(x) / x),
    "coefficients": LOG_SINHC_TAYLOR_COEFFS,
}
cosh_close_0 = {"function": gs.cosh, "coefficients": COSH_TAYLOR_COEFFS}
inv_sinch_close_0 = {
    "function": lambda x: x / gs.sinh(x),
    "coefficients": INV_SINHC_TAYLOR_COEFFS,
}
inv_tanh_close_0 = {
    "function": lambda x: x / gs.tanh(x),
    "coefficients": INV_TANH_TAYLOR_COEFFS,
}
tanh_close_0 = {
    "function": lambda x: gs.tanh(x) / x,
    "coefficients": TANH_TAYLOR_COEFFS,
}
log_tanh_close_0 = {
    "function": lambda x: gs.log(gs.tanh(x) / x),
    "coefficients": LOG_TANH_TAYLOR_COEFFS,
}
log1p_m_tanh_sq_close_0 = {
    "function": lambda x: gs.log1p(-gs.tanh(x) ** 2),
    "coefficients": LOG1P_M_TANH_SQ_TAYLOR_COEFFS,
}
arctanh_card_close_0 = {
    "function": lambda x: gs.arctanh(x) / x,
    "coefficients": ARCTANH_CARD_TAYLOR_COEFFS,
}


def taylor_exp_even_func(point, taylor_function, order=5, tol=EPSILON):
    """Taylor Approximation of an even function around zero.

    Parameters
    ----------
    point : array-like
        Argument of the function to approximate.
    taylor_function : dict with following keys
        function : callable
            Even function to approximate around zero.
        coefficients : list
            Taylor coefficients of even order at zero.
    order : int
        Order of the Taylor approximation.
        Optional, Default: 5.
    tol : float
        Threshold to use the approximation instead of the function's value.
        Where `abs(point) <= tol`, the approximation is returned.

    Returns
    -------
    function_value: array-like
        Value of the function at point.
    """
    approx = gs.einsum(
        "k,k...->...",
        gs.array(taylor_function["coefficients"][:order]),
        gs.array([point**k for k in range(order)]),
    )
    point_ = gs.where(gs.abs(point) <= tol, tol, point)
    exact = taylor_function["function"](gs.sqrt(point_))
    result = gs.where(gs.abs(point) < tol, approx, exact)
    return result