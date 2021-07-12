import math
from std/fenv import epsilon

proc gamma_p*[T: SomeFloat](a, z: T): T =
    if a == 2.0:
        return 1.0 - exp(-0.5 * z)

    if z == 0.0:
        return 0.0

    let factor = pow(z, a) * exp(-z)
    var k = 0.0

    var sum = 0.0
    var next = 1.0 / gamma(a + 1)
    while (next > sum * epsilon(T)):
        next = pow(z, k) / gamma(a + k + 1)
        sum += next
        k += 1.0

    result = factor * sum

#proc gamma_p_z[T: SomeFloat](a, z: T): T =
#    result = exp(-z) * pow(z, a - 1.0) / gamma(a)

proc hypergeometric_func*[T: SomeFloat](a, b, c, z: T): T =
    #let factor = exp(lgamma(c) - (lgamma(a) + lgamma(b)))
    result = 0.0
    #var next = 1.0
    var next_frac = 1.0
    var n : T  = 0.0
    var lfac = 0.0
    var frac = 1.0
    while (abs(next_frac) > result * epsilon(T)):
        #next = exp(lgamma(a + n) + lgamma(b + n) - (lgamma(c + n) + lfac)) * pow(z, n)
        next_frac = frac * pow(z, n) / exp(lfac)
        frac *= (a + n) * (b + n) / (c + n)
        result += next_frac
        n += 1.0
        lfac += ln(n)

    #result *= factor

proc d_hypergeometric_func_dz*[T: SomeFloat](a, b, c, z: T): T =
    return a * b / c * hypergeometric_func(a + 1.0, b + 1.0, c + 1.0, z)

proc beta_func*[T: SomeFloat](x, y: T): T =
    result = exp(lgamma(x) + lgamma(y) - lgamma(x + y))

proc reg_inc_beta_func*[T: SomeFloat](z, a, b: T): T =
    result = pow(z, a) / a * hypergeometric_func(a, 1.0 - b, a + 1.0, z) / beta_func(a, b)

if isMainModule:
    var x = 0.0
    while x < 20:
        let y : float = gamma_p(0.1, 0.2)
        echo y
        x += 0.1
    echo epsilon(float)
