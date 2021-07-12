import math
import gamma
from std/fenv import epsilon
import arraymancer
import utils

proc unif_pdf*(x: float, a = 0.0, b = 1.0): float =
    if a > b:
        raise newException(ValueError, "left bound a must be smaller than right bound b")
    result = 0.0
    if x > a and x < b:
        result = 1.0 / (b - a)

proc unif_cdf*(x:float, a = 0.0, b = 1.0): float =
    if a > b:
        raise newException(ValueError, "left bound a must be smaller than right bound b")
    result = 0.0
    if x > a and x < b:
        result = x / (b - a)

type
    Theta*[T] = tuple
        mu: Tensor[T]
        sigma: Tensor[T]

proc norm_pdf*(x: float, mu = 0.0, sigma = 1.0): float =
    if sigma < 0:
        raise newException(ValueError, "sigma must be greater than 0")
    const factor = 1.0 / sqrt(2.0 * PI)
    let z = (x - mu) / sigma

    result = factor / sigma * exp(-0.5 * z * z)

proc norm_pdf*[T: SomeFloat](x, mu, sigma: Tensor[T]): Tensor[T] =
    let d = sigma.shape[0].T
    if x.shape[0].T == d:
        return exp(-0.5 * transpose(x - mu) * inv(sigma) * (x - mu)) / sqrt(pow(2*PI, d) * abs(det(sigma)).T)
    return exp(-0.5 * (x - mu) * inv(sigma) * transpose(x - mu)) / sqrt(pow(2*PI, d) * abs(det(sigma)).T)

proc norm_pdf*[T: SomeFloat](x: Tensor[T]; param: Theta[T]): T =
    let d = param.sigma.shape[0].T
    if x.shape[0].T == d:
        let z = transpose(x - param.mu) * inv(param.sigma) * (x - param.mu)
        return exp(-0.5 * z[0, 0]) / sqrt(pow(2*PI, d) * abs(det(param.sigma)).T)
    let z = (x - param.mu) * inv(param.sigma) * transpose(x - param.mu)
    return exp(-0.5 * z[0, 0]) / sqrt(pow(2*PI, d) * abs(det(param.sigma).T))

proc norm_pdf*[T: SomeFloat](x: Tensor[T]; params: Tensor[Theta[T]]): Tensor[T] =
    let p = params.shape[0].int
    result = zeros[T](1, p)
    
    for i in 0..p-1:
        result[0, i] = norm_pdf(x, params[i])

proc norm_pdf*[T: SomeFloat](x: T, mu, sigma: Tensor[T]): Tensor[T] =
    const factor = 1.0 / sqrt(2.0 * PI)
    let z = (mu -. x) /. sigma

    result = factor /. sigma * exp(-0.5 * z *. z)

proc norm_cdf*[T: SomeFloat](x: T, mu = 0.0, sigma = 1.0): float =
    if sigma < 0:
        raise newException(ValueError, "sigma must be greater than 0")
    let z = (x - mu) / sigma

    result = 0.5 * erfc(- z / sqrt(2.0))

proc std_norm_cdf*[T: SomeFloat](x: T): float =
    result = norm_cdf(x, 0.0, 1.0)

proc gamma_pdf(x, k: float, scale = 1.0): float =
    if k < 0:
        raise newException(ValueError, "k must be greater than 0")
    if scale < 0:
        raise newException(ValueError, "scale must be greater than 0")
    result = 1.0 / (gamma(k) * pow(scale, k)) * pow(x, (k - 1.0)) * exp(-x / scale)

proc gamma_cdf(x, k: float, scale = 1.0): float =
    if k < 0:
        raise newException(ValueError, "k must be greater than 0")
    if scale < 0:
        raise newException(ValueError, "scale must be greater than 0")
    result = gamma_p[float](k, x / scale)

proc chi2_pdf*(x, k: float): float =
    if k < 0:
        raise newException(ValueError, "k must be greater than 0")
    result = gamma_pdf(x, k / 2.0, scale = 2.0)

proc chi2_cdf*(x, k: float): float =
    if k < 0:
        raise newException(ValueError, "k must be greater than 0")
    result = gamma_cdf(x, k / 2.0, scale = 2.0)

proc t_pdf*(x, nu: float): float =
    if nu <= 0:
        raise newException(ValueError, "nu must be greater than 0")

    result = gamma((nu + 1.0)/2.0) * pow(1.0 + x*x/nu, -(nu + 1.0)/2.0) / (sqrt(nu * PI) * gamma(nu/2.0))

proc t_cdf*(x, nu: float): float =
    if nu <= 0:
        raise newException(ValueError, "nu must be greater than 0")

    result = 0.5 + x * gamma((nu + 1.0)/2.0) * hypergeometric_func(0.5, (nu + 1.0)/2.0, 1.5, -x*x/nu) / (sqrt(PI * nu) * gamma(nu/2.0))

#proc t_quantile*(p, nu: float): float =
#    if p < 0.0 or p > 1.0:
#        raise newException(ValueError, "p must be in (0, 1), p = " & $p)
#
#    if nu == 1.0:
#        return tan(PI * (p - 0.5))
#    if nu == 2.0:
#        let alpha = 4 * p * (1.0 - p)
#        return 2 * (p - 0.5) * sqrt(2.0 / alpha)
#    if nu == 4.0:
#        let alpha = 4 * p * (1.0 - p)
#        let q = cos(1.0 / 3.0 * cos(sqrt(alpha))) / sqrt(alpha)
#
#        return sgn(p - 0.5).float * 2.0 * sqrt(q - 1.0)
#    
#    var next = -1.0
#    var previous = 0.0
#
#    while abs(next - previous) > epsilon(float):
#        previous = next
#        next = previous + (p - t_cdf(previous, nu)) / t_pdf(previous, nu)
#        echo next, " ", previous
#
#    return previous

#{.compile: "r_t_qt.c".}
#proc qt(p: cdouble; ndf: cdouble; lower_tail: cint; log_pi: cint): cdouble {.importc.}
#
#proc t_quantile*(p, nu: float): float =
#    if p < 0.0 or p > 1.0:
#        raise newException(ValueError, "p must be in (0, 1), p = " & $p)
#
#    return qt(p.cdouble, nu.cdouble, 1.cint, 0.cint)

proc F_pdf*(x, d1, d2: float): float =
    result = sqrt(pow(d1*x, d1)*pow(d2, d2) / pow(d1*x + d2, d1 + d2)) / (x * beta_func(d1/2.0, d2/2.0))

proc F_cdf*(x, d1, d2: float): float =
    result = reg_inc_beta_func(d1 * x / (d1 * x + d2), d1/2.0, d2/2.0)
