import distribution
import arraymancer
import utils

type
    Test[T: SomeFloat] = tuple
        stat: T
        pvalue: T

proc hotelling_t2*[T: SomeFloat](x, mu0, sigma: Tensor[T], sigma_known = true): Test[T] =
    let n = x.shape[0].int
    let p = x.shape[1].int
    let xbar = mean(x, axis = 0)

    if sigma_known:
        result.stat = (n.float * (xbar - mu0) * inv(sigma) * transpose(xbar - mu0))[0, 0]
        result.pvalue = 1.0 - chi2_cdf(result.stat, p.float)
    else:
        result.stat = (n - 1).float * ((xbar - mu0) * inv(sigma) * transpose(xbar - mu0))[0, 0]
        result.pvalue = 1.0 - F_cdf(result.stat * (n - p).float / (p * (n - 1)).float, p.float, (n-p).float)

proc lr_test*[T: SomeFloat](x, mu0, sigma: Tensor[T], test_mu = true, sigma_knwown = true): Test[T] =
    let n = x.shape[0].int
    let p = x.shape[1].int
    let xbar = mean(x, axis = 0)
    echo xbar.shape
    echo sigma.shape
    if test_mu:
        if sigma_knwown:
            result.stat = (n.float * (xbar - mu0) * inv(sigma) * transpose(xbar - mu0))[0, 0]
            result.pvalue = 1.0 - chi2_cdf(result.stat, p.float)
        else:
            result.stat = (n.float * ln(1.0 +. (xbar - mu0) * inv(sigma) * transpose(xbar - mu0)))[0, 0]
            result.pvalue = 1.0 - chi2_cdf(result.stat, p.float)
    else:
        var S = transpose(x -. xbar) * (x -. xbar) / n.float
        let Sigma_1_S = inv(sigma) * S
        result.stat = (n.float * (trace(Sigma_1_S) - ln(det(Sigma_1_S)) - p.float))
        echo "sskljd: ", p.float * (p + 1).float * 0.5
        result.pvalue = 1.0 - chi2_cdf(result.stat, p.float * (p + 1).float / 2.0)

proc pca_test*[T: SomeFloat](lambdas: Tensor[T], n, p, k: int, sorted = false): Test[T] =
    result.stat = 1.0
    if max(lambdas.shape) != p:
        raise newException(ValueError, "Number of eigenvalues is different from p " & $max(lambdas.shape) & "!=" & $p)
    if k > p - 1:
        raise newException(ValueError, "Number of eigenvalues to be tested is larger or equal than the number of eigenvalues " & $k & ">=" & $p)

    let l_bar = mean(lambdas.reshape(p)[k..^1])
    let g = product(lambdas.reshape(p)[k..^1]) / pow(l_bar, (p - k).float)

    result.stat = -(n * p).float * ln(g)
    result.pvalue = 1.0 - chi2_cdf(result.stat, ((p - k + 2) * (p - k - 1)).float / 2.0)
