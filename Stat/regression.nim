import arraymancer
import strutils
import distribution
import math
from std/fenv import epsilon
import stat
import utils
import test

type
    Fit[T] = object
        R2: T
        adj_R2: T
        params: Tensor[T]
        pvalues: Tensor[T]
        scores: Tensor[T]
        y_hat: Tensor[T]
        std_err: Tensor[T]
        conf_inter: Tensor[(T, T)]
        log_lik: T
        AIC: T
        BIC: T
        F_stat: T
        F_stat_pvalue: T

proc square[T: SomeNumber](x: T): T =
    x * x

proc OLS[T: SomeFloat](x_input, y: Tensor[T]; fit_intercept = true): Fit[T] =
    var x = x_input
    if fit_intercept:
        x = concat(ones[T](x_input.shape[0],1), x_input, axis=1)

    let N = x.shape[0].int
    let p = x.shape[1].int
    let df = (N - p).float
    let xx = inv(transpose(x) * x)
    result.params = xx * transpose(x) * y
    
    result.y_hat = x * result.params
    let residue = y - result.y_hat
    let sigma_hat2 = sum(residue.map(square)) / df
    let ybar = mean(y)
    let SS_tot = sum((y -. y_bar).map(square))
    let SS_res = sum((result.y_hat -. y_bar).map(square))
    result.R2 = SS_res / SS_tot
    result.adj_R2 = 1.0 - (1.0 - result.R2) * (N - 1).float / df
    result.log_lik = -0.5 * N.float * ln(2.0 * PI * sigma_hat2) - 0.5 * sum(residue.map(square)) / sigma_hat2

    result.AIC = (2 * (p)).float - 2.0 * result.log_lik
    result.BIC = (p).float * ln(N.float) - 2.0 * result.log_lik
    result.F_stat = SS_res / (p-1).float / sigma_hat2
    result.F_stat_pvalue = 1.0 - F_cdf(result.F_stat, (p-1).float, (N - p).float)

    result.std_err = sqrt(diag(xx * sigma_hat2)).reshape(p, 1)
    result.scores = result.params /. result.std_err
    #result.pvalues = 2.0 * (1.0 -. (abs(result.scores)).map(std_norm_cdf))
    result.pvalues = ones[float](p, 1)
    for i in 0..p-1:
        result.pvalues[i,0] = 1 - t_cdf(abs(result.scores[i,0]), df) + t_cdf(-abs(result.scores[i,0]), df)

    #let lower_bound = t_quantile(0.025, df)
    #let upper_bound = t_quantile(0.975, df)
    
type
    Fit_simple[T: SomeFloat] = object
        R2: T
        params: Tensor[T]

proc soft_threshold[T: SomeFloat](z, gamma: T): T =
    #echo z, " ", gamma, " ", max(abs(z) - gamma, 0.T)
    #return sgn(z).T * max(abs(z) - gamma, 0.T)
    if gamma < abs(z):
        if z > 0.T:
            return z - gamma
        if z < 0.T:
            return z + gamma
    else:
        return 0.T

proc soft_threshold[T: SomeFloat](z, gamma, N: T): T =
    #echo z, " ", gamma, " ", max(abs(z) - gamma, 0.T)
    return sgn(z).T * max(abs(z) - gamma, 0.T)
    #if gamma < abs(z):
    #    if z > 0.T:
    #        return z - gamma
    #    if z < 0.T:
    #        return z + gamma
    #else:
    #    return 0.T

proc elastic_net[T: SomeFloat](x_input, y: Tensor[T]; lambda: T; alpha: T; fit_intercept = true, active_set = true): Fit_simple[T] =
    var x = x_input
    if fit_intercept:
        x = concat(ones[T](x_input.shape[0],1), x_input, axis=1)

    let p = x.shape[1].int
    let N = x.shape[0].int
    var params = zeros[T](1, p)
    var old_params = clone(params)
    var params_zero = zeros[bool](1, p)
    let xx = sum(x *. x, axis=0) / N.float

    var residue: Tensor[T]
    while true:
        for p in 0..p-1:
            if params_zero[0, p] and active_set:
                continue
            residue = y - sum(x *. params, axis=1)
            params[0, p] = soft_threshold(mean(residue *. x[_,p]) + xx[0, p] * params[0, p], lambda * alpha) / (xx[0, p] + lambda * (1.0 - alpha))

            if params[0, p] == 0.0:
                params_zero[0, p] = true

        if sum(abs(params - old_params)) < epsilon(T):
            break
        old_params = clone(params)
        
    result.params = transpose(params)
    let y_hat = x * result.params
    result.R2 = variance(y_hat) / variance(y)


proc ridge_regression[T: SomeFloat](x_input, y: Tensor[T]; lambda: T; fit_intercept = true, analytical = true): Fit_simple[T] =
    if analytical:
        var x = x_input
        if fit_intercept:
            x = concat(ones[T](x_input.shape[0],1), x_input, axis=1)

        let N = x.shape[0].int
        let p = x.shape[1].int
        let xx_lambda = inv(transpose(x) * x + N.float * lambda * eye[T](p))
        result.params = xx_lambda * transpose(x) * y
        let y_hat = x * result.params
        let ybar = mean(y)
        let SS_tot = sum((y -. y_bar).map(square))
        let SS_res = sum((y_hat -. y_bar).map(square))
        result.R2 = SS_res / SS_tot

    else:
        result = elastic_net(x_input, y, lambda, 0.0, fit_intercept=true, active_set=false)
    

var data : seq[float]
var file_data = readFile("clean_data.dat").split("\n")[1..^2]

for line in file_data:
    for sub_line in line.split(","):
        data.add(parseFloat(sub_line))

var x = data.toTensor().reshape(67, 9)
var y = x[_,^1]
#x = concat(ones[float](x.shape[0],1), x[_,0..^2], axis=1)
var x_stand = (x -. x.mean(axis = 0)) /. x.std(axis = 0)
let x_centered = x -. x.mean(axis = 0)

var res_EN = elastic_net(x[_,0..^2], y, 0.1, 0.5, fit_intercept=true)
var res = OLS(x[_,0..^2], y, fit_intercept=true)
echo res_EN.params, " ", sum(abs(res_EN.params))
echo res_EN.R2
echo res.params.transpose()
echo ridge_regression(x[_,0..^2], y, 0.1, analytical=false)
echo ridge_regression(x[_,0..^2], y, 0.1, analytical=true)
echo lr_test(x, [1.313491552656717, 3.626107686567165, 60.74626865671642, 0.07143990820895517, 0.2238805970149254, -0.2142030095522391, 6.731343283582089, 26.26865671641791, 2.452345085074628].toTensor().reshape([1,9]), transpose(x_centered) * x_centered / x.shape[0].float, false, false)
echo hotelling_t2(x, [1.313491552656717, 3.626107686567165, 60.74626865671642, 0.07143990820895517, 0.2238805970149254, -0.2142030095522391, 6.731343283582089, 26.26865671641791, 2.452345085074628].toTensor().reshape([1,9]), transpose(x_centered) * x_centered / x.shape[0].float, false)
echo pca_test([8.64710906e+02, 5.20089156e+01, 1.81608597e+00, 1.99032426e+00, 4.65890572e-01, 8.24428054e-02, 1.40346813e-01, 2.16017221e-01].toTensor(), 100, 8, 5)
#echo res.std_err
#echo res.scores
#echo res.pvalues
#echo res.R2
#echo res.adj_R2
#echo res.log_lik
#echo res.AIC
#echo res.BIC
#echo res.F_stat
#echo res.F_stat_pvalue
#echo F_cdf(1.1, 9.0 , 2.0)

#var res = inv(a)
#echo res
#echo a.reshape(10, 10)
#
#echo eye[float](3)
