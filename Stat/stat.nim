import math
from std/fenv import epsilon
import algorithm
import utils
import arraymancer

proc percentile*[T: SomeFloat](x: openarray[T]; perc: float): T =
    ## Compute the perc-th percentile of x where perc is given in percent.
    let sorted_x = sorted(x, system.cmp)
    if perc == 100.0:
        return sorted_x[^1]
    if perc == 0.0:
        return sorted_x[0]

    let n = len(x).float
    let idx = floor((n - 1) * perc / 100.0)
    let fraction = (n - 1) * perc / 100.0 - idx

    result = sorted_x[idx.int]*(1.0 - fraction) + sorted_x[idx.int + 1]*(fraction)

proc median*[T: SomeFloat](x: openarray[T]): T =
    ## Compute the median of x 
    ## Linear interpolation is used when the position of the
    ## median is not an integer.
    result = percentile(x, 50.0)

proc mean*[T: SomeFloat](x: openarray[T]): T =
    ## Compute the mean of x.
    result = 0.0
    let n = len(x).float
    for i in low(x)..high(x):
        result = result + (x[i].float)

    result /= n

proc trimmed_mean*[T: SomeFloat](x: openarray[T], cut: int): T =
    ## Compute the trimmed mean of x by removing cut elements 
    ## from both end of the vector
    if (2*cut > len(x)):
        raise newException(ValueError, "Too many entries are cut")
    result = mean(x[cut..^(cut+1)])

proc variance[T: SomeFloat](x: openarray[T], ddof = 0.0): float =
    result = 0.0
    var sum2 = 0.0
    let n = len(x).float
    for i in low(x)..high(x):
        sum2 += (x[i] * x[i]).float
        result += x[i].float

    result = n/(n-ddof) * (sum2 / n - result * result / (n*n))

proc std[T: SomeFloat](x: openarray[T], ddof = 0.0): float =
    result = sqrt(variance(x, ddof))

proc mad[T: SomeFloat](x: openarray[T], factor: T = 1.0): T =
    let med = median(x)
    var dev : seq[T]
    for i in low(x)..high(x):
        dev.add( abs(x[i] - med))

    result = median(dev) / factor

type
    Ranks = tuple
        index: int
        value: float

proc my_cmp(x, y: Ranks): int =
    if x.value < y.value: -1
    elif x.value == y.value: 0
    else: 1

proc rank[T: SomeFloat](x: openarray[T]): seq[int] =
    var pairs: seq[Ranks]

    for i in low(x)..high(x):
        pairs.add((index: i, value: x[i]))

    pairs.sort(my_cmp)
    
    for i in low(pairs)..high(pairs):
        result.add(pairs[i].index)

type
    Summary = object
        min: float
        p25: float
        med: float
        p75: float
        max: float

proc summary[T: SomeFloat](x: openarray[T]): Summary =
    result = Summary(min: min(x), p25: percentile(x,25.0), med: median(x), p75: percentile(x,75.0), max: max(x))

proc skewness*[T: SomeFloat](x: openarray[T], bias = true): float =
    let mu = mean(x)
    var m3 = 0.0
    var m2 = 0.0
    for i in low(x)..high(x):
        m3 += (x[i] - mu)^3
        m2 += (x[i] - mu)^2

    let g1 = m3 / pow(m2, 1.5)
    if bias:
        result = g1
    else:
        let N = len(x).float
        result = sqrt(N * (N - 1.0)) / N * g1

proc quartile_skewness[T: SomeFloat](x: openarray[T]): T =
    let q1 = percentile(x, 25.0)
    let q2 = percentile(x, 50.0)
    let q3 = percentile(x, 75.0)

    result = (q3 + q1 - 2.0 * q2) / (q3 - q1)

proc octile_skewness[T: SomeFloat](x: openarray[T]): T =
    let q05 = percentile(x, 12.5)
    let q2  = percentile(x, 50.0)
    let q25 = percentile(x, 87.5)

    result = (q25 + q05 - 2.0 * q2) / (q25 - q05)

proc kurtosis[T: SomeFloat](x: openarray[T]): float =
    let mu = mean(x)
    var m2 = 0.0
    var m4 = 0.0
    for i in low(x)..high(x):
        m2 += (x[i] - mu)^2
        m4 += (x[i] - mu)^4

    let N = len(x).float

    result = N * m4 / m2^2

proc robust_kurtosis[T: SomeFloat](x: openarray[T]): float =
    let q125 = percentile(x, 12.5)
    let q25  = percentile(x, 25.0)
    let q375 = percentile(x, 37.5)
    let q625 = percentile(x, 62.5)
    let q75  = percentile(x, 75.0)
    let q875 = percentile(x, 87.5)

    result = ((q875 - q625) + (q375 - q125)) / (q75 - q25)

type 
    Test = object
        stat: float
        prob: float

proc jarque_bera[T: SomeFloat](x: openarray[T]): Test =
    let N = len(x).float
    let S = skewness(x)
    let K = kurtosis(x)
    let jb = N / 6.0 * (S^2 + 0.25*(K - 3)^2)
    result = Test(stat: jb, prob: 1 - exp(-0.5*jb))

proc weighted_mean[T: SomeFloat](x, w: openarray[T]): T =
    var sum_w = 0.0
    for i in low(w)..high(w):
        sum_w += w[i]
        result += w[i] * x[i]

    result /= sum_w

type
    Pairs = tuple
        weight: float
        value: float     

proc my_cmp(x, y: Pairs): int =
    if x.value < y.value: -1
    elif x.value == y.value: 0
    else: 1

proc weighted_median[T: SomeFloat](x: openarray[T], w: openarray[T]): T =
    var sum_w = 0.0
    for w_i in w.items:
        sum_w += w_i

    var pairs : seq[Pairs]
    for i in low(x)..high(x):
        pairs.add((weight: w[i], value: x[i]))

    pairs.sort(my_cmp)

    var running_sum_w = 0.0
    var pn = 0.0
    var pn_previous = 0.0
    var i : int = 0
    while pn < 0.5:
        running_sum_w += w[i]
        pn_previous  = pn
        pn = (running_sum_w - w[i]) / (sum_w - w[i])
        i += 1

    result = pairs[i-2].value + (pairs[i-1].value - pairs[i-2].value)/(pn - pn_previous) * (0.5 - pn_previous)
     
proc my_cmp_decreasing[T: SomeFloat](x, y: T): int =
    if x < y: 1
    elif x > y: -1
    else: 0

proc h[T: SomeFloat](Zp, Zm: openarray[T], p, i, j: int): T =
    let a = Zp[i]
    let b = Zm[j]
    if abs(a - b) <= 2 * epsilon(float):
        return sgn(p - 1 - i - j).float
    else:
        return (a + b) / (a - b)

proc medcouple_slow[T: SomeFloat](x: openarray[T]): T =
    var x_sorted = sorted(x, my_cmp_decreasing)

    let x_median = median(x_sorted)

    var Zp : seq[T]
    var Zm : seq[T]

    #var scale : T
    #if x[0] - x_median > x_median - x[^1]:
    #    scale = 2 * (x[0] - x_median)
    #else:
    #    scale = 2 * (x_median - x[^1])

    for x_i in x_sorted.items:
        if x_i >= x_median:
            Zp.add((x_i - x_median))# / scale)
        if x_i <= x_median:
            Zm.add((x_i - x_median))# / scale)

    let p = len(Zp)
    let q = len(Zm)

    var H : seq[T]
    
    for i, zm in Zm.pairs:
        for j, zp in Zp.pairs:
            H.add(h(Zp, Zm, p, i, j))

    result = median(H)

proc greater_h[T: SomeFloat](Zp, Zm: openarray[T], p, q: int, u: float): seq[int] =
    var j = 0
    var P = newSeq[int](p)
    for i in countdown(p-1, 0):
        while (j < q) and (h(Zp, Zm, p, i, j) > u):
            j += 1  
        P[i] = j - 1
    return P

proc lower_h[T: SomeFloat](Zp, Zm: openarray[T], p, q: int, u: float): seq[int] =
    var j = q - 1
    var Q = newSeq[int](p)

    for i in countup(0, p-1):
        while (j >= 0) and (h(Zp, Zm, p, i, j) < u):
            j = j - 1
        Q[i] = j + 1

    return Q

proc medcouple[T: SomeFloat](x: openarray[T]): T =
    if len(x) < 3:
        return 0

    var x_sorted = sorted(x, my_cmp_decreasing)

    let x_median = median(x_sorted)

    var scale : T
    if x[0] - x_median > x_median - x[^1]:
        scale = 2 * (x[0] - x_median)
    else:
        scale = 2 * (x_median - x[^1])

    var Zp : seq[T]
    var Zm : seq[T]

    for x_i in x_sorted.items:
        if x_i >= x_median:
            Zp.add((x_i - x_median) / scale)
        if x_i <= x_median:
            Zm.add((x_i - x_median) / scale)
    
    let p = len(Zp)
    let q = len(Zm)

    var L : seq[int]
    var R : seq[int]

    for i in 0..p-1:
        L.add(0)
        R.add(q-1)

    var L_tot = 0
    var R_tot = p * q

    var medcouple_idx = floor(R_tot / 2).int
    
    while R_tot - L_tot > p:
        var row_medians : seq[T]
        var weights : seq[T]
        for i in 0..p-1:
            if L[i] <= R[i]:
                row_medians.add(h(Zp, Zm, p, i, ((L[i] + R[i])/2).int))
                weights.add( (R[i] - L[i] + 1).float )

        let wm = weighted_median(row_medians, weights)

        let P = greater_h(Zp, Zm, p, q, wm)
        let Q = lower_h(Zp, Zm, p, q, wm)

        let P_tot: int = sum(P) + len(P)
        let Q_tot: int = sum(Q)

        if medcouple_idx <= P_tot - 1:
            R = P
            R_tot = P_tot
        else:
            if medcouple_idx > Q_tot - 1:
                L = Q
                L_tot = Q_tot
            else:
                return wm

    var remaining : seq[T]
    for i in 0..p-1:
        for j in L[i]..R[i]:
            remaining.add(-h(Zp, Zm, p, i,j))
    
    if abs(medcouple_idx.float - R_tot.float / 2.0) != 0:
        return (-select(remaining, 0, len(remaining)-1, (medcouple_idx - L_tot).int)) * 0.5 + (-select(remaining, 0, len(remaining)-1, (medcouple_idx - L_tot - 1).int)) * 0.5
    else:
        return -select(remaining, 0, len(remaining)-1, (medcouple_idx - L_tot).int)

proc clamp[T: SomeNumber](x, lower, upper: T): T =
    if x > upper:
        return upper
    elif x < lower:
        return lower
    else:
        return x

proc dclamp_dx[T: SomeNumber](x, lower, upper: T): T =
    if x > upper:
        return 0
    elif x < lower:
        return 0
    else:
        return 1

proc location_huber[T: SomeFloat](x: openarray[T], k = 1.345): (T, T, T) =
    var beta = median(x)
    var old_beta = beta
    let scale = mad(x, 0.67449)
    let n = len(x).float
    var w = newSeq[T](n.int)
    while true:

        for i in low(x)..high(x):
            var r = (x[i] - beta) / scale
            if r == 0.0:
                w[i] = 1.0
            else:
                w[i]  = clamp[T](r, -k, k) / r
        
        old_beta = beta
        beta = weighted_mean(x, w)
        
        if abs(beta - old_beta) < epsilon(T):
            result[0] = beta
            break

    var sum2 = 0.0
    var sump = 0.0
    for i in low(x)..high(x):
        var r = (x[i] - result[0]) / scale
        sum2 += pow(clamp[T](r, -k, k), 2.0)
        sump += dclamp_dx[T](r, -k, k)

    let tau = sum2 / n / pow(sump / n, 2.0)
    result[1] = tau
    result[2] = scale

#when isMainModule:
#    echo mean(sample)
#    echo "Mean: " & $is_close(mean(sample), -0.1368128423, 1e-12)
#    echo "Variance pop: " & $is_close(variance(sample), 0.9655833209435284, 1e-12)
#    echo "Stdev pop: " & $is_close(std(sample), 0.9826409929081569, 1e-12)
#    echo "Stdev sample: " & $is_close(std(sample, 1), 0.9875913566965568, 1e-12)
#    echo skewness(sample)
#    echo jarque_bera(sample)
#    echo percentile(sample, 0.0)
#    echo summary(sample)
#    echo trimmed_mean([1.0, 2.0, 3.0, 4.0, 5.0], 0)
#    echo mad(sample)
#    echo rank(sample)
#    #echo weighted_median([2.0, 1.0, 3.0, 4.0, 5.0], [2.0, 1.0, 3.0, 1.0 , 0.5])
#    echo weighted_median([1.0, 2.0, 3.5, 4.5, 5.5, 6.5, 7.5, 12, 13, 14], [0.1, 123.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
#    echo "Slow ", medcouple_slow([1.0, 2.0, 3.0, 4.0, 5.0])
#    echo medcouple([1.0, 2.0, 3.0, 4.0, 5.0])
#    echo "Slow ", medcouple_slow(sample)
#    echo medcouple(sample)
#    echo mean([1.2, 2.4, 1.3, 1.3, 0.0, 1.0, 1.8, 0.8, 4.6, 1.4])
#    echo mad([1.2, 2.4, 1.3, 1.3, 0.0, 1.0, 1.8, 0.8, 4.6, 1.4], 0.6745)
#    echo location_huber([1.2, 2.4, 1.3, 1.3, 0.0, 1.0, 1.8, 0.8, 4.6, 1.4])
