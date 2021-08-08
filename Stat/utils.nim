import random
import arraymancer

proc partition[T: SomeFloat](list: var openarray[T], left, right, pivot_idx: int): int =
    let pivot_value = list[pivot_idx]
    swap(list[pivot_idx], list[right])
    var store_idx = left

    for i in left..right-1:
        if list[i] < pivot_value:
            swap(list[store_idx], list[i])
            store_idx += 1
    swap(list[right], list[store_idx])

    return store_idx

proc select*[T: SomeFloat](list: var openarray[T], left, right, k: int): T =
    randomize()
    var start = left
    var stop  = right
    while true:
        if start == stop:
            return list[start]

        var pivot_idx = rand(start..stop)
        pivot_idx = partition(list, start, stop, pivot_idx)

        if k == pivot_idx:
            return list[k]
        elif k < pivot_idx:
            stop = pivot_idx - 1
        else:
            start = pivot_idx + 1

proc eye*[T: SomeNumber](n: int): Tensor[T] =
    result = zeros[T](n, n)
    for i in 0..n-1:
        result[i,i] = 1.T

proc diag*[T: SomeNumber](x: Tensor[T]): Tensor[T] =
    var shape = x.shape
    if shape.len != 2:
        raise newException(ValueError, "Not implemented")
    else:
        if shape[0] != shape[1]:
            raise newException(ValueError, "The matrix is not square, shape = " & $shape)
    
    var diagonal = newSeq[T](shape[0].int)
    for i in 0..shape[0].int-1:
        diagonal[i] = x[i, i]

    return diagonal.toTensor()


proc dgetrf*(M   : ptr cint;
             N   : ptr cint;
             A   : ptr cdouble;
             lda : ptr cint;
             ipiv: ptr cint;
             info: ptr cint
             )
             {.importc: "dgetrf_", dynlib: "liblapack.so".}
 
proc dgetri*(N: ptr cint;
             a: ptr cdouble;
             lda: ptr cint;
             ipiv: ptr cint;
             work: ptr cdouble;
             lwork: ptr cint;
             info: ptr cint)
             {.importc: "dgetri_", dynlib: "liblapack.so".}

proc inv*(x: Tensor[float]): Tensor[float] =
    var input = clone(x)
    let shape = input.shape
    var N, M: cint
    if shape.len > 2:
        raise newException(ValueError, "Only square matrices are handled, shape = " & $shape)
    if shape.len == 2:
        if shape[0] != shape[1]:
            raise newException(ValueError, "The matrix is not square, shape = " & $shape)
        else:
            input = input.reshape(shape[0] * shape[1])
            N = shape[0].cint
            M = shape[0].cint
    if shape.len == 1:
        if sqrt(shape[0].float) != floor(sqrt(shape[0].float)):
            raise newException(ValueError, "The matrix is not square, shape = " & $shape)
        else:
            N = sqrt(shape[0].float).cint
            M = sqrt(shape[0].float).cint

    var 
        lda = N
        ipiv = newSeq[cint](N)
        info, info2 : cint
        lwork : cint = (2*N).cint
        work = newSeq[cdouble](lwork)
        
    dgetrf(addr N, addr M, addr input[0], addr lda, addr ipiv[0], addr info)
    dgetri(addr N, addr input[0], addr lda, addr ipiv[0], addr work[0], addr lwork, addr info2)

    if info2 != 0:
        raise newException(ValueError, "[LAPACK] dgetri error, code " & $info2)

    return input.reshape(N.int, N.int)

proc det*(x: Tensor[float]): float =
    var input = x.clone
    let shape = input.shape
    var N, M: cint
    if shape.len > 2:
        raise newException(ValueError, "Only square matrices are handled, shape = " & $shape)
    if shape.len == 2:
        if shape[0] != shape[1]:
            raise newException(ValueError, "The matrix is not square, shape = " & $shape)
        else:
            input = input.reshape(shape[0] * shape[1])
            N = shape[0].cint
            M = shape[0].cint
    if shape.len == 1:
        if sqrt(shape[0].float) != floor(sqrt(shape[0].float)):
            raise newException(ValueError, "The matrix is not square, shape = " & $shape)
        else:
            N = sqrt(shape[0].float).cint
            M = sqrt(shape[0].float).cint

    var 
        lda = N
        ipiv = newSeq[cint](N)
        info : cint
        
    dgetrf(addr N, addr M, addr input[0], addr lda, addr ipiv[0], addr info)
    
    var fixed : int = 0
    for i, i_piv in ipiv.pairs():
        if i == (i_piv - 1):
            fixed += 1
    #    echo i, " ", (i_piv - 1)

    #echo N - fixed

    var i = 0
    var u = 1.0
    while i < N:
        u *= input[i * (N + 1)]
        i += 1

    return -(2 * (fixed mod 2) - 1).float * u

proc trace*(x: Tensor[float]): float =
    var input = clone(x)
    let shape = input.shape
    var N, M: cint
    if shape.len > 2:
        raise newException(ValueError, "Only square matrices are handled, shape = " & $shape)
    if shape.len == 2:
        if shape[0] != shape[1]:
            raise newException(ValueError, "The matrix is not square, shape = " & $shape)
        else:
            input = input.reshape(shape[0] * shape[1])
            N = shape[0].cint
            M = shape[0].cint
    if shape.len == 1:
        if sqrt(shape[0].float) != floor(sqrt(shape[0].float)):
            raise newException(ValueError, "The matrix is not square, shape = " & $shape)
        else:
            N = sqrt(shape[0].float).cint
            M = sqrt(shape[0].float).cint

    var i = 0
    result = 0.0
    while i < N:
        result += input[(N + 1) * i]
        i += 1

proc rand*[T: SomeFloat](n = 1, d = 1): Tensor[T] =
    let size = n * d
    result = zeros[T](size)

    for i in 0..size-1:
        result[i] = rand(1.0)

    return result.reshape(n, d)

proc randn*[T: SomeFloat](n = 1, d = 1, mu = 0.0.T, sigma = 0.0.T): Tensor[T] =
    let size = n * d
    result = zeros[T](size)

    for i in 0..size-1:
        result[i] = gauss(mu, sigma)

    return result.reshape(n, d)
    
proc argmin*[T: SomeNumber](x: openarray[T]): int =
    let minimum = min(x)

    return find(x, minimum)

proc argmax*[T: SomeNumber](x: openarray[T]): int =
    let maximum = max(x)

    return find(x, maximum)

proc flatten*[T](x: Tensor[T]): Tensor[T] =
    let nbr_element = x.size.int

    return x.reshape(nbr_element)

proc argmin*[T: SomeNumber](x: Tensor[T]): int =
    let flatten_x = flatten(x)
    let minimum = min(flatten_x)

    return find(flatten_x, minimum)

proc argmax*[T: SomeNumber](x: Tensor[T]): int =
    let flatten_x = flatten(x)
    let maximum = max(flatten_x)

    return find(flatten_x, maximum)

proc `<`*[T: SomeFloat](x: Tensor[T]; k: T): Tensor[bool] =
    proc check(x: float): bool =
        return x < k

    return x.map(check)

proc `<=`*[T: SomeFloat](x: Tensor[T]; k: T): Tensor[bool] =
    proc check(x: float): bool =
        return x <= k

    return x.map(check)

proc `>`*[T: SomeFloat](x: Tensor[T]; k: T): Tensor[bool] =
    proc check(x: float): bool =
        return x > k

    return x.map(check)

proc `>=`*[T: SomeFloat](x: Tensor[T]; k: T): Tensor[bool] =
    proc check(x: float): bool =
        return x >= k

    return x.map(check)

proc `==`*[T: SomeFloat](x: Tensor[T]; k: T): Tensor[bool] =
    proc check(x: float): bool =
        return x == k

    return x.map(check)

proc `!=`*[T: SomeFloat](x: Tensor[T]; k: T): Tensor[bool] =
    proc check(x: float): bool =
        return x != k

    return x.map(check)
