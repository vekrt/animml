import arraymancer
import utils
import strutils

type 
    Euclidean* = object
    Manhattan* = object
    Single_linkage* = object
    Complete_linkage* = object
    Weighted_linkage* = object

    Node_kind* = enum
     Leaf, Branch

    Node*[T] = ref object
        case kind*: Node_kind
        of Branch:
            children*: seq[Node[T]]
            distance*: T
        of Leaf:
            index*: int

proc disp[T: SomeNumber](node: Node[T], level = 0): void =
    let shift = "\t".repeat(level)
    if node.kind == Leaf:
        echo shift, "Index: ", node.index
    else:
        echo shift, "Distance: ", node.distance
        for c in node.children:
            disp(c, level + 1)

proc pairwise[T: SomeFloat](metric: typedesc[Euclidean]; x,y: Tensor[T]): T =
    let diff = x - y

    return sqrt(sum(diff *. diff))

proc pairwise[T: SomeFloat](metric: typedesc[Manhattan]; x,y: Tensor[T]): T =
    let diff = x - y

    return sum(abs(diff))

proc distance_matrix[T: SomeFloat](metric: typedesc[Euclidean]; X: Tensor[T]): Tensor[T] =
    let n = X.shape[0].int

    result = zeros[T](n, n)

    for i in 0..(n-1):
        for j in i+1..(n-1):
            let dist = metric.pairwise(X[i,_], X[j,_])
            result[i, j] = dist
            result[j, i] = dist

proc find_min[T: SomeFloat](dist_matrix: Tensor[T]): seq[int] =
    let n = dist_matrix.shape[0].int

    var minimum = 1e30
    for i in 0..(n-1):
        for j in i+1..(n-1):
            if dist_matrix[i, j] < minimum:
                minimum = dist_matrix[i, j]
                result = @[i, j]
            elif dist_matrix[i, j] == minimum:
                if not (i in result):
                    result.add(i)
                if not (j in result):
                    result.add(j)

proc comparison*[T: SomeNumber](linkage: typedesc[Single_linkage]; x, y: T): T =
    return min(x, y)

proc comparison*[T: SomeNumber](linkage: typedesc[Single_linkage]; x: Tensor[T], axis = 0): Tensor[T] =
    return min(x, axis = axis)

proc comparison*[T: SomeNumber](linkage: typedesc[Complete_linkage]; x, y: T): T =
    return max(x, y)

proc comparison*[T: SomeNumber](linkage: typedesc[Complete_linkage]; x: Tensor[T], axis = 0): Tensor[T] =
    return max(x, axis = axis)

proc comparison*[T: SomeNumber](linkage: typedesc[Weighted_linkage]; x: Tensor[T], axis = 0): Tensor[T] =
    return mean(x, axis = axis)

proc HClustering*[T: SomeNumber](linkage: typedesc[Single_linkage or Complete_linkage or Weighted_linkage];
                                    dist: Tensor[T]): T =
    var dist_matrix = dist
    var n = dist_matrix.shape[0].int
    var nodes = newSeq[Node[T]](n)
    var main_indices = [0..n-1].toTensor()

    for i in 0..n-1:
        nodes[i] = Node[T](kind: Leaf, index: i)

    while n > 1:
        let indices = find_min(dist_matrix)
        #echo "cluster ", indices
        var children = newSeq[Node[T]](len(indices))
        for i in 0..high(indices):
            children[i] = nodes[indices[i]]

        nodes[indices[0]] = Node[T](kind: Branch, distance: dist_matrix[indices[0], indices[1]] / 2.0, children: children)

        let new_cluster = linkage.comparison(dist_matrix[_, indices], axis=1)
        dist_matrix[_, indices[0]] = new_cluster
        dist_matrix[indices[0], _] = transpose(new_cluster)
        #nodes[indices[0]] = Node[T](distance: dist_matrix[indices[0], indices[1]]/2.0, index: -1, children: nodes[indices])
        var keep_idx = ones[T](n).astype(bool)
        for ii in 1..high(indices):
            keep_idx[indices[ii]] = false
            nodes.delete(indices[ii])
        dist_matrix = dist_matrix.masked_axis_select(keep_idx, axis=0).masked_axis_select(keep_idx, axis=1)
        #echo dist_matrix.shape
        n = dist_matrix.shape[0].int

    disp(nodes[0])


proc HClustering*[T: SomeNumber](linkage: typedesc[Single_linkage or Complete_linkage or Weighted_linkage];
                                  metric: typedesc[Euclidean];
                                       X: Tensor[T]): T =
    var dist_matrix = distance_matrix(metric, X)

    HClustering(linkage, dist_matrix)

var x = [1.0, 1.0].toTensor()
var y = [-1.0, -1.0].toTensor()

var X = [-0.3616294, -0.26037983, 1.26567644, 2.34676237, -0.03166053, 0.68876382
, -1.36468868, 0.53570823, -1.35740064, 0.57883562].toTensor().reshape(10,1)

#echo "Clustering"
#discard HClustering(Single_linkage, Euclidean, X)

var dist = [0,17,21,31,23,
            17,0,30,34,21,
            21,30,0,28,39,
            31,34,28,0,43,
            23,21,39,43,0].toTensor().reshape(5,5).astype(float)

echo "Distance matrix"
discard HClustering(Weighted_linkage, dist)
#var a1,a2 : Node[float]
#var b : Node[float]
#var b1 : Node[float]
#var c : Node[float]
#a1 = Node[float](distance: 2.0, children: @[])
#a2 = Node[float](distance: 1.0, children: @[])
#b = Node[float](distance: 3.0, children: @[a1, a2])
#b1 = Node[float](distance: 3.5, children: @[])
#c = Node[float](distance: 4.0, children: @[b, b1])

#disp(c)
