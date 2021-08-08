import arraymancer
import utils
import strutils

type
    Gini = object
    Entropy = object
    Split_data[T] = object
        feature_idx: int
        split_at_idx: int
        split_at_value: T
        impurity: T
        N_sample: int
        count: Tensor[T]

    Node_kind* = enum
        Leaf, Branch

    B_Node*[T] = ref object
        case kind*: Node_kind
        of Branch:
            value: Split_data[T]
            left : B_Node[T]
            right: B_Node[T]
        of Leaf:
            class: int

    Tree*[T] = object
        tree: B_node[T]
        nbr_leaf: int

proc disp[T](tree: B_Node[T]; prefix = ""): string =
    let split  = " ├── "
    let leaf   = " └── "
    let space  = "     "
    let branch = " |   "

    if tree.kind == Leaf:
        return "Class: " & $tree.class & "\n"
    else:
        let left  = disp(tree.left, prefix & space)
        let right = disp(tree.right, prefix & branch)
        let value = "[" & $tree.value.feature_idx & "]"  & "<=" & $tree.value.split_at_value & "\n"
        return value &
               prefix & split & right &
               prefix & leaf & left


proc count[T: SomeNumber](y: Tensor[T]): Tensor[T] =
    let K = max(y).int + 1
    var count = zeros[T](K)

    for el in y:
        count[el.int] += 1.0

    return count

proc count[T: SomeNumber](y: Tensor[T], K: int): Tensor[T] =
    var count = zeros[T](K)

    for el in y:
        count[el.int] += 1.0

    return count

proc impurity[T: SomeFloat](criterion: typedesc[Entropy]; y: Tensor[T]): T =
    let ps = count(y) / y.size().float
    
    for p in ps:
        if p == 0.0:
            result += 0.0
        else:
            result += -p * log2(p)
    
    return result

proc impurity[T: SomeFloat](criterion: typedesc[Gini]; y: Tensor[T]): T =
    let p = count(y) / y.size().float
    
    return 1.0 - sum(p *. p)

proc find_split_c[T: SomeFloat](x, y: Tensor[T]; indices: Tensor[int], p_avoid: int, criterion: typedesc[Entropy or Gini]): Split_data[T] =
    let n = x.shape[0].int
    let p = x.shape[1].int
    var min_idx_p = -1
    var min_min_impurity = high(T)
    var min_min_n = 1.0.T
    var min_min_n_idx = 0

    var features_indices : seq[int]
    for i in 0..(p-1):
        if i == p_avoid:
            continue
        features_indices.add(i)

    for i in features_indices:
        let idx = indices[_, i]
        var sub_x = (x[_, i])[idx.flatten()]
        
        let y_ordered = y[idx.flatten()]
        
        let mid_point = (sub_x[1..^1] + sub_x[0..^2]) / 2.0

        var min_impurity = high(T)
        var min_n = 1.0.T
        var min_n_idx = -1
        for j in 0..(n-2):
            if sub_x[j, 0] == sub_x[j+1, 0]:
                continue 

            let left_impurity = criterion.impurity(y_ordered[0..j])
            let right_impurity = criterion.impurity(y_ordered[j+1..^1])
            let total_impurity = ((j + 1).float * left_impurity + (n - j - 1).float * right_impurity) / n.float
            if total_impurity < min_impurity:
                min_impurity = total_impurity
                min_n = mid_point[j, 0]
                min_n_idx = j

        if min_impurity < min_min_impurity:
            min_min_impurity = min_impurity
            min_min_n = min_n
            min_idx_p = i
            min_min_n_idx = min_n_idx
            
    result.feature_idx = min_idx_p
    result.split_at_idx = min_min_n_idx
    result.split_at_value = min_min_n
    result.impurity = criterion.impurity(y)
    result.N_sample = n
    result.count = count(y)

proc build_tree_c[T: SomeFloat](x, y: Tensor[T], criterion: typedesc[Gini or Entropy], nbr_leaf: var int, p_avoid = -1): B_Node[T] =
    let n = x.shape[0].int
    let p = x.shape[1].int

    if (n == 1):
        nbr_leaf += 1
        return B_Node[T](kind: Leaf, class: argmax(count(y)))

    var indices = zeros[int](n, p)
    for i in 0..(p-1):
        indices[_, i] = argsort(x[_, i].flatten()).reshape(n, 1)

    let res_split = find_split_c(x, y, indices, p_avoid, criterion)

    if res_split.impurity == 0.0:
        nbr_leaf += 1
        return B_Node[T](kind: Leaf, class: argmax(res_split.count))

    let idx = indices[_, res_split.feature_idx].flatten()
    let y_left = y[idx][0..res_split.split_at_idx]
    let x_left = x[idx, _][0..res_split.split_at_idx]

    let a = build_tree_c(x_left, y_left, criterion, nbr_leaf, p_avoid = res_split.feature_idx)

    let y_right = y[idx][res_split.split_at_idx+1..^1]
    let x_right = x[idx, _][res_split.split_at_idx+1..^1]

    let b = build_tree_c(x_right, y_right, criterion, nbr_leaf, p_avoid = res_split.feature_idx)

    return B_Node[T](kind: Branch, left: a, right: b, value: res_split)

proc CART_c[T: SomeFloat](x, y: Tensor[T]; criterion = Gini): Tree[T] =
    result.tree = build_tree_c(x, y, criterion, result.nbr_leaf)

proc load_txt(path: string): Tensor[float] =
    var data_X = split(readfile(path), "\n")[0..^2]
    let n = len(data_X)
    let p = len(split(data_X[0], " "))

    var X = zeros[float](n, p)

    for i in 0..n-1:
        X[i, _] = split(data_X[i], " ").toTensor().map(parseFloat).reshape(1, p)

    return X

var
    X = load_txt("iris_X.dat")
    y = load_txt("iris_y.dat")

let res = CART_c(X, y, Entropy)

echo disp(res.tree)
echo res.nbr_leaf
