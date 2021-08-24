import arraymancer
import utils
import strutils
import random
import tables

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
            class: Tensor[int]

    Tree*[T] = object
        tree: B_node[T]
        nbr_leaf: int
        nbr_features: int
        nbr_classes: int

proc disp[T](tree: B_Node[T]; prefix = ""): string =
    let split  = " ├── "
    let leaf   = " └── "
    let space  = "     "
    let branch = " |   "

    if tree.kind == Leaf:
        var class_str = "Class: "
        for class in tree.class:
            class_str = class_str & " " & $class
        return class_str & "\n"
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
        count[el.int] += 1.0.T

    return count

proc count[T: SomeNumber](y: Tensor[T], K: int): Tensor[T] =
    var count = zeros[T](K)

    for el in y:
        count[el.int] += 1.0.T

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

proc find_split_c[T: SomeFloat](x, y: Tensor[T]; indices: Tensor[int], criterion: typedesc[Entropy or Gini], random_p = false): Split_data[T] =
    let n = x.shape[0].int
    let p = x.shape[1].int
    var min_idx_p = -1
    var min_min_impurity = high(T)
    var min_min_n = 1.0.T
    var min_min_n_idx = 0

    var features_indices : seq[int]
    for i in 0..(p-1):
        features_indices.add(i)
    
    if random_p:
        shuffle(features_indices)
        features_indices = features_indices[0..floor(sqrt(p.float)).int]


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

proc build_tree_c[T: SomeFloat](x, y: Tensor[T], criterion: typedesc[Gini or Entropy], nbr_leaf: var int, max_depth = -1, random_p = false, depth_counter = 0): B_Node[T] =
    let n = x.shape[0].int
    let p = x.shape[1].int

    if depth_counter >= max_depth and max_depth != -1:
        nbr_leaf += 1
        return B_Node[T](kind: Leaf, class: count(y).astype(int))

    if (n == 1):
        nbr_leaf += 1
        return B_Node[T](kind: Leaf, class: count(y).astype(int))

    var indices = zeros[int](n, p)
    for i in 0..(p-1):
        indices[_, i] = argsort(x[_, i].flatten()).reshape(n, 1)

    let res_split = find_split_c(x, y, indices, criterion, random_p = random_p)

    if res_split.impurity == 0.0:
        nbr_leaf += 1
        return B_Node[T](kind: Leaf, class: res_split.count.astype(int))

    let idx = indices[_, res_split.feature_idx].flatten()
    let y_left = y[idx][0..res_split.split_at_idx]
    let x_left = x[idx, _][0..res_split.split_at_idx]

    let a = build_tree_c(x_left, y_left, criterion, nbr_leaf, random_p = random_p, depth_counter = depth_counter + 1, max_depth = max_depth)

    let y_right = y[idx][res_split.split_at_idx+1..^1]
    let x_right = x[idx, _][res_split.split_at_idx+1..^1]

    let b = build_tree_c(x_right, y_right, criterion, nbr_leaf, random_p = random_p, depth_counter = depth_counter + 1, max_depth = max_depth)

    return B_Node[T](kind: Branch, left: a, right: b, value: res_split)

proc CART_c[T: SomeFloat](x, y: Tensor[T]; criterion = Gini, max_depth = -1, random_p = false): Tree[T] =
    result.tree = build_tree_c(x, y, criterion, result.nbr_leaf, max_depth = max_depth, random_p = random_p)
    result.nbr_features = x.shape[1].int
    result.nbr_classes = count(y).size

proc load_txt(path: string): Tensor[float] =
    var data_X = split(readfile(path), "\n")[0..^2]
    let n = len(data_X)
    let p = len(split(data_X[0], " "))

    var X = zeros[float](n, p)

    for i in 0..n-1:
        X[i, _] = split(data_X[i], " ").toTensor().map(parseFloat).reshape(1, p)

    return X

type
    Bsample[T] = object
        x: Tensor[T]
        y: Tensor[T]
        mask_oob: Tensor[bool]

proc bootstrapping[T: SomeFloat](x, y: Tensor[T]): Bsample[T] =
    randomize()

    let n = x.shape[0].int
    let p = x.shape[1].int

    result.x = zeros[T](n, p)
    result.y = zeros[T](n, 1)
    result.mask_oob = ones[int](n).astype(bool)

    for i in 0..(n-1):
        let line_idx = random.rand(n-1)
        result.x[i, _] = x[line_idx, _]
        result.y[i, _] = y[line_idx, _]
        result.mask_oob[line_idx] = false

proc predict_classes[T: SomeFloat](x: Tensor[T], fit: Tree[T]): Tensor[int] =
    let n = x.shape[0].int
    let p = x.shape[1].int

    if p != fit.nbr_features:
        raise newException(ValueError, "Number of features from the tree and x are different")

    result = zeros[int](n, fit.nbr_classes)
    let tree = fit.tree
    for i in 0..(n-1):
        var next = tree
        let sample = x[i, _]
        while next.kind != Leaf:
            if sample[0, next.value.feature_idx] <= next.value.split_at_value:
                next = next.left
            else:
                next = next.right

        let classes = next.class
        result[i, 0..(classes.size.int-1)] = classes.reshape(1, classes.size.int)

proc predict_proba_c[T: SomeFloat](x: Tensor[T], fit: Tree[T]): Tensor[(int, float)] =
    let classes = predict_classes(x, fit)
    let n = x.shape[0].int
    result = newTensor[(int, float)](n)
    for i in 0..(n-1):
        result[i] = (argmax(classes[i, _]), max(classes[i, _]) / sum(classes[i, _]))

type
    forest[T] = Tensor[Tree[T]]
    
proc predict_proba_c[T: SomeFloat](x: Tensor[T], fits: forest[T]): Tensor[float] =
    let N = fits.shape[0].int
    let n = x.shape[0].int

    var hard_votes = zeros[int](n, fits[0].nbr_classes)
    var soft_votes = zeros[float](n, fits[0].nbr_classes)
    for i in 0..(N-1):
        let classes = predict_classes(x, fits[i])
        let class_vote = argmax(classes, 1)
        let class_proba = classes.astype(float) /. sum(classes, 1).astype(float)
        soft_votes += class_proba
        for j in 0..(n-1):
            hard_votes[j, class_vote[j, 0]] += 1

    return concat(argmax(hard_votes, 1).astype(float), argmax(soft_votes, 1).astype(float), max(soft_votes, 1), axis = 1)

proc predict_c[T: SomeFloat](x: Tensor[T], fit: Tree[T]): Tensor[int] =
    let n = x.shape[0].int
    let res = predict_proba_c(x, fit)
    result = -ones[int](n)

    for i in 0..(n-1):
        result[i] = res[i][0]

proc random_forest_c[T: SomeFloat](x, y: Tensor[T], N = 100, criterion = Gini, max_depth = -1, random = false): forest[T] =
    result = newTensor[Tree[T]](N)

    for i in 0..N-1:
        let bs_x = bootstrapping(x, y)
        result[i] = CART_c(bs_x.x, bs_x.y, criterion, max_depth = max_depth, random_p = true)
    
var
    X = load_txt("iris_X.dat")
    y = load_txt("iris_y.dat")
    depth = 2

let res = CART_c(X, y, Entropy, max_depth = depth)

echo disp(res.tree)
echo res.nbr_leaf

let y_pred = predict_c(X, res)
echo y_pred.shape
echo y.flatten().shape

echo (y_pred ==. y.flatten().astype(int)).astype(int).sum()
let y_pred_prob = predict_proba_c(X, res)
echo y_pred_prob

let res_rf = random_forest_c(X, y, max_depth = depth)
echo predict_proba_c(X, res_rf)

