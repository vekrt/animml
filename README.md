# animml
A small nim implementation of various statistics/ML concepts. Mostly untested and poorly written collection of things I have learned once.


# Decision tree classification
Classification of the Iris dataset using CART with Gini and full depth.

 ```
 [2]<=2.45
 ├── [3]<=1.75
 |    ├── [2]<=4.85
 |    |    ├── Class: 2
 |    |    └── [0]<=5.95
 |    |         ├── Class: 2
 |    |         └── Class: 1
 |    └── [2]<=4.95
 |         ├── [3]<=1.55
 |         |    ├── [0]<=6.95
 |         |    |    ├── Class: 2
 |         |    |    └── Class: 1
 |         |    └── Class: 2
 |         └── [3]<=1.65
 |              ├── Class: 2
 |              └── Class: 1
 └── Class: 0
```

# Hierarchical clustering
[Single linkage clustering ](https://en.wikipedia.org/wiki/Single-linkage_clustering#The_single-linkage_dendrogram)
```
Distance: 14.0
 ├── Distance: 10.5
 |    ├── Distance: 8.5
 |    |    ├── Index: 0
 |    |    └── Index: 1
 |    ├── Index: 2
 |    └── Index: 4
 └── Index: 3
```
