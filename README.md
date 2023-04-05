# automatic differentiation

This is a study of forward and backward mode automatic differentation using computational graphs.

We can represent each term or operation in a computational graph as a node. The Node() constructor function creates nodes. Here are the nodes of the computation a/x² + b, one node for each step in the computation.

```julia
a = 2
b = 30
x = 3
n1 = Node(Constant, (a,), "a")
n2 = Node(Constant, (b,), "b")
n3 = Node(Variable, (x,), "x")
n4 = Node(Identity, (n3,), "Identity(x)")
n5 = Node(Multiplication, (n3, n4), "x²")
n6 = Node(Division, (n1, n5), "a/x²")
n7 = Node(Addition, (n6, n2), "a/x² + b")
```

A Node carries informaiton on what type of term or variable it represents. Some nodes go into other nodes as inputs. Nodes can be given names to specify which point in the computation they represent.

Once the nodes are created, we can create a graph. The Graph() constructor creates graphs. Here is the graph for the above

```julia
graph1 = Graph([n7, n7, n1, n2, n3, n4, n5, n6])
```

It does not matter which order the nodes are supplied in, and whether the same nodes are supplied twice. The constructor function keeps only unique nodes, and sorts them topologically. When it is done creating a graph, it shows how the nodes are ordered and also shows the directed edges in an adjacency matrix.

```
order of nodes:
a
x
Identity(x)
x²
a/x²
b
a/x² + b
edges:
7×7 Matrix{Bool}:
 0  0  0  0  1  0  0
 0  0  1  1  0  0  0
 0  0  0  1  0  0  0
 0  0  0  0  1  0  0
 0  0  0  0  0  0  1
 0  0  0  0  0  0  1
 0  0  0  0  0  0  0
 ```
 
 Note that the bottom left of the adjacency matrix must be empty. Otherwise, it means the graph has cycles, and can not represent a valid computation.
 
 Now that the graph is constructed, we can do automatic differentiation, both forwards and backwards.
 
 ```julia
autodiff!(graph1, n3)
autodiff!(graph1, n7, backward=true)
@assert bw_diff(n3) == fw_diff(n7) == -2a/x^3
```

The assertion confirms that the we get the same, correct results using both methods.

Here, we differentiated a terminal output node (one that is not itself an input to another node) with respect to a terminal input node (one that is not an output of another node). But we can also differentiate an intermediate node with respect to another intermediate node.

```julia
autodiff!(graph1, n5)
autodiff!(graph1, n6, backward=true)
@assert bw_diff(n5) == fw_diff(n6) == -a/x^4
```

Note that nodes not explicitly supplied as an argument to the graph constructor can also get added to the graph. For example:

```julia
nn1 = Node(Identity, (Node(),), "1")
nn2 = Node(Identity, (nn1,), "2")
nn3 = Node(Identity, (nn2,), "3")
nn1 = Node(Identity, (nn3,), "4")

graph2 = Graph([nn1, nn2, nn3])
```

This creates the following graph:
```
order of nodes:
node
node 1
node 2
node 3
node 4
edges:
5×5 Matrix{Bool}:
 0  1  0  0  0
 0  0  1  0  0
 0  0  0  1  0
 0  0  0  0  1
 0  0  0  0  0
 ```

Here, we supplied three nodes to the graph, but the graph has five nodes. Confirm this by:
```julia
@assert length(nodes(graph2)) == 5
```

I had fun creating this. It was a great way to learn Julia. If you find errors in my code or have comments, suggestions, do share.

references [retrieved 2023-04-04]:

[an explanation of automatic differentiation using computaitonal graphs](https://colah.github.io/posts/2015-08-Backprop/)

[an implementation of automatic differentiation using graphs and topological sorting](https://github.com/Jmkernes/Automatic-Differentiation/blob/main/AutomaticDifferentiation.ipynb)
