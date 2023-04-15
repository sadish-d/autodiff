# automatic differentiation

This is a study of forward and backward mode automatic differentation using computational graphs.

We represent each term or operation in a computation as a node in a graph. A node carries informaiton on what type of term or variable it represents. Some nodes go into other nodes as inputs. Nodes can be given names to specify the computation they represent. The `Node()` constructor function creates nodes.

We an put nodes in a graph. A graph carries a topologically sorted list of nodes and an adjacency matrix of edges that denote which nodes are inputs to which other nodes. The `Graph()` constructor creates graphs.

Some terminology:
- output: For a given node, its output node is another node that has the former as an input.
- terminal nodes: nodes that do not have other nodes as inputs (terminal input nodes) or that do not have other nodes as outputs (terminal output nodes).
- intermediate nodes: nodes that have other nodes as inputs and outputs.
- ancestors: For a given node, its ancestors are its input nodes and, recursively, the inputs of inputs.
- descendents: For a given node, its descendents are its outputs and, recursively, the outputs of its outputs.
- orphan: An orphan node is a node that is neither an input nor an output node to any other node in a graph.

```julia
using .AutoDiff
```

Let's look at two functions: ax+by and abx+y².
```julia
_a = 2
_b = 3
_x = 4
_y = 5
a = Node(Constant, (_a,), "a")
b = Node(Constant, (_b,), "b")
x = Node(Variable, (_x,), "x")
y = Node(Variable, (_y,), "y")

orphan = Node(EmptyNode, (EmptyNode(),), "orphan") # We will use this node to demonstrate some features.

f  = a * x + b * y             # ax+by
ab = a * b                     # ab : We will use this node to demonstrate some features.
g  = ab * x + y * identity(y)  # abx+y²
```

Let's create a graph for `f`. For any node supplied to the `Graph()` constructor as an argument, all its ancestor nodes also get added to the graph.

```julia
Graph([f])
```

We get:
```
nodes (ordered):
a
x
node
b
y
node
node
edges:
7×7 Matrix{Bool}:
 0  0  1  0  0  0  0
 0  0  1  0  0  0  0
 0  0  0  0  0  0  1
 0  0  0  0  0  1  0
 0  0  0  0  0  1  0
 0  0  0  0  0  0  1
 0  0  0  0  0  0  0
```

The matrix shows, for instance, that the first node `a` (first row) and second node `x` (second row) are inputs to the third node (third column).

The bottom left of the adjacency matrix must be empty. Otherwise, it means the graph has cycles, and can not represent a valid computation.

Graphs can hold multiple computations or functions involving multiple inputs and outputs. Orphan ndoes in a graph do not affect the computations.

```julia
graph = Graph([f, g, orphan])
```

Nodes may be entered in any order or repeated.

```julia
Graph([g, b, f, a, a]) |> nodes |> Set == Graph([f, g]) |> nodes |> Set
!(orphan in nodes(Graph([f, g]))) # This node is not added to the graph.
```

The terminal nodes `f` and `g` have their own sets of intermediate nodes in the graph but `a`, `b`, `x`, and `y` appear only once since we used the same objects (with bindings to the symbols `a`, `b`, `x`, and `y`) to define both `f` and `g`.
These are the nodes in the graph:
1. a
2. b
3. x
4. y
5. ax
6. by
7. f or ax+by
8. ab
9. abx
10. identity(y)
11. y² or y*identity(y)
12. g or abx+y²
13. orphan

```julia
@assert length(nodes(graph)) == 13
for n in [a, b, x, y] # These nodes only appear in the graph once.
	@assert count(==(n), nodes(graph)) == 1
end
```

Now that we have the graph, let's perform automatic differentiation.

Simultaneously differentiating `f` and `g` with respect to `x`:

```julia
autodiff!(graph, x)
@assert fw_diff(f) == _a       # a
@assert fw_diff(g) == _a * _b  # ab
```

Differentiating `f` with respect to `x` and `y` simultaneously:

```julia
autodiff!(graph, f, backward=true)
@assert bw_diff(x) == _a  # a
@assert bw_diff(y) == _b  # ab
```

We can also differentiate intermediate nodes with respect to other nodes, or differentiate nodes with respect to intermediate nodes.

```julia
autodiff!(graph, ab)
autodiff!(graph, ab, backward=true)
@assert bw_diff(x) == 0   # 0
@assert bw_diff(y) == 0   # 0
@assert bw_diff(a) == _b  # b
@assert bw_diff(b) == _a  # a
@assert fw_diff(f) == 0   # 0
@assert fw_diff(g) == _x  # x
```


If you find errors in my code or have comments, do share.

references [retrieved 2023-04-04]:
- [an explanation of automatic differentiation using computaitonal graphs](https://colah.github.io/posts/2015-08-Backprop/)
- [an implementation of automatic differentiation using graphs and topological sorting](https://github.com/Jmkernes/Automatic-Differentiation/blob/main/AutomaticDifferentiation.ipynb)
