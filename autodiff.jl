# VERSION == v"1.8.5" #= Julia =#

module AutoDiff

using InteractiveUtils # need this to call subtypes()

# types and structs
export Constant
export Variable

export Identity
export Addition
export Multiplication
export Subtraction
export Division

export EmptyNode

export Node
export Graph

# functions
export inputs
export name
export output
export value
export bw_diff
export fw_diff

export nodes
export edges

export immdiff
export autodiff!


abstract type WhateverNodesAre end
abstract type Operation end
abstract type Term end

abstract type Constant<:Term end
abstract type Variable<:Term end

abstract type Identity      <:Operation end
abstract type Addition      <:Operation end
abstract type Multiplication<:Operation end
abstract type Subtraction   <:Operation end
abstract type Division      <:Operation end

struct EmptyNode<:WhateverNodesAre end

all_node_types = Union{EmptyNode, subtypes(Term)..., subtypes(Operation)...}

"""
    Node

    a term or operation in a computational graph

    node_type: The type (a singleton instance of a type) which the node represents.
               It must be a subtype of WhateverNodesAre, Term or Operation.

    inputs: tuple with at least one argument where each argument is a Real or an WhateverNodesAre
            All elements of the tuple must be a subtype of WhateverNodesAre, except
                the first argument, which may be a Real.
            For an Identity node, the input is a tuple of one node.
            For an Operation node, the inputs are two other nodes whose values it operates on.
            For a Term node, the inputs is a Real value to be assigned to it.
            Each argument in the input must be a unique object. For example, if N is a node, and you
                want to square it, you cannot do Node(Multiplication, (N, N)). You can use the
                Identity operator to create another node M = Node(Identity, (N,)) and then do
                Node(Multiplication, (N, M)).
    
    name: a string to help identify a node

    outputs: a zero-dimensional array of a tuple of nodes
             It records the outputs of the node.

    value: a zero-dimensional array that records the value of the node for the given inputs

    fw_diff: a zero-dimensional array that records a node's forward derivative

    bw_diff: a zero-dimensional array that records a node's backward derivative
"""
struct Node{T<:all_node_types}<:WhateverNodesAre
    node_type::Type{T} # node_type is a Type object
    inputs::Tuple{type1, Vararg{type2}} where {type1<:Union{Real, WhateverNodesAre}, type2<:WhateverNodesAre}
    name::String
    # Zero dimensional arrays cannot be created with values other than nothing, undef, or missing.
    # So we need to first create an zero-dimensional array with nothing (or undef or missing) and
    #     only then can update it with a Tuple. So first do:
    #     x = Array{Union{Nothing, Tuple{Vararg{Node}}}, 0}(nothing)
    # And then:
    #     x[] = (Node(),)
    # We can not directly do:
    #     x = Array{Union{Nothing, Tuple{Vararg{Node}}}, 0}((Node(),))
    # see: https://discourse.julialang.org/t/zero-dimensional-arrays-are-acting-strange/96909?u=sadish-d
    outputs::Array{Union{Nothing, Tuple{Vararg{Node}}}, 0}
    value  ::Array{Union{Nothing, Real}, 0}
    fw_diff::Array{Union{Nothing, Real}, 0}
    bw_diff::Array{Union{Nothing, Real}, 0}

    empty_outputs() = Array{Union{Nothing, Tuple{Vararg{Node}}}, 0}(nothing)
    empty_values()  = Array{Union{Nothing, Real}, 0}(nothing)

    function Node{T}(node_type::Type{T}, inputs, name, outputs, value, fw_diff, bw_diff) where T<:all_node_types
        length(inputs) == length(Set(inputs)) || error("Each input must be a unique object.")
        return new{T}(node_type, inputs, name, outputs, value, fw_diff, bw_diff)
    end

    Node(node_type::Type{T}, inputs, name, outputs, value, fw_diff, bw_diff) where T<:all_node_types = Node{T}(node_type::Type{T}, inputs, name, outputs, value, fw_diff, bw_diff)
    Node(node_type, inputs, name) = Node(node_type, inputs, name, empty_outputs(), empty_values(), empty_values(), empty_values())
    Node(node_type, inputs) = Node(node_type, inputs, "node", empty_outputs(), empty_values(), empty_values(), empty_values())
    # Nodes with only type or empty nodes may be useful for creating graphs. Maybe.
    Node(node_type) = Node(node_type, (EmptyNode(),), "node", empty_outputs(), empty_values(), empty_values(), empty_values())
    Node() = Node(EmptyNode)
end

function node_type(node::Node)
    return node.node_type
end

function inputs(node::Node)
    return node.inputs
end

function name(node::Node)
    return node.name
end

function outputs!(node::Node, outputs::Tuple{Vararg{Node}})
    node.outputs[] = outputs
    return nothing
end

function outputs(node::Node)
    return node.outputs[]
end

function value!(node::Node, value::Union{Nothing, Real})
    node.value[] = value
    return nothing
end

function value(node::Node)
    return node.value[]
end

function fw_diff!(node::Node, value::Union{Nothing, Real})
    node.fw_diff[] = value
    return nothing
end

function fw_diff(node::Node)
    return node.fw_diff[]
end

function bw_diff!(node::Node, value::Union{Nothing, Real})
    node.bw_diff[] = value
    return nothing
end

function bw_diff(node::Node)
    return node.bw_diff[]
end

# operator overloading
Base.:+(node1::Node, node2::Node) = Node(Addition, (node1, node2))
Base.:*(node1::Node, node2::Node) = Node(Multiplication, (node1, node2))
Base.:-(node1::Node, node2::Node) = Node(Subtraction, (node1, node2))
Base.:/(node1::Node, node2::Node) = Node(Division, (node1, node2))
Base.identity(node::Node)         = Node(Identity, (node,))

Base.show(io::IO, node::Node) = print(io, node.name)

"""
    Graph

    a computational graph of nodes

    The argument supplied to the constructor is an iterable of nodes.
    The constructor takes the set of unique nodes in the argument and topologically sorts the nodes
        before recording them. So the nodes in the argument do not need to be in topological order and
        may be repeated.
    Topological sort is not unique, so it makes sense to sort nodes once before recording them in the
        struct so that they do not need to be sorted again, potentially with a different order.
    If any of the nodes in the argument has an input node that is not in the argument, that input node
        also gets added to the graph (and its inputs recursively) because the topological sort algorithm
        calls that input and does not check whether it is in the argument.
    The constructor also assignes values and outputs to each node.
    
    nodes: a tuple of (topologically sorted) nodes

    edges: a boolean matrix showing the directed edges between (topologically sorted) nodes
"""
struct Graph
    nodes::Tuple{Vararg{Node}}
    edges::Array{Bool, 2}

    """
    topologically sort nodes of a computational graph
    """
    function topological_sort(nodes::Tuple{Vararg{Node}})
        unique_nodes = Set(nodes) |> Tuple

        nodes_accounted = Set()
        nodes_ordered = []
        function place_node(node)
            if !(node in nodes_accounted)
                push!(nodes_accounted, node)
                for input in inputs(node)
                    input isa Node && place_node(input) # `EmptyNode isa Node == false`
                end
                push!(nodes_ordered, node)
            end
        end
        for node in unique_nodes
            place_node(node)
        end
        return Tuple(nodes_ordered)
    end

    """
        given (ordered) nodes of a graph, returns an adjacency matrix of edges
    """
    function graph_edges(ordered_nodes::Tuple{Vararg{Node}})
        _edges = zeros(Bool, length(ordered_nodes), length(ordered_nodes))
        for i in eachindex(ordered_nodes)
            for input in inputs(ordered_nodes[i])
                if input in ordered_nodes
                    input_index = findall(==(input), ordered_nodes)[1]
                    _edges[input_index, i] = true
                end
            end
        end
        return _edges
    end

    """
        given (ordered) nodes of a graph, calculates the value of each node
    """
    function graph_values!(ordered_nodes::Tuple{Vararg{Node}})
        calc_value(node::Node{EmptyNode})        = nothing
        calc_value(node::Node{Identity})         = calc_value(inputs(node)[1])
        calc_value(node::Node{Constant})         = inputs(node)[1]
        calc_value(node::Node{Variable})         = inputs(node)[1]
        calc_value(node::Node{Addition})         = (inputs(node)[1] |> value) + (inputs(node)[2] |> value)
        calc_value(node::Node{Multiplication})   = (inputs(node)[1] |> value) * (inputs(node)[2] |> value)
        calc_value(node::Node{Subtraction})      = (inputs(node)[1] |> value) - (inputs(node)[2] |> value)
        calc_value(node::Node{Division})         = (inputs(node)[1] |> value) / (inputs(node)[2] |> value)
    
        for node in ordered_nodes
            calc_value(node) |> o-> value!(node, o)
        end
        return nothing
    end

    """
        given nodes, gets their outputs
    """
    function graph_outputs!(nodes::Tuple{Vararg{Node}})
        for node in nodes
            # order of output nodes does not matter
            output_nodes = Set{Node}()
            # loop over candidate output nodes
            for candidate in nodes
                for input in inputs(candidate)
                    (input == node) && push!(output_nodes, candidate)
                end
            end
            output_nodes = output_nodes |> Tuple
            outputs!(node, output_nodes)
        end
        return nothing
    end

    function Graph(nodes::Tuple{Vararg{Node}})
        nodes_ordered = topological_sort(nodes)
        _edges = graph_edges(nodes_ordered)

        # Check if graph is cyclical.
        #     Given how the struct Node is created, a node can not be defined before its inputs are defined.
        #     Before a node can be defined, its inputs have to be defined. And the inputs of a node can not be
        #     edited since Nodes are not mutable. So we probably cannot get cyclical graphs. Still:
        lower_left_half = [(i >= j) ? (_edges[i, j]) : (false) for i in 1:size(_edges, 1), j in 1:size(_edges, 2)]
        any(lower_left_half) && error("The graph is not a directed acyclic graph.")

        graph_values!(nodes_ordered)

        graph_outputs!(nodes_ordered)

        # ----
        println("nodes (ordered):")
        for node in nodes_ordered
            println(name(node))
        end
        println("edges:")
        display(_edges)
        # ----
        return new(nodes_ordered, _edges)
    end

    Graph() = Graph((Node(),))
end
Graph(nodes::Array{<:Node, 1}) = Graph(Tuple(nodes))
Graph(nodes::Set{<:Node})      = Graph(Tuple(nodes))

function nodes(graph::Graph)
    return graph.nodes
end

function edges(graph::Graph)
    return graph.edges
end

Base.show(io::IO, graph::Graph) = print(io, "nodes: ", graph.nodes, " edges: ", graph.edges)


"""
    When applied to nodes, returns differentiation of a node
        with respect to its immediate input nodes
    Returns 1 if differentiating with respect to itself.
    Returns 0 for all nodes that are not immediate inputs.

    When applied on graph returns a matrix of differentiation of
        each node with respect to another.
"""
function immdiff(node::Node{Identity}, n::Node)
    n == node && return 1
    n == inputs(node)[1] && return 1
    return 0
end
function immdiff(node::Node{Constant}, n::Node)
    n == node && return 1
    return 0
end
function immdiff(node::Node{Variable}, n::Node)
    n == node && return 1
    return 0
end
function immdiff(node::Node{Addition}, n::Node)
    n == inputs(node)[1] && return 1
    n == inputs(node)[2] && return 1
    n == node && return 1
    return 0
end
function immdiff(node::Node{Multiplication}, n::Node)
    n == inputs(node)[1] && return inputs(node)[2] |> value
    n == inputs(node)[2] && return inputs(node)[1] |> value
    n == node && return 1
    return 0
end
function immdiff(node::Node{Subtraction}, n::Node)
    n == inputs(node)[1] && return 1
    n == inputs(node)[2] && return -1
    n == node && return 1
    return 0
end
function immdiff(node::Node{Division}, n::Node)
    n == inputs(node)[1] && return 1 / (inputs(node)[2] |> value)
    n == inputs(node)[2] && return -(inputs(node)[1] |> value) / ((inputs(node)[2] |> value) * (inputs(node)[2] |> value))
    n == node && return 1
    return 0
end
function immdiff(graph::Graph)
    _nodes = nodes(graph) # Nodes are topologically sorted.
    _edges  = edges(graph)
    # matrix for storing differentiation at each edge
    d_graph = zeros(Float64, size(_edges))
    for i in CartesianIndices(d_graph)
        node_in  = _nodes[i[1]]
        node_out = _nodes[i[2]]
        d_graph[i] = immdiff(node_out, node_in)
        i == CartesianIndex(2, 4) && begin
        end
    end
    return d_graph
end

function autodiff!(graph::Graph, node::Node; backward::Bool=false)
    node in nodes(graph) || error("The node is not in the graph.")
    _nodes = nodes(graph) # Nodes are topologically sorted.

    # functions to determine the order of two nodes
    is_before(node1::Node, node2::Node) = (findall(==(node1), _nodes) < findall(==(node2), _nodes))
    is_after(node1::Node, node2::Node)  = (findall(==(node1), _nodes) > findall(==(node2), _nodes))

    # clear bw_diff or fw_diff field
    backward && for n in _nodes bw_diff!(n, 0) #= bw_diff!(n, nothing) =# end
    backward || for n in _nodes fw_diff!(n, 0) #= fw_diff!(n, nothing) =# end

    if backward # backward differentiation ====================================
        for node_in_path in reverse(_nodes) # start from the node at the end of the computational graph
            if is_before(node_in_path, node)
                # specify Real as type of bw_diffs so that the array can be summed even when empty.
                bw_diffs = Real[] # to collect the individual contribution of each output and sum later
                for output in outputs(node_in_path) # since we start from the end of the graph, all outputs should have bw_diff defined now
                    # might not need this condition. All outputs should be nodes.
                    (output isa Node) && bw_diff(output) * immdiff(output, node_in_path) |> o-> push!(bw_diffs, o) # the chain rule of calculus applied backward
                end
                # sum of bw_diffs through all backward paths originating at source node
                sum(bw_diffs) |> o-> bw_diff!(node_in_path, o)
            elseif is_after(node_in_path, node)
                bw_diff!(node_in_path, 0)
            else
                node_in_path == node || error("Expected to differentiate a node with respect to itself.")
                bw_diff!(node_in_path, 1)
            end
        end
    else # forward differentiation ============================================
        for node_in_path in _nodes # start from the node at the end of the computational graph
            if is_after(node_in_path, node)
                # specify Real as type of fw_diffs so that the array can be summed even when empty.
                fw_diffs = Real[] # to collect the individual contribution of each input and sum later
                for input in inputs(node_in_path) # since we start from the end of the graph, all inputs should have fw_diff defined now
                    # for Constants and Variables, inputs are not Nodes.
                    (input isa Node) && fw_diff(input) * immdiff(node_in_path, input) |> o-> push!(fw_diffs, o) # the chain rule of calculus applied forward
                end
                # sum of fw_diffs through all forward paths orginating at source node
                sum(fw_diffs) |> o-> fw_diff!(node_in_path, o)
            elseif is_before(node_in_path, node)
                fw_diff!(node_in_path, 0)
            else
                node_in_path == node || error("Expected to differentiate a node with respect to itself.")
                fw_diff!(node_in_path, 1)
            end
        end
    end

    return nothing
end

end #= end of module =#

# references [retrieved 2023-04-04]:
# [an explanation of automatic differentiation using computaitonal graphs](https://colah.github.io/posts/2015-08-Backprop/)
# [an implementation of automatic differentiation using graphs and topological sorting](https://github.com/Jmkernes/Automatic-Differentiation/blob/main/AutomaticDifferentiation.ipynb)

#=
# Examples ----------------------------------------------------
# automatic differentiation

# This is a study of forward and backward mode automatic differentation using computational graphs.

# We represent each term or operation in a computation as a node in a graph. A node carries informaiton on what type of term or variable it represents. Some nodes go into other nodes as inputs. Nodes can be given names to specify the computation they represent. The `Node()` constructor function creates nodes.

# We an put nodes in a graph. A graph carries a topologically sorted list of nodes and an adjacency matrix of edges that denote which nodes are inputs to which other nodes. The `Graph()` constructor creates graphs.

# Some terminology:
# - output: For a given node, its output node is another node that has the former as an input.
# - terminal nodes: nodes that do not have other nodes as inputs (terminal input nodes) or that do not have other nodes as outputs (terminal output nodes).
# - intermediate nodes: nodes that have other nodes as inputs and outputs.
# - ancestors: For a given node, its ancestors are its input nodes and, recursively, the inputs of inputs.
# - descendents: For a given node, its descendents are its outputs and, recursively, the outputs of its outputs.
# - orphan: An orphan node is a node that is neither an input nor an output node to any other node in a graph.

using .AutoDiff

# Let's look at two functions: ax+by and abx+y².
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

# Let's create a graph for `f`. For any node supplied to the `Graph()` constructor as an argument, all its ancestor nodes also get added to the graph.

Graph([f])

# We get:

"""
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
"""

# The matrix shows, for instance, that the first node `a` (first row) and second node `x` (second row) are inputs to the third node (third column).

# The bottom left of the adjacency matrix must be empty. Otherwise, it means the graph has cycles, and can not represent a valid computation.

# Graphs can hold multiple computations or functions involving multiple inputs and outputs. Orphan ndoes in a graph do not affect the computations.

graph = Graph([f, g, orphan])

# Nodes may be entered in any order or repeated.

Graph([g, b, f, a, a]) |> nodes |> Set == Graph([f, g]) |> nodes |> Set
!(orphan in nodes(Graph([f, g]))) # This node is not added to the graph.

# The terminal nodes `f` and `g` have their own sets of intermediate nodes in the graph but `a`, `b`, `x`, and `y` appear only once since we used the same objects (with bindings to the symbols `a`, `b`, `x`, and `y`) to define both `f` and `g`.
# These are the nodes in the graph:
# 1. a
# 2. b
# 3. x
# 4. y
# 5. ax
# 6. by
# 7. f or ax+by
# 8. ab
# 9. abx
# 10. identity(y)
# 11. y² or y*identity(y)
# 12. g or abx+y²
# 13. orphan

@assert length(nodes(graph)) == 13
for n in [a, b, x, y] # These nodes only appear in the graph once.
    @assert count(==(n), nodes(graph)) == 1
end

# Now that we have the graph, let's perform automatic differentiation.

# Simultaneously differentiating `f` and `g` with respect to `x`:

autodiff!(graph, x)
@assert fw_diff(f) == _a       # a
@assert fw_diff(g) == _a * _b  # ab

# Differentiating `f` with respect to `x` and `y` simultaneously:

autodiff!(graph, f, backward=true)
@assert bw_diff(x) == _a  # a
@assert bw_diff(y) == _b  # ab

# We can also differentiate intermediate nodes with respect to other nodes, or differentiate nodes with respect to intermediate nodes.

autodiff!(graph, ab)
autodiff!(graph, ab, backward=true)
@assert bw_diff(x) == 0   # 0
@assert bw_diff(y) == 0   # 0
@assert bw_diff(a) == _b  # b
@assert bw_diff(b) == _a  # a
@assert fw_diff(f) == 0   # 0
@assert fw_diff(g) == _x  # x
# -------------------------------------------------------------
=#

