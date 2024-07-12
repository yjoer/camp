# %%
from camp import GraphString
from camp import dijkstra

# %%
graph = GraphString()

graph.add_node("A")
graph.add_node("B")
graph.add_node("C")
graph.add_node("D")
graph.add_node("E")

graph.add_edge("A", "B", 5)
graph.add_edge("B", "C", 6)
graph.add_edge("C", "D", 2)
graph.add_edge("A", "C", 15)

dijkstra(graph, "A", "D")

# %%
