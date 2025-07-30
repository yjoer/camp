// %%
gClingOpts->AllowRedefinition = 1;

// %%
#include <algorithm>
#include <functional>
#include <iostream>
#include <queue>
#include <stack>
#include <vector>

// %%
typedef std::pair<int, int> ii;
typedef std::vector<int> vi;
typedef std::vector<ii> vii;
typedef std::vector<std::string> vs;
typedef std::pair<std::vector<vi>, std::vector<vi>> vvivvi;

// %% [markdown]
// ## Union-Find

// %%
class UnionFind {
protected:
  vi parent;
  vi rank;

public:
  UnionFind(int n);

  int find(int x);
  bool unite(int x, int y);
};

// %%
UnionFind::UnionFind(int n) {
  parent.resize(n);
  rank.resize(n, 0);
  for (int i = 0; i < n; ++i)
    parent[i] = i;
}

// %%
int UnionFind::find(int x) {
  // path compression
  if (parent[x] != x)
    parent[x] = find(parent[x]);

  return parent[x];
}

// %%
bool UnionFind::unite(int x, int y) {
  int rx = find(x);
  int ry = find(y);

  if (rx == ry)
    return false;

  // union by rank
  if (rx < ry) {
    parent[rx] = ry;
  } else if (ry < rx) {
    parent[ry] = rx;
  } else {
    parent[ry] = rx;
    rank[rx]++;
  }

  return true;
}

// %% [markdown]
// ### Usage

// %%
UnionFind uf(5);
uf.unite(1, 2);
uf.unite(3, 4);
std::cout << (uf.find(1) == uf.find(2)) << std::endl;
std::cout << (uf.find(1) == uf.find(3));

// %% [markdown]
// ## Adjacency List

// %%
class AdjacencyGraph {
protected:
  std::vector<vii> adj_list;
  vs labels;
  int e = 0;

public:
  int n_vertices();
  int n_edges();

  int add_vertex(std::string label);
  void add_edge(int u, int v, int weight);
  AdjacencyGraph transpose();

  void breadth_first_search(int u, std::function<void(int)> visitor);
  void depth_first_search(int u, std::function<void(int)> visitor);

  std::pair<vi, vi> dijkstra(int s);
  std::pair<std::vector<vi>, std::vector<vi>> floyd_warshall();

  AdjacencyGraph kruskal();
  AdjacencyGraph prim();

  bool kosaraju_test();
  std::vector<vi> kosaraju();
  std::vector<vi> tarjan();

  friend class GraphViewer;
};

// %%
int AdjacencyGraph::n_vertices() {
  return adj_list.size();
}

// %%
int AdjacencyGraph::n_edges() {
  return e;
}

// %%
int AdjacencyGraph::add_vertex(std::string label) {
  vii vertex;
  adj_list.push_back(vertex);
  labels.push_back(label);
  return adj_list.size() - 1;
}

// %%
void AdjacencyGraph::add_edge(int u, int v, int weight) {
  for (auto &[v_old, w] : adj_list[u])
    if (v_old == v)
      return;

  adj_list[u].emplace_back(v, weight);
  e++;
}

// %%
AdjacencyGraph AdjacencyGraph::transpose() {
  AdjacencyGraph graph;
  for (int u = 0; u < adj_list.size(); u++)
    graph.add_vertex(labels[u]);

  for (int u = 0; u < adj_list.size(); u++)
    for (auto &[v, w] : adj_list[u])
      graph.add_edge(v, u, w);

  return graph;
}

// %% [markdown]
// ### Core Searches

// %%
void AdjacencyGraph::breadth_first_search(int u, std::function<void(int)> visitor) {
  std::vector<bool> visited(adj_list.size());
  std::queue<int> queue;
  queue.push(u);

  while (!queue.empty()) {
    u = queue.front();
    queue.pop();

    if (!visited[u]) {
      visitor(u);
      visited[u] = true;
    }

    for (auto &[v, w] : adj_list[u]) {
      if (!visited[v])
        queue.push(v);
    }
  }
}

// %%
void AdjacencyGraph::depth_first_search(int u, std::function<void(int)> visitor) {
  std::vector<bool> visited(adj_list.size());
  std::stack<int> stack;
  stack.push(u);

  while (!stack.empty()) {
    u = stack.top();
    stack.pop();

    if (!visited[u]) {
      visitor(u);
      visited[u] = true;
    }

    for (auto &[v, w] : adj_list[u]) {
      if (visited[v])
        continue;
      stack.push(v);
    }
  }
}

// %% [markdown]
// ### Shortest Paths

// %%
std::pair<vi, vi> AdjacencyGraph::dijkstra(int s) {
  vi dist(adj_list.size(), INT_MAX);
  dist[s] = 0;

  vi prev(adj_list.size(), INT_MAX);

  std::priority_queue<ii, std::vector<ii>, std::greater<ii>> pq;
  pq.push({0, s});

  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();

    for (auto &[v, w] : adj_list[u]) {
      if (dist[u] + w < dist[v]) {
        dist[v] = dist[u] + w;
        prev[v] = u;
        pq.push({dist[v], v});
      }
    }
  }

  return {dist, prev};
}

// %%
vvivvi AdjacencyGraph::floyd_warshall() {
  std::vector<vi> dist(adj_list.size());
  std::vector<vi> next(adj_list.size());
  dist.assign(adj_list.size(), vi(adj_list.size(), INT_MAX));
  next.assign(adj_list.size(), vi(adj_list.size(), -1));

  for (int u = 0; u < adj_list.size(); ++u) {
    dist[u][u] = 0;
    next[u][u] = u;

    for (auto &[v, w] : adj_list[u]) {
      dist[u][v] = w;
      next[u][v] = v;
    }
  }

  for (int k = 0; k < adj_list.size(); ++k) {
    for (int i = 0; i < adj_list.size(); ++i) {
      if (dist[i][k] == INT_MAX)
        continue;

      for (int j = 0; j < adj_list.size(); ++j) {
        if (dist[k][j] == INT_MAX)
          continue;

        if (dist[i][j] > dist[i][k] + dist[k][j]) {
          dist[i][j] = dist[i][k] + dist[k][j];
          next[i][j] = next[i][k];
        }
      }
    }
  }

  return {dist, next};
}

// %% [markdown]
// ### Minimum Spanning Tree

// %%
AdjacencyGraph AdjacencyGraph::kruskal() {
  std::vector<std::tuple<int, int, int>> edges;
  for (int u = 0; u < adj_list.size(); ++u) {
    for (auto &[v, w] : adj_list[u]) {
      edges.emplace_back(u, v, w);
    }
  }

  auto compare = [](std::tuple<int, int, int> a, std::tuple<int, int, int> b) {
    return std::get<2>(a) < std::get<2>(b);
  };
  std::sort(edges.begin(), edges.end(), compare);

  UnionFind uf(adj_list.size());
  std::vector<std::tuple<int, int, int>> mst;
  for (auto &[u, v, w] : edges) {
    if (uf.unite(u, v)) {
      mst.emplace_back(u, v, w);
    }
  }

  AdjacencyGraph graph;
  for (int u = 0; u < adj_list.size(); ++u)
    graph.add_vertex(labels[u]);
  for (auto &[u, v, w] : mst)
    graph.add_edge(u, v, w);

  return graph;
}

// %%
AdjacencyGraph AdjacencyGraph::prim() {
  vi parent(adj_list.size(), -1);
  vi key(adj_list.size(), INT_MAX);
  std::vector<bool> included(adj_list.size());

  std::priority_queue<ii, vii, std::greater<ii>> pq;
  pq.push({0, 0});
  key[0] = 0;

  while (!pq.empty()) {
    auto [w, u] = pq.top();
    pq.pop();

    if (included[u])
      continue;

    included[u] = true;

    for (auto &[v, w] : adj_list[u]) {
      if (!included[v] && w < key[v]) {
        parent[v] = u;
        key[v] = w;
        pq.push({w, v});
      }
    }
  }

  AdjacencyGraph graph;
  for (int u = 0; u < adj_list.size(); ++u)
    graph.add_vertex(labels[u]);
  for (int v = 0; v < parent.size(); ++v)
    if (parent[v] != -1)
      graph.add_edge(parent[v], v, key[v]);

  return graph;
}

// %% [markdown]
// ### Connected Components

// %%
bool AdjacencyGraph::kosaraju_test() {
  std::vector<bool> visited(adj_list.size());
  auto visitor = [&visited](int u) { visited[u] = true; };

  depth_first_search(0, visitor);

  for (bool v : visited)
    if (!v)
      return false;

  AdjacencyGraph graph_t = transpose();
  std::fill(visited.begin(), visited.end(), false);
  graph_t.depth_first_search(0, visitor);

  for (bool v : visited)
    if (!v)
      return false;

  return true;
}

// %%
std::vector<vi> AdjacencyGraph::kosaraju() {
  std::vector<bool> visited(adj_list.size());
  vi order;
  auto visitor = [&visited, &order](int u) {
    if (!visited[u]) {
      visited[u] = true;
      order.push_back(u);
    }
  };

  for (int u = 0; u < adj_list.size(); u++) {
    if (!visited[u])
      depth_first_search(u, visitor);
  }

  AdjacencyGraph graph_t = transpose();
  std::fill(visited.begin(), visited.end(), false);
  std::reverse(order.begin(), order.end());

  vi component;
  std::vector<vi> components;
  auto visitor_b = [&visited, &component](int u) {
    if (!visited[u]) {
      visited[u] = true;
      component.push_back(u);
    }
  };

  for (int u : order) {
    if (!visited[u]) {
      component.clear();
      graph_t.depth_first_search(u, visitor_b);
      components.push_back(std::move(component));
    }
  }

  return components;
}

// %%
std::vector<vi> AdjacencyGraph::tarjan() {
  int index = 0;
  vi indexes(adj_list.size());
  vi low_link(adj_list.size());
  std::vector<bool> indexes_has(adj_list.size());
  std::stack<int> stack;
  std::vector<bool> stack_has(adj_list.size());

  std::vector<vi> components;

  auto tj = [this, &components, &index, &indexes, &indexes_has, &low_link, &stack,
             &stack_has](int u) {
    std::stack<ii> call_stack;
    call_stack.push({u, 0});

    while (!call_stack.empty()) {
      auto [u, i] = call_stack.top();
      call_stack.pop();

      if (!indexes_has[u]) {
        low_link[u] = indexes[u] = ++index;
        indexes_has[u] = true;
        stack.push(u);
        stack_has[u] = true;
      }

      bool finished = true;
      for (; i < adj_list[u].size(); i++) {
        auto &[v, w] = adj_list[u][i];

        if (!indexes_has[v]) {
          call_stack.push({u, i + 1});
          call_stack.push({v, 0});
          finished = false;

          break;
        } else if (stack_has[v]) {
          low_link[u] = std::min(low_link[u], indexes[v]);
        }
      }

      if (!finished)
        continue;

      for (auto &[v, w] : adj_list[u])
        if (stack_has[v])
          low_link[u] = std::min(low_link[u], low_link[v]);

      if (low_link[u] == indexes[u]) {
        vi component;
        while (!stack.empty()) {
          int v = stack.top();
          stack.pop();
          stack_has[v] = false;
          component.push_back(v);

          if (v == u)
            break;
        }

        components.push_back(std::move(component));
      }
    }
  };

  for (int u = 0; u < adj_list.size(); u++) {
    if (!indexes_has[u])
      tj(u);
  }

  return components;
}

// %% [markdown]
// ### Usage

// %%
class GraphViewer {
  AdjacencyGraph graph;

public:
  GraphViewer(AdjacencyGraph g) : graph(g) {}

  void info();
  void display();

  std::string undecorate(int u);
  vi reconstruct_path(int t, vi &prev);
  vi reconstruct_path(int s, int t, std::vector<vi> &next);
};

// %%
void GraphViewer::info() {
  std::cout << "|V|: " << graph.n_vertices() << ", " << "|E|: " << graph.n_edges()
            << std::endl;
}

// %%
void GraphViewer::display() {
  for (int u = 0; u < graph.adj_list.size(); u++)
    for (auto &[v, w] : graph.adj_list[u])
      std::cout << graph.labels[u] << " -> " << graph.labels[v] << " " << w
                << std::endl;
}

// %%
std::string GraphViewer::undecorate(int v) {
  return graph.labels[v];
}

// %%
vi GraphViewer::reconstruct_path(int t, vi &prev) {
  vi path;
  for (int i = t; i != INT_MAX; i = prev[i])
    path.push_back(i);

  std::reverse(path.begin(), path.end());
  return path;
}

// %%
vi GraphViewer::reconstruct_path(int s, int t, std::vector<vi> &next) {
  if (next[s][t] == -1)
    return {};

  vi path;
  for (int u = s; u != t; u = next[u][t]) {
    if (u == -1)
      return {};

    path.push_back(u);
  }

  path.push_back(t);
  return path;
}

// %%
AdjacencyGraph graph;
int a = graph.add_vertex("a");
int b = graph.add_vertex("b");
int c = graph.add_vertex("c");
int d = graph.add_vertex("d");
int e = graph.add_vertex("e");
int f = graph.add_vertex("f");
int g = graph.add_vertex("g");

graph.add_edge(a, b, 1);
graph.add_edge(b, c, 2);
graph.add_edge(b, d, 1);
graph.add_edge(c, d, 2);
graph.add_edge(c, f, 3);
graph.add_edge(d, a, 4);
graph.add_edge(d, e, 2);
graph.add_edge(e, f, 1);
graph.add_edge(f, g, 1);
graph.add_edge(g, e, 3);

// %%
GraphViewer gv(graph);
gv.info();
gv.display();

// %%
AdjacencyGraph graph_t = graph.transpose();
GraphViewer gvt(graph_t);
gvt.info();
gvt.display();

// %%
auto visitor = [](int u) { std::cout << "visited " << gv.undecorate(u) << std::endl; };
graph.depth_first_search(a, visitor);

// %%
graph.breadth_first_search(a, visitor);

// %%
std::pair<vi, vi> result = graph.dijkstra(a);

for (int v : gv.reconstruct_path(g, result.second))
  std::cout << gv.undecorate(v) << " ";

// %%
std::pair<std::vector<vi>, std::vector<vi>> result = graph.floyd_warshall();

for (int v : gv.reconstruct_path(a, g, result.second))
  std::cout << gv.undecorate(v) << " ";
std::cout << std::endl;

for (int v : gv.reconstruct_path(c, e, result.second))
  std::cout << gv.undecorate(v) << " ";
std::cout << std::endl;

for (int v : gv.reconstruct_path(d, f, result.second))
  std::cout << gv.undecorate(v) << " ";

// %%
AdjacencyGraph mst_k = graph.kruskal();
GraphViewer gv_mst_k(mst_k);
gv_mst_k.info();
gv_mst_k.display();

// %%
AdjacencyGraph mst_p = graph.prim();
GraphViewer gv_mst_p(mst_p);
gv_mst_p.info();
gv_mst_p.display();

// %%
bool is_strongly_connected = graph.kosaraju_test();
std::vector<vi> components = graph.kosaraju();

std::cout << is_strongly_connected << std::endl;
for (vi component : components) {
  for (int v : component)
    std::cout << gv.undecorate(v) << " ";

  std::cout << std::endl;
}

// %%
std::vector<vi> components = graph.tarjan();

for (vi component : components) {
  for (int v : component)
    std::cout << gv.undecorate(v) << " ";

  std::cout << std::endl;
}

// %%
