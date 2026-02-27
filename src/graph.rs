use std::collections::{BinaryHeap, VecDeque};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

struct Graph<T> {
	adjacency_list: Vec<Vec<(u32, u32)>>,
	labels: Vec<T>,
}

impl<T: PartialEq> Graph<T> {
	fn new() -> Self {
		Graph {
			adjacency_list: Vec::new(),
			labels: Vec::new(),
		}
	}

	fn add_node(&mut self, label: T) {
		self.adjacency_list.push(Vec::new());
		self.labels.push(label);
	}

	fn add_edge(&mut self, from: T, to: T, weight: u32) {
		let u = match self.labels.iter().position(|r| r == &from) {
			Some(i) => i,
			None => return,
		};

		let v_new = match self.labels.iter().position(|r| r == &to) {
			Some(i) => i as u32,
			None => return,
		};

		for &(v, _w) in &self.adjacency_list[u] {
			if v == v_new {
				return;
			}
		}

		self.adjacency_list[u].push((v_new, weight));
	}
}

fn dijkstra<T: Clone + PartialEq>(graph: &Graph<T>, start: T, end: T) -> Option<(Vec<T>, u32)> {
	let start_index = match graph.labels.iter().position(|r| r == &start) {
		Some(i) => i,
		None => return None,
	};

	let end_index = match graph.labels.iter().position(|r| r == &end) {
		Some(i) => i,
		None => return None,
	};

	let mut previous = vec![u32::MAX; graph.adjacency_list.len()];
	let mut distance = vec![u32::MAX; graph.adjacency_list.len()];
	distance[start_index] = 0;

	let mut pq = BinaryHeap::new();
	pq.push((0, start_index));

	while let Some((d, u)) = pq.pop() {
		if d > distance[u] {
			continue;
		}

		for &(v, w) in &graph.adjacency_list[u] {
			let v = v as usize;

			if distance[u].saturating_add(w) >= distance[v] {
				continue;
			}

			previous[v] = u as u32;
			distance[v] = distance[u] + w;

			pq.push((distance[v], v));
		}
	}

	let mut path = VecDeque::new();
	let mut current = end_index;

	while previous[current] != u32::MAX {
		path.push_front(graph.labels[current].clone());
		current = previous[current] as usize;
	}

	path.push_front(graph.labels[current].clone());

	Some((Vec::from(path), distance[end_index]))
}

macro_rules! create_graph_interface {
	($name: ident, $type: ident) => {
		#[pyclass]
		pub struct $name {
			inner: Graph<$type>,
		}

		#[pymethods]
		impl $name {
			#[new]
			pub fn new() -> Self {
				Self {
					inner: Graph::new(),
				}
			}

			pub fn add_node(&mut self, label: $type) {
				self.inner.add_node(label);
			}

			pub fn add_edge(&mut self, from: $type, to: $type, weight: u32) {
				self.inner.add_edge(from, to, weight);
			}
		}

		impl Default for $name {
			fn default() -> Self {
				Self::new()
			}
		}
	};
}

create_graph_interface!(GraphString, String);

#[pyfunction]
#[pyo3(name = "dijkstra")]
pub fn dijkstra_string(
	graph: &mut GraphString, start: String, end: String,
) -> PyResult<(Vec<String>, u32)> {
	match dijkstra(&graph.inner, start, end) {
		Some((path, distance)) => Ok((path, distance)),
		None => Err(PyValueError::new_err("Invalid start or end node.")),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn graph() {
		let mut graph = Graph::new();

		graph.add_node("A");
		graph.add_node("B");
		graph.add_node("C");

		graph.add_edge("A", "B", 1);
		graph.add_edge("B", "C", 2);

		assert_eq!(graph.adjacency_list[0], vec![(1, 1)]);
		assert_eq!(graph.adjacency_list[1], vec![(2, 2)]);

		assert_eq!(graph.labels[0], "A");
		assert_eq!(graph.labels[1], "B");
		assert_eq!(graph.labels[2], "C");
	}

	#[test]
	fn dijkstra() {
		let mut graph = Graph::new();

		graph.add_node("A");
		graph.add_node("B");
		graph.add_node("C");
		graph.add_node("D");
		graph.add_node("E");

		graph.add_edge("A", "B", 5);
		graph.add_edge("B", "C", 6);
		graph.add_edge("C", "D", 2);
		graph.add_edge("A", "C", 15);

		let (path, distance) = super::dijkstra(&graph, "A", "D").unwrap();

		assert_eq!(path, vec!["A", "B", "C", "D"]);
		assert_eq!(distance, 13);
	}
}
