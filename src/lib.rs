use std::time::Instant;

use pyo3::prelude::*;

mod graph;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
	Ok((a + b).to_string())
}

#[pyfunction]
fn fibonacci(n: i32) -> PyResult<i64> {
	let mut a = 0;
	let mut b = 1;

	if n == 0 {
		return Ok(a);
	}

	for _ in 1..n {
		let c = a + b;
		a = b;
		b = c;
	}

	Ok(b)
}

fn collatz(mut n: u64) -> u16 {
	let mut steps = 0;

	while n != 1 {
		if n.is_multiple_of(2) {
			n /= 2;
		} else {
			n = 3 * n + 1;
		}

		steps += 1;
	}

	steps
}

#[pyfunction]
fn collatz_repeat(n: u32) {
	let mut sequence = Vec::new();

	let metric_period: u32 = n / 10;
	let start_time = Instant::now();

	for i in 1..n + 1 {
		let steps = collatz(i as u64);

		if i % metric_period == 0 {
			sequence.push((i, steps));
		}
	}

	let elapsed_time = start_time.elapsed().as_millis();
	println!("Elapsed time: {} ms", elapsed_time);

	for (i, steps) in sequence {
		println!("n: {}, steps: {}", i, steps);
	}
}

#[pymodule]
fn rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
	m.add_function(wrap_pyfunction!(fibonacci, m)?)?;
	m.add_function(wrap_pyfunction!(collatz_repeat, m)?)?;

	m.add_class::<graph::GraphString>()?;
	m.add_function(wrap_pyfunction!(graph::dijkstra_string, m)?)?;

	Ok(())
}
