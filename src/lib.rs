use pyo3::prelude::*;

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

#[pymodule]
fn camp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(fibonacci, m)?)?;

    Ok(())
}
