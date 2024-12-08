mod matrix;
use pyo3::prelude::*;

#[pyfunction]
fn echelon_form_last_row(x: Vec<Vec<u8>>) -> (Vec<Vec<u8>>, Vec<(usize, usize)>) {
    let mut matrix = matrix::Matrix::new(x);
    //let mut matrix = matrix::Matrix { elements: x };
    let (m, b) = matrix.echf_2();
    (m.elements, b)
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn algebraic_immunity_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(echelon_form_last_row, m)?)?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<matrix::Matrix>()?;
    Ok(())
}
