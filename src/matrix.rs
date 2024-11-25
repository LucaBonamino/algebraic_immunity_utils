use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Matrix {
    pub elements: Vec<Vec<u8>>,
}

#[pymethods]
impl Matrix {
    #[new]
    pub fn new(elements: Vec<Vec<u8>>) -> Self {
        Matrix { elements }
    }

    pub fn __repr__(&self) -> String {
        let rows: Vec<String> = self
            .elements
            .iter()
            .map(|row| format!("{:?}", row))
            .collect();
        format!("[{}]", rows.join(", "))
    }
    //
    fn nrows(&self) -> usize {
        self.elements.len()
    }

    fn ncols(&self) -> usize {
        if !self.elements.is_empty() {
            self.elements[0].len()
        } else {
            0
        }
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn get(&self, row: usize, col: usize) -> u8 {
        self.elements[row][col]
    }

    fn add_rows(&mut self, target: usize, source: usize) {
        for i in 0..self.ncols() {
            self.elements[target][i] ^= self.elements[source][i];
        }
    }

    fn swap_rows(&mut self, row1: usize, row2: usize) {
        self.elements.swap(row1, row2);
    }

    fn is_zero_row(&self, row: usize) -> bool {
        self.elements[row].iter().all(|&x| x == 0)
    }

    pub fn echf_2(&mut self) -> (Self, Vec<(usize, usize)>) {
        let mut m_copy = self.copy();
        let mut last_row = m_copy.elements[m_copy.nrows() - 1].clone();
        let last_row_index = m_copy.nrows() - 1;
        let mut operations = Vec::new();

        for _ in 0..m_copy.ncols() {
            let p_index = Matrix::get_pivot(&last_row);
            if p_index.is_none() {
                break;
            }
            let p_index = p_index.unwrap();

            let mut p_row: Option<Vec<u8>> = None;
            let mut j_index: Option<usize> = None;

            for j in 0..m_copy.nrows() - 1 {
                if m_copy.get(j, p_index) == 1 && !(0..p_index).any(|k| m_copy.get(j, k) == 1) {
                    p_row = Some(m_copy.elements[j].clone());
                    j_index = Some(j);
                }
            }

            if p_row.is_none() {
                if p_index == last_row_index {
                    let mut swap_index_row: Option<usize> = None;
                    for r in 0..m_copy.nrows() - 1 {
                        if m_copy.is_zero_row(r) {
                            swap_index_row = Some(r);
                            break;
                        }
                    }

                    if let Some(swap_index_row) = swap_index_row {
                        m_copy.swap_rows(last_row_index, swap_index_row);
                        operations.push((swap_index_row, last_row_index));
                        operations.push((last_row_index, swap_index_row));
                        operations.push((swap_index_row, last_row_index));
                    }
                    break;
                }
                m_copy.swap_rows(last_row_index, p_index);
                last_row = m_copy.elements[last_row_index].clone();
                operations.push((p_index, last_row_index));
                operations.push((last_row_index, p_index));
                operations.push((p_index, last_row_index));
            } else if p_row.unwrap()[p_index] == 1 {
                m_copy.add_rows(last_row_index, j_index.unwrap());
                last_row = m_copy.elements[last_row_index].clone();
                operations.push((last_row_index, j_index.unwrap()));
            }
        }

        (m_copy, operations)
    }

    fn append_row(&mut self, v: Vec<u8>) {
        self.elements.push(v)
    }

    fn append_column(&mut self, v: Vec<u8>) {
        for i in 0..self.nrows() {
            self.elements[i].push(v[i]);
        }
    }

    fn rank(&self) -> usize {
        let mut count = 0;
        for i in 0..self.nrows() {
            let p = Matrix::get_pivot(&self.elements[i]);
            if p.is_none() {
                break;
            } else {
                count += 1
            }
        }
        count
    }

    fn kernel(&self) -> Vec<Vec<u8>> {
        let rows = self.nrows();
        let cols = self.ncols();

        let mut pivots: Vec<usize> = Vec::new();
        let mut kernel_base: Vec<Vec<u8>> = Vec::new();
        let mut free_columns: Vec<usize> = Vec::new();
        let mut row = 0;

        println!("Starting kernel computation.");
        println!("Matrix dimensions: {} rows, {} cols", rows, cols);

        // Identify pivot and free columns
        for j in 0..cols {
            if row < rows && self.elements[row][j] == 1 {
                pivots.push(j);
                row += 1;
            } else {
                free_columns.push(j);
            }
        }

        println!("Pivot columns: {:?}", pivots);
        println!("Free columns: {:?}", free_columns);

        // Construct kernel basis vectors
        for &free_col in &free_columns {
            let mut kernel_vector = vec![0; cols];
            kernel_vector[free_col] = 1; // Set the free variable

            println!("Constructing kernel vector for free column {}", free_col);

            // Compute dependent variables for each pivot column
            for (i, &pivot_col) in pivots.iter().enumerate() {
                let mut sum = 0;
                println!("Row {}: Calculating for pivot column {}", i, pivot_col);

                // Sum contributions from non-free columns (pivot columns) in the same row
                for &free_col in &free_columns {
                    println!(
                        "Row {}, Free column {}: Adding matrix[{}][{}] = {}",
                        i, free_col, i, free_col, self.elements[i][free_col]
                    );
                    sum ^= self.elements[i][free_col]; // XOR contributions from free columns
                }
                kernel_vector[pivot_col] = sum; // Assign the computed value to the pivot column
                println!("Result for pivot column {}: {}", pivot_col, sum);
            }

            println!("Constructed kernel vector: {:?}", kernel_vector);

            kernel_base.push(kernel_vector);
        }

        println!("Kernel basis completed: {:?}", kernel_base);

        kernel_base
    }



    fn kernel2(&self)  -> Vec<Vec<u8>> {
        let rows = self.nrows();
        let cols = self.ncols();

        let mut pivots: Vec<usize> = Vec::new();
        let mut kernel_base: Vec<Vec<u8>> = Vec::new();
        let mut free_columns : Vec<usize> = Vec::new();
        let mut row = 0;

        for j in 0..cols{
            if row < rows && self.elements[row][j] == 1 {
                pivots.push(j);
                row += 1;
            }
            else {
                free_columns.push(j);
            }
        }
        println!("Pivot columns: {:?}", pivots);
        println!("Free columns: {:?}", free_columns);

        // Construct kernel basis vectors
        for &free_col in &free_columns {
            let mut kernel_vector = vec![0; cols];
            kernel_vector[free_col] = 1; // Set the free variable

            println!("Constructing kernel vector for free column {}", free_col);

            for (i, &pivot_col) in pivots.iter().enumerate() {
                // Sum all contributions from free columns in the same row
                let mut sum = 0;
                for &free_col in &free_columns {
                    sum ^= self.elements[i][free_col]; // Add the element from the same row
                }
                kernel_vector[pivot_col] = sum; // Set the computed value for the pivot column
            }


            kernel_base.push(kernel_vector);
        }

        kernel_base
    }
}

impl Matrix {
    fn get_pivot(row: &Vec<u8>) -> Option<usize> {
        row.iter().position(|&x| x == 1)
    }
}
