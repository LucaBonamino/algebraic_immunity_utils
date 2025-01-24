use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Matrix {
    pub elements: Vec<Vec<u8>>,
}

#[pymethods]
impl Matrix {
    #[new]
    pub fn new(elements: Vec<Vec<u8>>) -> Self {
        Matrix { elements }
    }

    pub fn get_sub_matrix(&self, start: usize, end: usize) -> Self {
        Self {
            elements: self.elements[start..end].to_vec(),
        }
    }


    pub fn __repr__(&self) -> String {
        let rows: Vec<String> = self
            .elements
            .iter()
            .map(|row| format!("{:?}", row))
            .collect();
        format!("[{}]", rows.join(", "))
    }

    fn to_list(&self) -> Vec<Vec<u8>> {
        self.elements.clone()
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

    fn echelon_form_last_row2(&mut self) -> (Self, Vec<(usize, usize)>) {
        let mut m_copy = self.copy();
        let mut last_row = m_copy.elements[m_copy.nrows() - 1].clone();
        let last_row_index = m_copy.nrows() - 1;
        let mut operations = Vec::new();

        for _ in 0..m_copy.ncols() {
            let p_index = Matrix::get_pivot(&last_row);
            if p_index.is_none() {
                for j in (1..m_copy.nrows()).rev() {
                    if m_copy.is_zero_row(j) {
                        continue;
                    }
                    let curr_pivot = Matrix::get_pivot(&m_copy.elements[j]).unwrap();
                    let prev_pivot = Matrix::get_pivot(&m_copy.elements[j - 1]);
                    //println!("The number is {}\n",curr_pivot);
                    //println!("{}", &m_copy.elements[j-1]);
                    println!("The points are: {:?}", &m_copy.elements[j]);
                    println!("The points are: {:?}", &m_copy.elements[j - 1]);
                    if !prev_pivot.is_none(){
                        println!("The number is {} {}\n",j, prev_pivot.unwrap());
                    }
                    if prev_pivot.is_none()
                        || (!prev_pivot.is_none() && curr_pivot < prev_pivot.unwrap())
                    {
                        m_copy.swap_rows(j, j - 1);
                        operations.push((j, j - 1));
                        operations.push((j - 1, j));
                        operations.push((j, j - 1));
                    } else if !prev_pivot.is_none()
                        && (prev_pivot.unwrap() == curr_pivot)
                    {
                        //println!(prev_pivot.unwrap());
                        println!("The number is {}\n",curr_pivot);
                        // corner case: matrix self.elements[:-1][:-1] was not in echelon form due to the last appended column
                        m_copy.add_rows(j, j - 1);
                        operations.push((j, j - 1));
                    }
                }
                break;
            } else {
                let p_index = p_index.unwrap();

                let mut p_row: Option<Vec<u8>> = None;
                let mut j_index: Option<usize> = None;
                let mut distance_base = m_copy.nrows();
                let mut closest: Option<usize> = None;
                for j in 0..m_copy.nrows() - 1 {
                    let piv: Option<usize> = Matrix::get_pivot(&m_copy.elements[j]);
                    if piv.is_none() {
                        if closest.is_none() {
                            closest = Some(j);
                        }
                        break;
                    } else {
                        let piv_u = piv.unwrap();
                        if piv_u == p_index {
                            p_row = Some(m_copy.elements[j].clone());
                            j_index = Some(j);
                            break;
                        } else {
                            let d: isize = piv_u as isize - p_index as isize;
                            if 0 < d && (d as usize) < distance_base {
                                closest = Some(j);
                                distance_base = d as usize;
                            }
                        }
                    }
                    //if m_copy.get(j, p_index) == 1 && !(0..p_index).any(|k| m_copy.get(j, k) == 1) {
                    //    p_row = Some(m_copy.elements[j].clone());
                    //    j_index = Some(j);
                    //}
                }
                if p_row.is_none() {
                    // if p_index == last_row_index {
                    //     let mut swap_index_row: Option<usize> = None;
                    //     for r in 0..m_copy.nrows() - 1 {
                    //         if m_copy.is_zero_row(r) {
                    //             swap_index_row = Some(r);
                    //             break;
                    //         }
                    //     }
                    //
                    //     if let Some(swap_index_row) = swap_index_row {
                    //         m_copy.swap_rows(last_row_index, swap_index_row);
                    //         operations.push((swap_index_row, last_row_index));
                    //         operations.push((last_row_index, swap_index_row));
                    //         operations.push((swap_index_row, last_row_index));
                    //     }
                    //     break;
                    // }
                    let closest_u = closest.unwrap();
                    m_copy.swap_rows(last_row_index, closest_u);
                    last_row = m_copy.elements[last_row_index].clone();
                    operations.push((closest_u, last_row_index));
                    operations.push((last_row_index, closest_u));
                    operations.push((closest_u, last_row_index));
                } else if p_row.unwrap()[p_index] == 1 {
                    m_copy.add_rows(last_row_index, j_index.unwrap());
                    last_row = m_copy.elements[last_row_index].clone();
                    operations.push((last_row_index, j_index.unwrap()));
                }
            }
        }

        (m_copy, operations)
    }

    fn echelon_form_last_row(&mut self) -> (Self, Vec<(usize, usize)>) {
        let mut m_copy = self.copy();
        let last_row_index = m_copy.nrows() - 1;
        let mut last_row = m_copy.elements[last_row_index].clone();
        let mut operations = Vec::new();

        for _ in 0..m_copy.ncols() {
            let p_index = Matrix::get_pivot(&last_row);

            if p_index.is_none() {
                //println!("{:?}", m_copy);

                for row_index in (1..m_copy.nrows() - 1).rev() {
                    if m_copy.is_zero_row(row_index) {
                        continue;
                    }

                    let curr_pivot = Matrix::get_pivot(&m_copy.elements[row_index]).unwrap();
                    let prev_pivot = Matrix::get_pivot(&m_copy.elements[row_index - 1]);

                    if prev_pivot.is_none() || (curr_pivot < prev_pivot.unwrap()) {
                        m_copy.swap_rows(row_index, row_index - 1);
                        operations.push((row_index, row_index - 1));
                        operations.push((row_index - 1, row_index));
                        operations.push((row_index, row_index - 1));
                    } else if let Some(prev_pivot_value) = prev_pivot {
                        if prev_pivot_value == curr_pivot && prev_pivot_value == m_copy.ncols() - 1 {
                            m_copy.add_rows(row_index, row_index - 1);
                            operations.push((row_index, row_index - 1));
                        }
                    }
                }

                break;
            } else {
                let p_index = p_index.unwrap();
                let mut p_row: Option<Vec<u8>> = None;
                let mut j_index: Option<usize> = None;

                let mut d_base = m_copy.nrows() - 1;
                let mut closest: Option<usize> = None;

                for j in 0..m_copy.nrows() - 1 {
                    if let Some(piv) = Matrix::get_pivot(&m_copy.elements[j]) {
                        if piv == p_index {
                            p_row = Some(m_copy.elements[j].clone());
                            j_index = Some(j);
                            break;
                        } else {
                            let d = piv as isize - p_index as isize;
                            if 0 < d && (d as usize) < d_base {
                                closest = Some(j);
                                d_base = d as usize;
                            }
                        }
                    } else if closest.is_none() {
                        closest = Some(j);
                    }
                }

                if p_row.is_none() {
                    if let Some(closest_u) = closest {
                        m_copy.swap_rows(last_row_index, closest_u);
                        last_row = m_copy.elements[last_row_index].clone();
                        operations.push((closest_u, last_row_index));
                        operations.push((last_row_index, closest_u));
                        operations.push((closest_u, last_row_index));
                    }
                } else if let Some(p_row_value) = &p_row {
                    if p_row_value[p_index] == 1 {
                        m_copy.add_rows(last_row_index, j_index.unwrap());
                        last_row = m_copy.elements[last_row_index].clone();
                        operations.push((last_row_index, j_index.unwrap()));
                    }
                }
            }
        }

        (m_copy, operations)
    }


    fn echelon_form(&self) -> (Matrix, Vec<(usize, usize)>) {
        let mut m_copy = self.copy(); // Create a copy of the matrix
        let mut row = 0;
        let mut operations: Vec<(usize, usize)> = Vec::new();

        for col in 0..self.ncols() {
            // Find the pivot row
            let mut pivot_row = None;
            for r in row..self.nrows() {
                if m_copy.elements[r][col] == 1 {
                    pivot_row = Some(r);
                    break;
                }
            }

            if let Some(pivot_row_index) = pivot_row {
                // Swap the current row with the pivot row
                m_copy.swap_rows(row, pivot_row_index);
                operations.push((row, pivot_row_index));
                operations.push((pivot_row_index, row));
                operations.push((row, pivot_row_index));

                // Eliminate all other 1s in this column
                for r in 0..self.nrows() {
                    if r != row && m_copy.elements[r][col] == 1 {
                        m_copy.add_rows(r, row); // Add the pivot row to eliminate the 1
                        operations.push((r, row));
                    }
                }

                // Move to the next row
                row += 1;
            }
        }

        (m_copy, operations) // Return the new matrix in echelon form
    }

    // fn row_echelon_full_matrix(&self) -> (Self, Vec<(usize, usize)>) {
    //     let mut operations: Vec<(usize, usize)> = Vec::new();
    //     let mut m_copy = self.copy();
    //     let mut first = 0;
    //
    //     for r in 0..m_copy.nrows() {
    //         let mut i = r.clone();
    //         while m_copy.elements[i][first] == 0 {
    //             i += 1;
    //             if i == m_copy.nrows() {
    //                 i = r;
    //                 first = i;
    //             }
    //
    //         }
    //
    //     }
    //
    //     for i in 0..usize::min(self.nrows(), self.ncols()) {
    //         // Find the pivot in the current column (the first 1 in column i)
    //         let mut pivot_row: Option<usize> = None;
    //         for r in i..self.nrows() {
    //             if m_copy.get(r, i) == 1 {
    //                 pivot_row = Some(r);
    //                 break;
    //             }
    //         }
    //
    //         // If no pivot is found, skip this column
    //         if pivot_row.is_none() {
    //             continue;
    //         }
    //         let pivot_row = pivot_row.unwrap();
    //
    //         // Swap the current row with the pivot row
    //         if pivot_row != i {
    //             m_copy.swap_rows(i, pivot_row);
    //             operations.push((i, pivot_row));
    //             operations.push((pivot_row, i));
    //             operations.push((i, pivot_row));
    //         }
    //
    //         // Eliminate all rows below the pivot
    //         for j in (i + 1)..self.nrows() {
    //             if m_copy.get(j, i) == 1 {
    //                 m_copy.add_rows(j, i);
    //                 operations.push((j, i));
    //             }
    //         }
    //     }
    //
    //     (m_copy, operations)
    // }

    fn row_echelon_full_matrix(&self) -> (Self, Vec<(usize, usize)>) {
        let mut m_copy = self.clone();
        let rows = m_copy.nrows();
        let cols = m_copy.ncols();
        let mut operations: Vec<(usize, usize)> = Vec::new();
        let mut lead = 0;

        for r in 0..rows {
            if lead >= cols {
                break;
            }
            let mut i = r;
            while m_copy.elements[i][lead] == 0 {
                i += 1;
                if i == rows {
                    i = r;
                    lead += 1;
                    if lead == cols {
                        return (m_copy, operations);
                    }
                }
            }
            m_copy.swap_rows(r, i);
            if r != i {
                operations.push((r, i));
                operations.push((i, r));
                operations.push((r, i));
            }
            for i in 0..rows {
                if i != r && m_copy.elements[i][lead] == 1 {
                    for j in 0..cols {
                        m_copy.elements[i][j] = (m_copy.elements[i][j] + m_copy.elements[r][j]) % 2;
                    }
                    operations.push((i, r));
                }
            }
            lead += 1;
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
        let mut pivot_columns = std::collections::HashSet::new();

        for i in 0..self.nrows() {
            let p = Matrix::get_pivot(&self.elements[i]);
            if let Some(col) = p {
                if pivot_columns.insert(col) {
                    count += 1;
                }
            }
        }
        count
    }

    fn kernel(&self) -> Vec<Vec<u8>> {
        let rows = self.nrows();
        let cols = self.ncols();

        let mut pivots = std::collections::HashMap::new();
        let mut kernel_base: Vec<Vec<u8>> = Vec::new();
        let mut free_columns: Vec<usize> = Vec::new();
        let mut row_index = 0;

        for j in 0..cols {
            if row_index < rows && self.elements[row_index][j] == 1 {
                pivots.insert(j, row_index);
                row_index += 1;
            } else {
                free_columns.push(j);
            }
        }

        for &free_col in &free_columns {
            let mut kernel_vector = vec![0; cols];
            kernel_vector[free_col] = 1;

            for (&p_index, &p_row) in &pivots {
                let mut sum = 0;

                for col in (0..cols).rev() {
                    if col != p_index {
                        sum = sum ^ (self.elements[p_row][col] * kernel_vector[col]);
                    }
                }

                kernel_vector[p_index] = sum;
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
