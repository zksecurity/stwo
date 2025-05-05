use wasm_bindgen::prelude::wasm_bindgen;

use super::{ComputeCompositionPolynomialInput, ComputeCompositionPolynomialOutput};

#[wasm_bindgen]
pub fn comp_poly_input_length() -> usize {
    std::mem::size_of::<ComputeCompositionPolynomialInput>()
}

#[wasm_bindgen]
pub fn comp_poly_output_length() -> usize {
    std::mem::size_of::<ComputeCompositionPolynomialOutput>()
}
