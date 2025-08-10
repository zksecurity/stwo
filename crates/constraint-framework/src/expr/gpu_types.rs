use crate::expr::qm31::{GpuM31, GpuQM31};
use super::constants::*;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuExtendedColumn {
    pub data: [GpuM31; N_EXTENDED_ROWS as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct ComputeCompositionPolynomialInput {
    pub extended_trace: [GpuExtendedColumn; N_COLUMNS as usize],
    pub denom_inv: [GpuM31; 4],
    pub random_coeff_powers: [GpuQM31; N_CONSTRAINTS as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct ComputeCompositionPolynomialOutput {
    pub poly: [[GpuQM31; N_LANES as usize]; N_PACKED_ROWS as usize],
}
