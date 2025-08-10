use crate::expr::qm31::{GpuM31, GpuQM31};
use super::constants::{ConstraintConfig, DefaultConfig};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuExtendedColumn<const N_EXTENDED_ROWS: usize> {
    pub data: [GpuM31; N_EXTENDED_ROWS],
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct ComputeCompositionPolynomialInput<
    const N_EXTENDED_ROWS: usize,
    const N_CONSTRAINTS: usize,
    const N_COLUMNS: usize,
    const N_LOOKUP_ELEMENTS: usize,
> {
    pub extended_trace: [GpuExtendedColumn<N_EXTENDED_ROWS>; N_COLUMNS],
    pub denom_inv: [GpuM31; 4],
    pub random_coeff_powers: [GpuQM31; N_CONSTRAINTS],
    pub lookup_elements: GpuLookupElements<N_LOOKUP_ELEMENTS>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct GpuLookupElements<const N: usize> {
    pub z: GpuQM31,
    pub alpha: GpuQM31,
    pub alpha_powers: [GpuQM31; N],
}

impl<const N: usize> GpuLookupElements<N> {
    pub fn dummy() -> Self {
        use stwo::core::fields::qm31::SecureField;
        use num_traits::One;
        
        Self {
            z: GpuQM31::from(SecureField::from_u32_unchecked(1, 2, 3, 4)),
            alpha: GpuQM31::from(SecureField::one()),
            alpha_powers: [GpuQM31::from(SecureField::one()); N],
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct ComputeCompositionPolynomialOutput<
    const N_LANES: usize,
    const N_PACKED_ROWS: usize,
    const N_INTERMEDIATES: usize,
    const N_EXT_INTERMEDIATES: usize,
> {
    pub poly: [[GpuQM31; N_LANES]; N_PACKED_ROWS],
    pub intermediates: [GpuM31; N_INTERMEDIATES],
    pub ext_intermediates: [GpuQM31; N_EXT_INTERMEDIATES],
}

// Convenience type aliases for default configuration
pub type DefaultGpuExtendedColumn = GpuExtendedColumn<{ DefaultConfig::N_EXTENDED_ROWS as usize }>;
pub type DefaultGpuLookupElements = GpuLookupElements<{ DefaultConfig::N_LOOKUP_ELEMENTS as usize }>;
pub type DefaultComputeInput = ComputeCompositionPolynomialInput<
    { DefaultConfig::N_EXTENDED_ROWS as usize },
    { DefaultConfig::N_CONSTRAINTS as usize },
    { DefaultConfig::N_COLUMNS as usize },
    { DefaultConfig::N_LOOKUP_ELEMENTS as usize },
>;
pub type DefaultComputeOutput = ComputeCompositionPolynomialOutput<
    { DefaultConfig::N_LANES as usize },
    { DefaultConfig::N_PACKED_ROWS as usize },
    { DefaultConfig::N_INTERMEDIATES as usize },
    { DefaultConfig::N_EXT_INTERMEDIATES as usize },
>;
