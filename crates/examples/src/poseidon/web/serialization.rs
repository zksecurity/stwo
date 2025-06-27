use stwo_prover::core::backend::web::webgpu::qm31::GpuM31;
use stwo_prover::core::backend::web::webgpu::ByteSerialize;
use stwo_prover::core::backend::web::WebBackend;
use stwo_prover::core::backend::Column;
use stwo_prover::core::poly::circle::CirclePoly;

use crate::poseidon::web::{
    ComputeCompositionPolynomialInput, ComputeCompositionPolynomialOutput, GpuExtendedColumn,
    GpuLookupElements, GpuOriginalColumn, N_LANES, N_ORIGINAL_ROWS,
};
use crate::poseidon::PoseidonElements;

impl ByteSerialize for GpuExtendedColumn {}
impl ByteSerialize for GpuOriginalColumn {}
impl ByteSerialize for ComputeCompositionPolynomialOutput {}
impl ByteSerialize for ComputeCompositionPolynomialInput {}

#[allow(dead_code)]
impl ComputeCompositionPolynomialInput {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), std::mem::size_of::<Self>());
        unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const Self) }
    }
}

#[allow(dead_code)]
impl ComputeCompositionPolynomialOutput {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), std::mem::size_of::<Self>());
        unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const Self) }
    }
}

impl From<&&CirclePoly<WebBackend>> for GpuOriginalColumn {
    fn from(value: &&CirclePoly<WebBackend>) -> Self {
        let mut coeffs = [GpuM31 { 0: 0 }; (N_LANES * N_ORIGINAL_ROWS) as usize];
        let coeffs_vec = value.coeffs.to_cpu();
        for (i, &coeff) in coeffs_vec.iter().enumerate() {
            coeffs[i] = coeff.into();
        }

        GpuOriginalColumn { coeffs }
    }
}

impl From<&PoseidonElements> for GpuLookupElements {
    fn from(value: &PoseidonElements) -> Self {
        GpuLookupElements {
            z: value.0.z.into(),
            alpha: value.0.alpha.into(),
            alpha_powers: value
                .0
                .alpha_powers
                .iter()
                .map(|&x| x.into())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}
