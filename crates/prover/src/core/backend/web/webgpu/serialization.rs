use super::constants::*;
use super::qm31::GpuM31;
use super::{
    ComputeCompositionPolynomialInput, ComputeCompositionPolynomialOutput, GpuExtendedColumn,
    GpuLookupElements, GpuOriginalColumn,
};
use crate::core::backend::web::WebBackend;
use crate::core::backend::Column;
use crate::core::poly::circle::CirclePoly;
use crate::examples::poseidon::PoseidonElements;

pub trait ByteSerialize: Sized {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    fn from_bytes(bytes: &[u8]) -> &Self {
        assert!(bytes.len() >= std::mem::size_of::<Self>());
        unsafe { &*(bytes.as_ptr() as *const Self) }
    }
}

impl ByteSerialize for GpuExtendedColumn {}
impl ByteSerialize for GpuOriginalColumn {}
impl ByteSerialize for ComputeCompositionPolynomialOutput {}
impl ByteSerialize for ComputeCompositionPolynomialInput {}

impl ComputeCompositionPolynomialOutput {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        unsafe { *(bytes.as_ptr() as *const Self) }
    }
}

impl From<&&CirclePoly<WebBackend>> for GpuOriginalColumn {
    fn from(value: &&CirclePoly<WebBackend>) -> Self {
        let mut coeffs = [GpuM31 { data: 0 }; (N_LANES * N_ORIGINAL_ROWS) as usize];
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
