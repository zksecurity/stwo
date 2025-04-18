use std::ops::Mul;

use super::INTERACTION_TRACE_IDX;
use crate::constraint_framework::logup::LogupAtRow;
use crate::constraint_framework::EvalAtRow;
use crate::core::backend::simd::very_packed_m31::{VeryPackedBaseField, VeryPackedSecureField};
use crate::core::backend::web::WebBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SECURE_EXTENSION_DEGREE;
use crate::core::lookups::utils::Fraction;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::{CircleEvaluation, CirclePoly};
use crate::core::poly::BitReversedOrder;

/// Dummy evaluator for WebGPU.
pub struct WebDomainEvaluator<'a> {
    pub trace_poly: &'a TreeVec<Vec<&'a CirclePoly<WebBackend>>>,
    pub trace_eval: &'a TreeVec<Vec<&'a CircleEvaluation<WebBackend, BaseField, BitReversedOrder>>>,
    pub needs_to_extend: bool,
    pub logup: LogupAtRow<Self>,
}
impl<'a> WebDomainEvaluator<'a> {
    pub fn new(
        trace_poly: &'a TreeVec<Vec<&'a CirclePoly<WebBackend>>>,
        trace_eval: &'a TreeVec<Vec<&'a CircleEvaluation<WebBackend, BaseField, BitReversedOrder>>>,
        needs_to_extend: bool,
        log_size: u32,
        claimed_sum: SecureField,
    ) -> Self {
        Self {
            trace_poly,
            trace_eval,
            needs_to_extend,
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, claimed_sum, log_size),
        }
    }
}

/// Dummy implementation for WebGPU. These methods will be implemented as WGSL code, so they don't
/// need to be implemented here.
#[allow(unused_variables)]
impl EvalAtRow for WebDomainEvaluator<'_> {
    type F = VeryPackedBaseField;
    type EF = VeryPackedSecureField;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        unimplemented!()
    }
    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: Mul<G, Output = Self::EF> + From<G>,
    {
        unimplemented!()
    }

    fn combine_ef(values: [Self::F; SECURE_EXTENSION_DEGREE]) -> Self::EF {
        unimplemented!()
    }

    super::logup_proxy!();
}
