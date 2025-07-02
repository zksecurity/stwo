use std::ops::Mul;
use std::rc::Rc;

use num_traits::{One, Zero};
use stwo_prover::core::backend::simd::column::VeryPackedSecureColumnByCoords;
use stwo_prover::core::backend::simd::very_packed_m31::{
    VeryPackedBaseField, VeryPackedSecureField,
};
use stwo_prover::core::backend::web::WebBackend;
use stwo_prover::core::fields::m31::{BaseField, M31};
use stwo_prover::core::fields::qm31::{SecureField, QM31, SECURE_EXTENSION_DEGREE};
use stwo_prover::core::fields::FieldExpOps;
use stwo_prover::core::lookups::utils::Fraction;
use stwo_prover::core::pcs::TreeVec;
use stwo_prover::core::poly::circle::{CircleDomain, CircleEvaluation, CirclePoly};
use stwo_prover::core::poly::BitReversedOrder;

use super::logup::LogupAtRow;
use super::{EvalAtRow, INTERACTION_TRACE_IDX};

#[allow(dead_code)]
pub struct AIRCollector {}

/// Dummy evaluator for WebGPU.
pub struct WebDomainEvaluator<'a> {
    pub trace_poly: &'a TreeVec<Vec<&'a CirclePoly<WebBackend>>>,
    pub trace_eval: &'a TreeVec<Vec<&'a CircleEvaluation<WebBackend, BaseField, BitReversedOrder>>>,
    pub needs_to_extend: bool,
    pub col: &'a mut VeryPackedSecureColumnByCoords,
    pub random_coeff_powers: Vec<SecureField>,
    pub eval_domain: CircleDomain,
    pub trace_domain_log_size: u32,
    pub denom_inv: Vec<M31>,
    pub claimed_sum: SecureField,
    pub log_size: u32,
    pub logup: LogupAtRow<Self>,
}

impl<'a> WebDomainEvaluator<'a> {
    pub fn new(
        trace_poly: &'a TreeVec<Vec<&'a CirclePoly<WebBackend>>>,
        trace_eval: &'a TreeVec<Vec<&'a CircleEvaluation<WebBackend, BaseField, BitReversedOrder>>>,
        needs_to_extend: bool,
        col: &'a mut VeryPackedSecureColumnByCoords,
        random_coeff_powers: Vec<SecureField>,
        eval_domain: CircleDomain,
        trace_domain_log_size: u32,
        denom_inv: Vec<M31>,
        log_size: u32,
        claimed_sum: SecureField,
    ) -> Self {
        Self {
            trace_poly,
            trace_eval,
            needs_to_extend,
            col,
            random_coeff_powers,
            eval_domain,
            trace_domain_log_size,
            denom_inv,
            claimed_sum,
            log_size,
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

#[derive(Debug, Clone)]
enum BaseFieldInner {
    One,
    Zero,
    Neg(SymbolicBaseField),
    Cell(usize),
    Inverse(SymbolicBaseField),
    Mul(SymbolicBaseField, SymbolicBaseField),
    Sub(SymbolicBaseField, SymbolicBaseField),
    Add(SymbolicBaseField, SymbolicBaseField),
    Const(M31),
}

#[derive(Debug, Clone)]
enum ExtensionFieldInner {
    One,
    Zero,
    Neg(SymbolicExtensionField),
    Cell(usize),
    Inverse(SymbolicExtensionField),
    Mul(SymbolicExtensionField, SymbolicExtensionField),
    Sub(SymbolicExtensionField, SymbolicExtensionField),
    Add(SymbolicExtensionField, SymbolicExtensionField),
    Embed(SymbolicBaseField),
    Const(QM31),
}

#[derive(Debug, Clone)]
struct SymbolicBaseField(Rc<Box<BaseFieldInner>>);

impl SymbolicBaseField {
    fn embed(self) -> SymbolicExtensionField {
        ExtensionFieldInner::Embed(self).into()
    }

    fn m31(value: M31) -> Self {
        BaseFieldInner::Const(value).into()
    }
}

#[derive(Debug, Clone)]
struct SymbolicExtensionField(Rc<Box<ExtensionFieldInner>>);

impl SymbolicExtensionField {
    fn qm31(value: QM31) -> Self {
        ExtensionFieldInner::Const(value).into()
    }
}

impl std::ops::Neg for SymbolicBaseField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        BaseFieldInner::Neg(self).into()
    }
}

impl From<M31> for SymbolicBaseField {
    fn from(value: M31) -> Self {
        SymbolicBaseField::m31(value)
    }
}

impl Into<SymbolicBaseField> for BaseFieldInner {
    fn into(self) -> SymbolicBaseField {
        SymbolicBaseField(Rc::new(Box::new(self)))
    }
}

impl Into<SymbolicExtensionField> for ExtensionFieldInner {
    fn into(self) -> SymbolicExtensionField {
        SymbolicExtensionField(Rc::new(Box::new(self)))
    }
}

impl std::ops::Mul<SymbolicBaseField> for SymbolicBaseField {
    type Output = Self;

    fn mul(self, rhs: SymbolicBaseField) -> Self::Output {
        BaseFieldInner::Mul(self, rhs).into()
    }
}

impl std::ops::Add<SymbolicBaseField> for SymbolicBaseField {
    type Output = Self;

    fn add(self, rhs: SymbolicBaseField) -> Self::Output {
        BaseFieldInner::Add(self, rhs).into()
    }
}

impl std::ops::Add<QM31> for SymbolicBaseField {
    type Output = SymbolicExtensionField;

    fn add(self, rhs: QM31) -> Self::Output {
        ExtensionFieldInner::Add(self.embed(), ExtensionFieldInner::Const(rhs).into()).into()
    }
}

impl std::ops::Mul<QM31> for SymbolicBaseField {
    type Output = SymbolicExtensionField;

    fn mul(self, rhs: QM31) -> Self::Output {
        ExtensionFieldInner::Mul(self.embed(), ExtensionFieldInner::Const(rhs).into()).into()
    }
}

impl std::ops::Mul<M31> for SymbolicBaseField {
    type Output = Self;

    fn mul(self, rhs: M31) -> Self::Output {
        BaseFieldInner::Mul(self, SymbolicBaseField::m31(rhs)).into()
    }
}

impl std::ops::MulAssign<M31> for SymbolicBaseField {
    fn mul_assign(&mut self, rhs: M31) {
        *self = self.clone() * rhs;
    }
}

impl std::ops::Sub<SymbolicBaseField> for SymbolicBaseField {
    type Output = Self;

    fn sub(self, rhs: SymbolicBaseField) -> Self::Output {
        BaseFieldInner::Sub(self, rhs).into()
    }
}

impl std::ops::MulAssign<SymbolicBaseField> for SymbolicBaseField {
    fn mul_assign(&mut self, rhs: SymbolicBaseField) {
        *self = self.clone() * rhs;
    }
}

impl std::ops::AddAssign<SymbolicBaseField> for SymbolicBaseField {
    fn add_assign(&mut self, rhs: SymbolicBaseField) {
        *self = self.clone() + rhs;
    }
}

impl std::ops::Add<M31> for SymbolicBaseField {
    type Output = Self;

    fn add(self, rhs: M31) -> Self::Output {
        BaseFieldInner::Add(self, SymbolicBaseField::m31(rhs)).into()
    }
}

impl std::ops::Add<M31> for SymbolicExtensionField {
    type Output = Self;

    fn add(self, rhs: M31) -> Self::Output {
        ExtensionFieldInner::Add(self, SymbolicBaseField::m31(rhs).embed()).into()
    }
}

impl std::ops::Add<QM31> for SymbolicExtensionField {
    type Output = Self;

    fn add(self, rhs: QM31) -> Self::Output {
        ExtensionFieldInner::Add(self, ExtensionFieldInner::Const(rhs).into()).into()
    }
}

impl std::ops::Add<SymbolicBaseField> for SymbolicExtensionField {
    type Output = Self;

    fn add(self, rhs: SymbolicBaseField) -> Self::Output {
        ExtensionFieldInner::Add(self, rhs.embed()).into()
    }
}

impl std::ops::Add<SymbolicExtensionField> for SymbolicExtensionField {
    type Output = Self;

    fn add(self, rhs: SymbolicExtensionField) -> Self::Output {
        ExtensionFieldInner::Add(self, rhs).into()
    }
}

impl std::ops::AddAssign<SymbolicExtensionField> for SymbolicExtensionField {
    fn add_assign(&mut self, rhs: SymbolicExtensionField) {
        *self = self.clone() + rhs;
    }
}

impl std::ops::AddAssign<M31> for SymbolicBaseField {
    fn add_assign(&mut self, rhs: M31) {
        *self = self.clone() + rhs;
    }
}

impl std::ops::Mul<M31> for SymbolicExtensionField {
    type Output = Self;

    fn mul(self, rhs: M31) -> Self::Output {
        self * SymbolicBaseField::m31(rhs).embed()
    }
}

impl std::ops::Mul<QM31> for SymbolicExtensionField {
    type Output = Self;

    fn mul(self, rhs: QM31) -> Self::Output {
        self * Self::qm31(rhs)
    }
}

impl std::ops::Mul<SymbolicBaseField> for SymbolicExtensionField {
    type Output = Self;

    fn mul(self, rhs: SymbolicBaseField) -> Self::Output {
        self * rhs.embed()
    }
}

impl std::ops::Sub<QM31> for SymbolicExtensionField {
    type Output = Self;

    fn sub(self, rhs: QM31) -> Self::Output {
        ExtensionFieldInner::Sub(self, Self::qm31(rhs)).into()
    }
}

impl std::ops::Sub<SymbolicExtensionField> for SymbolicExtensionField {
    type Output = Self;

    fn sub(self, rhs: SymbolicExtensionField) -> Self::Output {
        ExtensionFieldInner::Sub(self, rhs).into()
    }
}

impl From<QM31> for SymbolicExtensionField {
    fn from(value: QM31) -> Self {
        Self::qm31(value)
    }
}

impl From<SymbolicBaseField> for SymbolicExtensionField {
    fn from(value: SymbolicBaseField) -> Self {
        value.embed()
    }
}

impl std::ops::Neg for SymbolicExtensionField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        ExtensionFieldInner::Neg(self).into()
    }
}

impl std::ops::Mul<SymbolicExtensionField> for SymbolicExtensionField {
    type Output = Self;

    fn mul(self, rhs: SymbolicExtensionField) -> Self::Output {
        ExtensionFieldInner::Mul(self, rhs).into()
    }
}

impl One for SymbolicExtensionField {
    fn one() -> Self {
        ExtensionFieldInner::One.into()
    }
}

impl One for SymbolicBaseField {
    fn one() -> Self {
        BaseFieldInner::One.into()
    }
}

impl Zero for SymbolicBaseField {
    fn zero() -> Self {
        BaseFieldInner::Zero.into()
    }

    fn is_zero(&self) -> bool {
        unimplemented!()
    }
}

impl Zero for SymbolicExtensionField {
    fn zero() -> Self {
        ExtensionFieldInner::Zero.into()
    }

    fn is_zero(&self) -> bool {
        unimplemented!()
    }
}

impl FieldExpOps for SymbolicBaseField {
    fn inverse(&self) -> Self {
        BaseFieldInner::Inverse(self.clone()).into()
    }
}

// Dummy implementation for WebGPU. These methods will be implemented as WGSL code, so they don't
// need to be implemented here.
#[allow(unused_variables)]
impl EvalAtRow for AIRCollector {
    type F = SymbolicBaseField;
    type EF = SymbolicExtensionField;

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
