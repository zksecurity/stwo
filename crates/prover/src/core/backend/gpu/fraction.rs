use std::borrow::Cow;

use super::compute_composition_polynomial::GpuQM31;
use super::gpu_common::{ByteSerialize, GpuComputeInstance, GpuOperation};
use crate::core::fields::qm31::QM31;
use crate::core::lookups::utils::Fraction;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GpuFraction {
    pub numerator: GpuQM31,
    pub denominator: GpuQM31,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ComputeInput {
    pub first: GpuFraction,
    pub second: GpuFraction,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ComputeOutput {
    pub result: GpuFraction,
}

impl ByteSerialize for ComputeInput {}
impl ByteSerialize for ComputeOutput {}

impl From<Fraction<QM31, QM31>> for GpuFraction {
    fn from(value: Fraction<QM31, QM31>) -> Self {
        GpuFraction {
            numerator: GpuQM31::from(value.numerator),
            denominator: GpuQM31::from(value.denominator),
        }
    }
}

pub enum FractionOperation {
    Add,
}

impl GpuOperation for FractionOperation {
    fn shader_source(&self) -> Cow<'static, str> {
        let base_source = include_str!("fraction.wgsl");
        let qm31_source = include_str!("qm31.wgsl");

        let inputs = r#"
            struct ComputeInput {
                first: Fraction,
                second: Fraction,
            }

            @group(0) @binding(0) var<storage, read> input: ComputeInput;
        "#;

        let output = r#"
            struct ComputeOutput {
                result: Fraction,
            }

            @group(0) @binding(1) var<storage, read_write> output: ComputeOutput;
        "#;

        let operation = match self {
            FractionOperation::Add => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = fraction_add(input.first, input.second);
                }
            "#
            }
        };

        format!("{qm31_source}\n{base_source}\n{inputs}\n{output}\n{operation}").into()
    }
}

pub async fn compute_fraction_operation(
    operation: FractionOperation,
    first: Fraction<QM31, QM31>,
    second: Fraction<QM31, QM31>,
) -> ComputeOutput {
    let input = ComputeInput {
        first: first.into(),
        second: second.into(),
    };

    let instance = GpuComputeInstance::new(&input, std::mem::size_of::<ComputeOutput>()).await;
    let (pipeline, bind_group) =
        instance.create_pipeline(&operation.shader_source(), operation.entry_point());

    let output = instance
        .run_computation::<ComputeOutput>(&pipeline, &bind_group, (1, 1, 1))
        .await;

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::qm31::QM31;

    #[test]
    fn test_fraction_add() {
        // CPU implementation
        let cpu_a = Fraction::new(
            QM31::from_u32_unchecked(1u32, 0u32, 0u32, 0u32),
            QM31::from_u32_unchecked(3u32, 0u32, 0u32, 0u32),
        );
        let cpu_b = Fraction::new(
            QM31::from_u32_unchecked(2u32, 0u32, 0u32, 0u32),
            QM31::from_u32_unchecked(6u32, 0u32, 0u32, 0u32),
        );
        let cpu_result = cpu_a + cpu_b;

        // GPU implementation
        let gpu_result = pollster::block_on(compute_fraction_operation(
            FractionOperation::Add,
            cpu_a,
            cpu_b,
        ));

        assert_eq!(cpu_result.numerator, gpu_result.result.numerator.into());
        assert_eq!(cpu_result.denominator, gpu_result.result.denominator.into());
    }
}
