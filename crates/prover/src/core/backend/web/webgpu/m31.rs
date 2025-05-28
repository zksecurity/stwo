use super::gpu_common::ByteSerialize;
use crate::core::fields::m31::M31;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(4))]
pub struct GpuM31(pub u32); // alias M31 = u32

impl From<M31> for GpuM31 {
    fn from(value: M31) -> Self {
        GpuM31 { 0: value.into() }
    }
}

impl From<GpuM31> for M31 {
    fn from(value: GpuM31) -> Self {
        M31::from(value.0)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ComputeM31Input {
    pub first: GpuM31,
    pub second: GpuM31,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ComputeM31Output {
    pub result: GpuM31,
}

impl ByteSerialize for ComputeM31Input {}
impl ByteSerialize for ComputeM31Output {}

pub enum M31Operation {
    Add,
    Subtract,
    Multiply,
    Negate,
    Inverse,
    Square,
    Pow3,
    Pow5,
    Pow8,
    Pow128,
    Pow256,
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;
    use crate::core::backend::web::webgpu::gpu_common::{GpuComputeInstance, GpuOperation};

    use crate::core::fields::m31::{M31, P};
    use crate::core::fields::FieldExpOps;

    impl GpuOperation for M31Operation {
        fn shader_source(&self) -> Cow<'static, str> {
            let base_source = include_str!("qm31.wgsl");

            let inputs = r#"
            struct ComputeM31Input {
                first: M31,
                second: M31,
            }

            @group(0) @binding(0) var<storage, read> input: ComputeM31Input;
        "#;

            let output = r#"
            struct ComputeM31Output {
                result: M31,
            }

            @group(0) @binding(1) var<storage, read_write> output: ComputeM31Output;
        "#;

            let operation = match self {
                M31Operation::Add => {
                    r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = m31_add(input.first, input.second);
                }
            "#
                }
                M31Operation::Multiply => {
                    r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = m31_mul(input.first, input.second);
                }
            "#
                }
                M31Operation::Subtract => {
                    r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = m31_sub(input.first, input.second);
                }
            "#
                }
                M31Operation::Negate => {
                    r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = m31_neg(input.first);
                }
            "#
                }
                M31Operation::Inverse => {
                    r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = m31_inverse(input.first);
                }
            "#
                }
                M31Operation::Square => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = m31_square(input.first);
                }
            "#
                }
                M31Operation::Pow3 => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = m31_pow3(input.first);
                }
            "#
                }
                M31Operation::Pow5 => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = m31_pow5(input.first);
                }
            "#
                }
                M31Operation::Pow8 => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = m31_pow8(input.first);
                }
            "#
                }
                M31Operation::Pow128 => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = m31_pow128(input.first);
                }
            "#
                }
                M31Operation::Pow256 => {
                r#"
                @compute @workgroup_size(1)
                fn main() {
                    output.result = m31_pow256(input.first);
                }
            "#
                }
        };

            format!("{base_source}\n{inputs}\n{output}\n{operation}").into()
        }
    }

    #[allow(dead_code)]
    pub async fn compute_field_operation(
        operation: M31Operation,
        first: M31,
        second: M31,
    ) -> M31 {
        let input = ComputeM31Input {
            first: first.into(),
            second: second.into(),
        };

        let instance = GpuComputeInstance::new(&input, std::mem::size_of::<ComputeM31Output>()).await;
        let (pipeline, bind_group) =
            instance.create_pipeline(&operation.shader_source(), operation.entry_point());

        let output = instance
            .run_computation::<ComputeM31Output>(&pipeline, &bind_group, (1, 1, 1))
            .await;

        output.result.into()
    }

    #[test]
    fn test_gpu_field_values() {
        let m0 = M31::from(2u32);
        let m1 = M31::from(3u32);

        // Test round-trip conversion CPU -> GPU -> CPU
        let gpu_m0 = GpuM31::from(m0);
        let gpu_m1 = GpuM31::from(m1);

        let cpu_m0 = M31::from(gpu_m0);
        let cpu_m1 = M31::from(gpu_m1);

        assert_eq!(
            m0, cpu_m0,
            "Round-trip conversion should preserve values for m0"
        );
        assert_eq!(
            m1, cpu_m1,
            "Round-trip conversion should preserve values for m1"
        );
    }

    #[test]
    fn test_gpu_m31_field_arithmetic() {
        // Test M31 field operations
        let m = M31::from(19u32);
        let one = M31::from(1u32);

        // Test addition
        let cpu_add = m + one;
        let gpu_add = pollster::block_on(compute_field_operation(M31Operation::Add, m, one));
        assert_eq!(gpu_add, cpu_add, "M31 addition failed");

        // Test subtraction
        let cpu_sub = m - one;
        let gpu_sub = pollster::block_on(compute_field_operation(
            M31Operation::Subtract,
            m,
            one,
        ));
        assert_eq!(gpu_sub, cpu_sub, "M31 subtraction failed");

        // Test multiplication
        let cpu_mul = m * one;
        let gpu_mul = pollster::block_on(compute_field_operation(
            M31Operation::Multiply,
            m,
            one,
        ));
        assert_eq!(gpu_mul, cpu_mul, "M31 multiplication failed");

        // Test negation
        let cpu_neg = -m;
        let gpu_neg = pollster::block_on(compute_field_operation(
            M31Operation::Negate,
            m,
            one,
        ));
        assert_eq!(gpu_neg, cpu_neg, "M31 negation failed");

        // Test inverse
        let cpu_inv = m.inverse();
        let gpu_inv = pollster::block_on(compute_field_operation(
            M31Operation::Inverse,
            m,
            one,
        ));
        assert_eq!(gpu_inv, cpu_inv, "M31 inverse failed");

        // Test square 
        let cpu_square = m.square();
        let gpu_square = pollster::block_on(compute_field_operation(
            M31Operation::Square,
            m,
            one,
        ));
        assert_eq!(
            gpu_square, cpu_square,
            "M31 square operation failed"
        );

        // Test pow5
        let cpu_pow5 = m.square().square() * m;
        let gpu_pow5 = pollster::block_on(compute_field_operation(
            M31Operation::Pow5,
            m,
            one,
        ));
        assert_eq!(
            gpu_pow5, cpu_pow5,
            "M31 pow5 operation failed"
        );

        // Test with large numbers (near P)
        let large_m = M31::from(P - 1);

        // Test large number multiplication
        let cpu_large_mul = large_m * m;
        let gpu_large_mul = pollster::block_on(compute_field_operation(
            M31Operation::Multiply,
            large_m,
            m,
        ));
        assert_eq!(
            gpu_large_mul, cpu_large_mul,
            "M31 large number multiplication failed"
        );

        // Test large number inverse
        let cpu_large_inv = one / large_m;
        let gpu_large_inv = pollster::block_on(compute_field_operation(
            M31Operation::Inverse,
            large_m,
            m,
        ));
        assert_eq!(
            gpu_large_inv, cpu_large_inv,
            "M31 large number inverse failed"
        );
    }
}
