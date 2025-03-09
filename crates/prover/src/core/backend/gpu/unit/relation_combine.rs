use std::borrow::Cow;

use crate::core::backend::gpu::gpu_common::{ByteSerialize, GpuComputeInstance, GpuOperation};
use crate::core::backend::gpu::qm31::{GpuM31, GpuQM31};
use crate::examples::poseidon::PoseidonElements;

pub const N_LANES: u32 = 16;
pub const N_STATE: u32 = 16;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuLookupElements {
    pub z: GpuQM31,
    pub alpha: GpuQM31,
    pub alpha_powers: [GpuQM31; N_STATE as usize],
}

impl From<PoseidonElements> for GpuLookupElements {
    fn from(value: PoseidonElements) -> Self {
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

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CombineInput {
    pub values: [GpuM31; N_STATE as usize],
    pub lookup_elements: GpuLookupElements,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SimdCombineInput {
    pub values: [[GpuM31; N_LANES as usize]; N_LANES as usize],
    pub lookup_elements: GpuLookupElements,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CombineOutput {
    pub state: GpuQM31,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SimdCombineOutput {
    pub state: [GpuQM31; N_LANES as usize],
}

impl ByteSerialize for GpuLookupElements {}
impl ByteSerialize for CombineInput {}
impl ByteSerialize for SimdCombineInput {}
impl ByteSerialize for CombineOutput {}
impl ByteSerialize for SimdCombineOutput {}

pub struct CombineOperation;
pub struct SimdCombineOperation;

impl GpuOperation for CombineOperation {
    fn shader_source(&self) -> Cow<'static, str> {
        let common_source = include_str!("../qm31.wgsl");
        let base_source = include_str!("relation_combine.wgsl");

        let inputs = r#"
            struct CombineInput {
                values: array<M31, N_STATE>,
                lookup_elements: LookupElements,
            }

            @group(0) @binding(0) var<storage, read> input: CombineInput;
        "#;

        let output = r#"
            struct CombineOutput {
                state: QM31,
            }

            @group(0) @binding(1) var<storage, read_write> output: CombineOutput;
        "#;

        let operation = r#"
            @compute @workgroup_size(1)
            fn main() {
                output.state = combine(input.values);
            }
        "#;

        format!("{common_source}\n{base_source}\n{inputs}\n{output}\n{operation}").into()
    }
}

impl GpuOperation for SimdCombineOperation {
    fn shader_source(&self) -> Cow<'static, str> {
        let common_source = include_str!("../qm31.wgsl");
        let base_source = include_str!("relation_combine.wgsl");

        let inputs = r#"
            struct SimdCombineInput {
                values: array<array<M31, N_LANES>, N_LANES>,
                lookup_elements: LookupElements,
            }

            @group(0) @binding(0) var<storage, read> input: SimdCombineInput;
        "#;

        let output = r#"
            struct SimdCombineOutput {
                state: array<QM31, N_LANES>,
            }

            @group(0) @binding(1) var<storage, read_write> output: SimdCombineOutput;
        "#;

        let operation = r#"
            @compute @workgroup_size(1)
            fn main() {
                output.state = combine_simd(input.values);
            }
        "#;

        format!("{common_source}\n{base_source}\n{inputs}\n{output}\n{operation}").into()
    }
}

pub async fn compute_combine_operation(
    operation: CombineOperation,
    values: [GpuM31; N_STATE as usize],
    lookup_elements: GpuLookupElements,
) -> CombineOutput {
    let input = CombineInput {
        values,
        lookup_elements,
    };

    let instance = GpuComputeInstance::new(&input, std::mem::size_of::<CombineOutput>()).await;
    let (pipeline, bind_group) =
        instance.create_pipeline(&operation.shader_source(), operation.entry_point());

    let output = instance
        .run_computation::<CombineOutput>(&pipeline, &bind_group, (1, 1, 1))
        .await;

    output
}

pub async fn compute_simd_combine_operation(
    operation: SimdCombineOperation,
    values: [[GpuM31; N_LANES as usize]; N_LANES as usize],
    lookup_elements: GpuLookupElements,
) -> SimdCombineOutput {
    let input = SimdCombineInput {
        values,
        lookup_elements,
    };

    let instance = GpuComputeInstance::new(&input, std::mem::size_of::<SimdCombineOutput>()).await;
    let (pipeline, bind_group) =
        instance.create_pipeline(&operation.shader_source(), operation.entry_point());

    let output = instance
        .run_computation::<SimdCombineOutput>(&pipeline, &bind_group, (1, 1, 1))
        .await;

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint_framework::logup::LookupElements;
    use crate::constraint_framework::Relation;
    use crate::core::backend::simd::m31::PackedM31;
    use crate::core::backend::simd::qm31::PackedSecureField;
    use crate::core::channel::Blake2sChannel;
    use crate::core::fields::m31::BaseField;
    use crate::core::fields::qm31::SecureField;
    use crate::examples::poseidon::PoseidonElements;

    #[test]
    fn test_gpu_combine() {
        let mut channel = Blake2sChannel::default();
        let lookup_elements = LookupElements::<{ N_STATE as usize }>::draw(&mut channel);
        let values: [BaseField; N_STATE as usize] =
            core::array::from_fn(|i| BaseField::from_u32_unchecked(i as u32));

        let gpu_lookup_elements = GpuLookupElements {
            z: lookup_elements.z.into(),
            alpha: lookup_elements.alpha.into(),
            alpha_powers: lookup_elements.alpha_powers.map(|p| p.into()),
        };
        let output = pollster::block_on(compute_combine_operation(
            CombineOperation,
            values.map(|v| v.into()),
            gpu_lookup_elements,
        ));
        println!("output: {:?}", output.state);
        println!(
            "lookup_elements.combine: {:?}",
            lookup_elements.combine::<BaseField, SecureField>(&values)
        );
        assert_eq!(
            lookup_elements.combine::<BaseField, SecureField>(&values),
            output.state.into(),
        );
    }

    #[test]
    fn test_simd_combine() {
        let mut channel = Blake2sChannel::default();
        let poseidon_elements = PoseidonElements::draw(&mut channel);
        let values: [PackedM31; N_STATE as usize] = core::array::from_fn(|i| {
            PackedM31::from_array([BaseField::from_u32_unchecked(i as u32); N_STATE as usize])
        });

        let poseidon_combine: PackedSecureField = poseidon_elements.combine(&values);

        let gpu_values: [[GpuM31; N_LANES as usize]; N_STATE as usize] =
            core::array::from_fn(|i| [GpuM31 { data: i as u32 }; N_LANES as usize]);
        let output = pollster::block_on(compute_simd_combine_operation(
            SimdCombineOperation,
            gpu_values,
            poseidon_elements.into(),
        ));

        let poseidon_combine_array = poseidon_combine.to_array();
        for i in 0..poseidon_combine_array.len() {
            let output_state = output.state[i];
            assert_eq!(poseidon_combine_array[i], output_state.into());
        }
    }

    #[test]
    fn test_gpu_combine_first_poseidon() {
        let mut channel = Blake2sChannel::default();
        let poseidon_elements = PoseidonElements::draw(&mut channel);
        let simd_values: [[GpuM31; N_LANES as usize]; N_STATE as usize] =
            core::array::from_fn(|i| [GpuM31 { data: i as u32 }; N_LANES as usize]);
        let values: [BaseField; N_STATE as usize] =
            core::array::from_fn(|i| BaseField::from_u32_unchecked(i as u32));

        let gpu_lookup_elements = GpuLookupElements {
            z: poseidon_elements.0.z.into(),
            alpha: poseidon_elements.0.alpha.into(),
            alpha_powers: poseidon_elements.0.alpha_powers.map(|p| p.into()),
        };

        let single_output = pollster::block_on(compute_combine_operation(
            CombineOperation,
            values.map(|v| v.into()),
            gpu_lookup_elements,
        ));

        let simd_output = pollster::block_on(compute_simd_combine_operation(
            SimdCombineOperation,
            simd_values,
            poseidon_elements.into(),
        ));

        for i in 0..N_LANES {
            assert_eq!(single_output.state, simd_output.state[i as usize]);
        }
    }
}
