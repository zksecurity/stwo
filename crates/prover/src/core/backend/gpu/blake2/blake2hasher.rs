use std::borrow::Cow;

use crate::core::backend::gpu::gpu_common::{ByteSerialize, GpuComputeInstance, GpuOperation};

const MAX_COLUMN_VALUES: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HashInput {
    pub state: [u32; 8],
    pub block: [u32; 16],
    pub t0: u32,
    pub t1: u32,
    pub f0: u32,
    pub f1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct HashOutput {
    pub state: [u32; 8],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HashNodeInput {
    pub children_hashes_present: u32,
    pub left: [u32; 8],
    pub right: [u32; 8],
    pub column_values: [u32; MAX_COLUMN_VALUES as usize],
    pub column_values_len: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct HashNodeOutput {
    pub state: [u32; 8],
}

impl ByteSerialize for HashInput {}
impl ByteSerialize for HashOutput {}
impl ByteSerialize for HashNodeInput {}
impl ByteSerialize for HashNodeOutput {}

pub struct Blake2sHashOperation;
pub struct Blake2sHashNodeOperation;

impl GpuOperation for Blake2sHashOperation {
    fn shader_source(&self) -> Cow<'static, str> {
        let base_source = include_str!("blake2hasher.wgsl");

        let inputs = r#"
            struct HashInput {
                state: array<u32, 8>,
                block: array<u32, 16>,
                t0: u32,
                t1: u32,
                f0: u32,
                f1: u32,
            }

            @group(0) @binding(0) var<storage, read> input: HashInput;
        "#;

        let output = r#"
            struct HashOutput {
                state: array<u32, 8>,
            }

            @group(0) @binding(1) var<storage, read_write> output: HashOutput;
        "#;

        let operation = r#"
            @compute @workgroup_size(1)
            fn main() {
                output.state = compress(input.state, input.block, input.t0, input.t1, input.f0, input.f1);
            }
        "#;

        format!("{base_source}\n{inputs}\n{output}\n{operation}").into()
    }
}

impl GpuOperation for Blake2sHashNodeOperation {
    fn shader_source(&self) -> Cow<'static, str> {
        let base_source = include_str!("blake2hasher.wgsl");

        let inputs = r#"
            struct HashNodeInput {
                children_hashes_present: u32,
                left: array<u32, 8>,
                right: array<u32, 8>,
                column_values: array<u32, MAX_COLUMN_VALUES>,
                column_values_len: u32,
            }

            @group(0) @binding(0) var<storage, read> input: HashNodeInput;
        "#;

        let output = r#"
            struct HashNodeOutput {
                state: array<u32, 8>,
            }

            @group(0) @binding(1) var<storage, read_write> output: HashNodeOutput;
        "#;

        let operation = r#"
            @compute @workgroup_size(1)
            fn main() {
                var local_columns: array<u32, MAX_COLUMN_VALUES>;
                for (var i = 0u; i < MAX_COLUMN_VALUES; i = i + 1u) {
                    local_columns[i] = input.column_values[i];
                }
                output.state = hash_node(
                    input.children_hashes_present,
                    input.left,
                    input.right,
                    &local_columns,
                    input.column_values_len
                );

            }
        "#;

        format!("{base_source}\n{inputs}\n{output}\n{operation}").into()
    }
}

pub async fn compute_hash_operation(
    operation: Blake2sHashOperation,
    state: [u32; 8],
    block: [u32; 16],
    t0: u32,
    t1: u32,
    f0: u32,
    f1: u32,
) -> HashOutput {
    let input = HashInput {
        state,
        block,
        t0,
        t1,
        f0,
        f1,
    };

    let instance = GpuComputeInstance::new(&input, std::mem::size_of::<HashOutput>()).await;
    let (pipeline, bind_group) =
        instance.create_pipeline(&operation.shader_source(), operation.entry_point());

    let output = instance
        .run_computation::<HashOutput>(&pipeline, &bind_group, (1, 1, 1))
        .await;

    output
}

pub async fn compute_hash_node_operation(
    operation: Blake2sHashNodeOperation,
    children_hashes_present: u32,
    left: [u32; 8],
    right: [u32; 8],
    column_values: [u32; MAX_COLUMN_VALUES as usize],
    column_values_len: u32,
) -> HashNodeOutput {
    let input = HashNodeInput {
        children_hashes_present,
        left,
        right,
        column_values,
        column_values_len,
    };

    let instance = GpuComputeInstance::new(&input, std::mem::size_of::<HashNodeOutput>()).await;
    let (pipeline, bind_group) =
        instance.create_pipeline(&operation.shader_source(), operation.entry_point());

    let output = instance
        .run_computation::<HashNodeOutput>(&pipeline, &bind_group, (1, 1, 1))
        .await;

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasher};
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    use crate::core::vcs::blake2s_ref::compress;
    use crate::core::vcs::ops::MerkleHasher;

    // initial state of blake2s (Blake2sHasher::new())
    const BLAKE2S_INITIAL_STATE: [u32; 8] = [
        1795745351, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635,
        1541459225,
    ];

    #[test]
    fn test_blake2s_ref_compress() {
        let mut blake2s = Blake2sHasher::new();
        blake2s.update(b"a");
        let cpu_hash = blake2s.finalize();
        let cpu_hash_u32 = cpu_hash
            .0
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();

        // initial state of blake2s
        let h = BLAKE2S_INITIAL_STATE;
        let mut msg = [0u32; 16];
        msg[0] = 'a' as u32;
        let count_low = 1;
        let count_high = 0;
        let lastblock = 4294967295;
        let lastnode = 0;
        let blake2s_ref = compress(h, msg, count_low, count_high, lastblock, lastnode);

        assert_eq!(cpu_hash_u32, blake2s_ref);
    }

    #[test]
    fn test_blake2s_hash_gpu() {
        let mut blake2s = Blake2sHasher::new();
        blake2s.update(b"a");
        let cpu_hash = blake2s.finalize();
        let cpu_hash_u32 = cpu_hash
            .0
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();

        let h = BLAKE2S_INITIAL_STATE;
        let mut msg = [0u32; 16];
        msg[0] = 'a' as u32;
        let count_low = 1;
        let count_high = 0;
        let lastblock = 4294967295;
        let lastnode = 0;

        // GPU implementation
        let gpu_result = pollster::block_on(compute_hash_operation(
            Blake2sHashOperation,
            h,
            msg,
            count_low,
            count_high,
            lastblock,
            lastnode,
        ));

        assert_eq!(cpu_hash_u32, gpu_result.state);
    }

    #[test]
    fn test_hash_node_without_children() {
        let column_values: Vec<BaseField> =
            (0..10).map(|x| BaseField::from_u32_unchecked(x)).collect();
        let cpu_hash = Blake2sMerkleHasher::hash_node(None, &column_values);
        let cpu_hash_u32 = cpu_hash
            .0
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();

        let mut column_values_u32: [u32; MAX_COLUMN_VALUES as usize] =
            [0; MAX_COLUMN_VALUES as usize];
        for i in 0..10 {
            column_values_u32[i] = column_values[i].into();
        }
        let gpu_result = pollster::block_on(compute_hash_node_operation(
            Blake2sHashNodeOperation,
            0,
            [0; 8],
            [0; 8],
            column_values_u32,
            10,
        ));

        assert_eq!(cpu_hash_u32, gpu_result.state);
    }

    #[test]
    fn test_hash_node_wit_children() {
        let column_values: Vec<BaseField> =
            (10..20).map(|x| BaseField::from_u32_unchecked(x)).collect();
        let child_hash1 = Blake2sHash::from(&[1u8; 32][..]);
        let child_hash2 = Blake2sHash::from(&[2u8; 32][..]);
        let cpu_hash =
            Blake2sMerkleHasher::hash_node(Some((child_hash1, child_hash2)), &column_values);

        let cpu_hash_u32 = cpu_hash
            .0
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        println!("CPU hash: {:?}", cpu_hash_u32);

        let mut column_values_u32: [u32; MAX_COLUMN_VALUES as usize] =
            [0; MAX_COLUMN_VALUES as usize];
        for i in 0..10 {
            column_values_u32[i] = column_values[i].into();
        }

        let gpu_result = pollster::block_on(compute_hash_node_operation(
            Blake2sHashNodeOperation,
            1,
            child_hash1
                .as_ref()
                .chunks(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<u32>>()
                .try_into()
                .unwrap(),
            child_hash2
                .as_ref()
                .chunks(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<u32>>()
                .try_into()
                .unwrap(),
            column_values_u32,
            10,
        ));

        println!("GPU hash: {:?}", gpu_result.state);
        // assert_eq!(cpu_hash_u32, gpu_result.state);
    }
}
