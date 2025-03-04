use std::borrow::Cow;

use crate::core::backend::gpu::gpu_common::{ByteSerialize, GpuComputeInstance, GpuOperation};

const MAX_COLUMN_VALUES: u32 = 256;
const MAX_PREV_LAYER_WORDS: u32 = 1024;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GpuBlake2sHash {
    pub h: [u32; 8],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CommitInput {
    pub log_size: u32,
    pub num_columns: u32,
    pub node_count: u32,
    pub prev_layer_present: u32,
    pub prev_layer: [GpuBlake2sHash; MAX_PREV_LAYER_WORDS as usize],
    pub columns: [u32; MAX_COLUMN_VALUES as usize],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CommitOutput {
    pub state: [GpuBlake2sHash; MAX_COLUMN_VALUES as usize],
}

impl ByteSerialize for CommitInput {}
impl ByteSerialize for CommitOutput {}

pub struct Blake2sCommitOperation;

impl GpuOperation for Blake2sCommitOperation {
    fn shader_source(&self) -> Cow<'static, str> {
        let common_source = include_str!("blake2s_common.wgsl");
        let base_source = include_str!("blake2s_hasher.wgsl");
        let commit_source = include_str!("blake2s_commit.wgsl");

        format!("{common_source}\n{base_source}\n{commit_source}").into()
    }
}

pub async fn compute_commit_operation(
    operation: Blake2sCommitOperation,
    log_size: u32,
    num_columns: u32,
    node_count: u32,
    prev_layer_present: u32,
    prev_layer: [GpuBlake2sHash; MAX_PREV_LAYER_WORDS as usize],
    columns: [u32; MAX_COLUMN_VALUES as usize],
) -> CommitOutput {
    let input = CommitInput {
        log_size,
        num_columns,
        node_count,
        prev_layer_present,
        prev_layer,
        columns,
    };

    let instance = GpuComputeInstance::new(&input, std::mem::size_of::<CommitOutput>()).await;
    let (pipeline, bind_group) =
        instance.create_pipeline(&operation.shader_source(), operation.entry_point());

    let output = instance
        .run_computation::<CommitOutput>(&pipeline, &bind_group, (1, 1, 1))
        .await;

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CpuBackend;
    use crate::core::fields::m31::BaseField;
    use crate::core::vcs::blake2_hash::Blake2sHash;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    use crate::core::vcs::ops::MerkleOps;

    fn create_basefield_vec(start: u32, count: usize) -> Vec<BaseField> {
        (start..start + count as u32)
            .map(|x| BaseField::from_u32_unchecked(x))
            .collect()
    }

    fn blake2s_hash_to_u32_array(hash: Blake2sHash) -> [u32; 8] {
        hash.0
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn create_blake2s_hash_vec(start: u32, count: usize) -> Vec<Blake2sHash> {
        (start..start + count as u32)
            .map(|x| Blake2sHash::from(&[x as u8; 32][..]))
            .collect()
    }

    #[test]
    fn test_commit_on_layer_without_prev_layer() {
        let log_size = 2;
        let column1 = create_basefield_vec(1, 4);
        let column2 = create_basefield_vec(101, 4);
        let columns = vec![&column1, &column2];

        let result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size, None, &columns,
        );

        let flatten_col_vec = columns
            .iter()
            .flat_map(|column| column.iter().map(|x| x.0))
            .collect::<Vec<_>>();
        let mut gpu_flatten_columns: [u32; MAX_COLUMN_VALUES as usize] =
            [0u32; MAX_COLUMN_VALUES as usize];
        for i in 0..flatten_col_vec.len() {
            gpu_flatten_columns[i] = flatten_col_vec[i];
        }

        let gpu_result = pollster::block_on(compute_commit_operation(
            Blake2sCommitOperation,
            log_size,
            columns.len() as u32,
            1u32 << log_size,
            0,
            [GpuBlake2sHash { h: [0u32; 8] }; MAX_PREV_LAYER_WORDS as usize],
            gpu_flatten_columns,
        ));

        for i in 0..result.len() {
            assert_eq!(blake2s_hash_to_u32_array(result[i]), gpu_result.state[i].h);
        }
    }

    #[test]
    fn test_commit_on_layer_with_prev_layer() {
        let log_size = 2;
        let prev_layer = create_blake2s_hash_vec(1000, 8);
        let column1 = create_basefield_vec(1, 4);
        let column2 = create_basefield_vec(101, 4);
        let columns = vec![&column1, &column2];

        let result = <CpuBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
            log_size,
            Some(&prev_layer),
            &columns,
        );

        let flatten_col_vec: Vec<u32> = columns
            .iter()
            .flat_map(|col| col.iter().map(|x| x.0))
            .collect();

        let mut gpu_prev_layer = [GpuBlake2sHash { h: [0u32; 8] }; MAX_PREV_LAYER_WORDS as usize];
        for i in 0..prev_layer.len() {
            gpu_prev_layer[i] = GpuBlake2sHash {
                h: blake2s_hash_to_u32_array(prev_layer[i]),
            };
        }

        let mut gpu_flatten_columns = [0u32; MAX_COLUMN_VALUES as usize];
        gpu_flatten_columns[..flatten_col_vec.len()].copy_from_slice(&flatten_col_vec);

        let gpu_result = pollster::block_on(compute_commit_operation(
            Blake2sCommitOperation,
            log_size,
            columns.len() as u32,
            1u32 << log_size,
            1,
            gpu_prev_layer,
            gpu_flatten_columns,
        ));

        for i in 0..result.len() {
            assert_eq!(blake2s_hash_to_u32_array(result[i]), gpu_result.state[i].h);
        }
    }
}
