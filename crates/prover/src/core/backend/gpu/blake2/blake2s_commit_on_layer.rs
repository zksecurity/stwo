use std::borrow::Cow;

use crate::core::backend::gpu::gpu_common::{ByteSerialize, GpuComputeInstance, GpuOperation};

const MAX_COLUMN_LENGTH: u32 = 256;
const MAX_COLUMNS: u32 = 256;
const MAX_PREV_LAYER_WORDS: u32 = 1024;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GpuBlake2sHash {
    pub h: [u32; 8],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GpuColumn {
    pub column: [u32; MAX_COLUMN_LENGTH as usize],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CommitInput {
    pub log_size: u32,
    pub num_columns: u32,
    pub node_count: u32,
    pub prev_layer_present: u32,
    pub prev_layer: [GpuBlake2sHash; MAX_PREV_LAYER_WORDS as usize],
    pub columns: [GpuColumn; MAX_COLUMNS as usize],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CommitOutput {
    pub state: [GpuBlake2sHash; MAX_COLUMNS as usize],
}

impl Default for GpuBlake2sHash {
    fn default() -> Self {
        Self { h: [0u32; 8] }
    }
}

impl Default for GpuColumn {
    fn default() -> Self {
        Self {
            column: [0u32; MAX_COLUMN_LENGTH as usize],
        }
    }
}

impl ByteSerialize for GpuBlake2sHash {}
impl ByteSerialize for GpuColumn {}
impl ByteSerialize for CommitInput {}
impl ByteSerialize for CommitOutput {}

pub struct Blake2sCommitOperation;

impl GpuOperation for Blake2sCommitOperation {
    fn shader_source(&self) -> Cow<'static, str> {
        let common_source = include_str!("blake2s_common.wgsl");
        let base_source = include_str!("blake2s_hasher.wgsl");
        let commit_source = include_str!("blake2s_commit_on_layer.wgsl");

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
    columns: [GpuColumn; MAX_COLUMNS as usize],
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

        let mut gpu_columns = [GpuColumn::default(); MAX_COLUMNS as usize];
        for i in 0..columns.len() {
            let mut col_vec: Vec<_> = columns[i].iter().map(|x| x.0).collect();
            let required_size = gpu_columns[i].column.len();
            col_vec.resize(required_size, Default::default());
            gpu_columns[i] = GpuColumn {
                column: col_vec.try_into().unwrap(),
            };
        }

        let gpu_result = pollster::block_on(compute_commit_operation(
            Blake2sCommitOperation,
            log_size,
            columns.len() as u32,
            1u32 << log_size,
            0,
            [GpuBlake2sHash::default(); MAX_PREV_LAYER_WORDS as usize],
            gpu_columns,
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

        let mut gpu_columns = [GpuColumn::default(); MAX_COLUMNS as usize];
        for i in 0..columns.len() {
            let mut col_vec: Vec<_> = columns[i].iter().map(|x| x.0).collect();
            let required_size = gpu_columns[i].column.len();
            col_vec.resize(required_size, Default::default());
            gpu_columns[i] = GpuColumn {
                column: col_vec.try_into().unwrap(),
            };
        }

        let mut gpu_prev_layer = [GpuBlake2sHash::default(); MAX_PREV_LAYER_WORDS as usize];
        for i in 0..prev_layer.len() {
            gpu_prev_layer[i] = GpuBlake2sHash {
                h: blake2s_hash_to_u32_array(prev_layer[i]),
            };
        }

        let gpu_result = pollster::block_on(compute_commit_operation(
            Blake2sCommitOperation,
            log_size,
            columns.len() as u32,
            1u32 << log_size,
            1,
            gpu_prev_layer,
            gpu_columns,
        ));

        for i in 0..result.len() {
            assert_eq!(blake2s_hash_to_u32_array(result[i]), gpu_result.state[i].h);
        }
    }
}
