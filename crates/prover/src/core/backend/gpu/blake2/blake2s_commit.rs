use std::borrow::Cow;

use crate::core::backend::gpu::gpu_common::{ByteSerialize, GpuComputeInstance, GpuOperation};

const MAX_COLUMN_LENGTH: u32 = 256;
const MAX_COLUMNS: u32 = 256;
const MAX_TOTAL_COLUMNS: u32 = 256 * 4;
const MAX_PREV_LAYER_WORDS: u32 = 1024;
const MAX_FLAT_SIZE: u32 = 65535;

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
    pub total_columns: u32,
    pub columns: [GpuColumn; MAX_TOTAL_COLUMNS as usize],
    pub columns_len: [u32; MAX_TOTAL_COLUMNS as usize],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InputData {
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
    pub flat_layers: [GpuBlake2sHash; MAX_FLAT_SIZE as usize],
    pub input_data: InputData,
    pub out_layer: [GpuBlake2sHash; MAX_PREV_LAYER_WORDS as usize],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct DebugOutput {
    pub debugs: [u32; 1024],
    pub count: u32,
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
impl ByteSerialize for InputData {}
impl ByteSerialize for DebugOutput {}

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
    total_columns: u32,
    columns: [GpuColumn; MAX_TOTAL_COLUMNS as usize],
    columns_len: [u32; MAX_TOTAL_COLUMNS as usize],
) -> CommitOutput {
    let input = CommitInput {
        total_columns,
        columns,
        columns_len,
    };

    let instance = GpuComputeInstance::new(&input, std::mem::size_of::<CommitOutput>()).await;
    let (pipeline, bind_group) =
        instance.create_pipeline_debug(&operation.shader_source(), operation.entry_point());

    let (output, _debug_output) = instance
        .run_computation_debug::<CommitOutput, DebugOutput>(&pipeline, &bind_group, (1, 1, 1))
        .await;

    output
}

#[cfg(test)]
mod tests {
    use std::cmp::Reverse;

    use itertools::Itertools;

    use super::*;
    use crate::core::backend::gpu::blake2::blake2s_common::blake2s_hash_to_u32_array;
    use crate::core::backend::{Col, CpuBackend};
    use crate::core::fields::m31::BaseField;
    use crate::core::utils::PeekableExt;
    use crate::core::vcs::blake2_hash::Blake2sHash;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleHasher;
    use crate::core::vcs::ops::MerkleOps;

    fn create_basefield_vec(start: u32, count: usize) -> Vec<BaseField> {
        (start..start + count as u32)
            .map(|x| BaseField::from_u32_unchecked(x))
            .collect()
    }

    fn commit_reference_impl(
        columns: Vec<&Col<CpuBackend, BaseField>>,
    ) -> Vec<Col<CpuBackend, Blake2sHash>> {
        if columns.is_empty() {
            return vec![];
        }

        let columns = &mut columns
            .into_iter()
            .sorted_by_key(|c| Reverse(c.len()))
            .peekable();
        let mut layers: Vec<Col<CpuBackend, Blake2sHash>> = Vec::new();

        let max_log_size = columns.peek().unwrap().len().ilog2();
        for log_size in (0..=max_log_size).rev() {
            // Take columns of the current log_size.
            let layer_columns = columns
                .peek_take_while(|column| column.len().ilog2() == log_size)
                .collect_vec();

            let layer: Col<CpuBackend, Blake2sHash> = <CpuBackend as MerkleOps<
                Blake2sMerkleHasher,
            >>::commit_on_layer(
                log_size, layers.last(), &layer_columns
            );
            layers.push(layer);
        }
        layers.reverse();
        layers
    }

    #[test]
    fn test_blake2s_commit() {
        // make 2 columns of 4 elements each
        let columns = vec![create_basefield_vec(1, 4), create_basefield_vec(101, 4)];

        let reference_layers = commit_reference_impl(columns.iter().map(|c| c.as_ref()).collect());

        let mut gpu_columns: [GpuColumn; MAX_TOTAL_COLUMNS as usize] =
            [GpuColumn::default(); MAX_TOTAL_COLUMNS as usize];
        for (i, c) in columns.iter().enumerate() {
            for (j, x) in c.iter().enumerate() {
                gpu_columns[i].column[j] = x.0;
            }
        }

        let mut gpu_columns_len: [u32; MAX_TOTAL_COLUMNS as usize] =
            [0; MAX_TOTAL_COLUMNS as usize];
        for (i, c) in columns.iter().enumerate() {
            gpu_columns_len[i] = c.len() as u32;
        }

        let gpu_result = pollster::block_on(compute_commit_operation(
            Blake2sCommitOperation,
            columns.len() as u32,
            gpu_columns,
            gpu_columns_len,
        ));

        // compare reference_layers and gpu_result.flat_layers
        let mut flattend_index = 0;
        for layer in &reference_layers {
            for x in layer {
                assert_eq!(
                    gpu_result.flat_layers[flattend_index].h,
                    blake2s_hash_to_u32_array(*x)
                );
                flattend_index += 1;
            }
        }
    }
}
