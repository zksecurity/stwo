use super::qm31::GpuM31;
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::m31::PackedM31;
use crate::core::fields::m31::M31;
use crate::examples::poseidon::LookupData;

pub const N_ROWS: u32 = 256;
pub const N_LANES: u32 = 16;
pub const N_STATE: u32 = 16;
pub const N_INSTANCES_PER_ROW: u32 = 1 << N_LOG_INSTANCES_PER_ROW;
pub const N_LOG_INSTANCES_PER_ROW: u32 = 3;

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

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuBaseColumn {
    data: [[GpuM31; N_LANES as usize]; N_ROWS as usize],
    length: u32,
}

impl ByteSerialize for GpuBaseColumn {}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Ids {
    workgroup_id_x: u32,
    workgroup_id_y: u32,
    workgroup_id_z: u32,
    local_invocation_id_x: u32,
    local_invocation_id_y: u32,
    local_invocation_id_z: u32,
    global_invocation_id_x: u32,
    global_invocation_id_y: u32,
    global_invocation_id_z: u32,
    local_invocation_index: u32,
    num_workgroups_x: u32,
    num_workgroups_y: u32,
    num_workgroups_z: u32,
    workgroup_index: u32,
    global_invocation_index: u32,
}

#[allow(dead_code)]
impl BaseColumn {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= std::mem::size_of::<Self>());
        let slice = unsafe { &*(bytes.as_ptr() as *const GpuBaseColumn) };
        (*slice).into()
    }
}

impl From<GpuBaseColumn> for BaseColumn {
    fn from(value: GpuBaseColumn) -> Self {
        BaseColumn {
            data: value
                .data
                .iter()
                .map(|f| {
                    let mut array: [M31; N_LANES as usize] = [M31(0); N_LANES as usize];
                    for (i, v) in f.iter().enumerate() {
                        array[i] = M31(v.data);
                    }
                    PackedM31::from_array(array)
                })
                .collect(),
            length: value.length as usize,
        }
    }
}

#[allow(dead_code)]
impl LookupData {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let base_column_size = std::mem::size_of::<GpuBaseColumn>();
        let base_column_vec_size = base_column_size * N_STATE as usize;
        let state_size = base_column_vec_size * N_INSTANCES_PER_ROW as usize;
        let lookup_data_size = state_size * 2;
        assert!(bytes.len() >= lookup_data_size);
        let initial_state_slice: [[BaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize] =
            bytes
                .chunks(base_column_vec_size)
                .take(N_INSTANCES_PER_ROW as usize)
                .map(|chunk| {
                    chunk
                        .chunks(base_column_size)
                        .take(N_STATE as usize)
                        .map(|chunk| BaseColumn::from_bytes(chunk))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
        let final_state_slice: [[BaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize] =
            bytes[state_size..]
                .chunks(base_column_vec_size)
                .take(N_INSTANCES_PER_ROW as usize)
                .map(|chunk| {
                    chunk
                        .chunks(base_column_size)
                        .take(N_STATE as usize)
                        .map(|chunk| BaseColumn::from_bytes(chunk))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
        Self {
            initial_state: initial_state_slice,
            final_state: final_state_slice,
        }
    }
}
