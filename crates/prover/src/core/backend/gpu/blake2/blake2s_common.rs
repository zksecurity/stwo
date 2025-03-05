use crate::core::vcs::blake2_hash::Blake2sHash;

pub fn blake2s_hash_to_u32_array(hash: Blake2sHash) -> [u32; 8] {
    hash.0
        .chunks(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
