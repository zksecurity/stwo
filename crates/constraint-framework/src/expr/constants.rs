pub const N_LOG_INSTANCES: u32 = 5;
pub const N_ROWS: u32 = 1 << (N_LOG_INSTANCES + 1); // 2^5 = 32
// manual
pub const N_CONSTRAINTS: u32 = 1;

pub const N_LANES: u32 = 16;
pub const N_EXTENDED_ROWS: u32 = N_ROWS * 4; // 32 * 4 = 128
pub const N_ORIGINAL_ROWS: u32 = N_ROWS;
pub const N_PACKED_ROWS: u32 = N_EXTENDED_ROWS / N_LANES; // 128 / 16 = 8
// manual
pub const N_COLUMNS: u32 = 3; // We need 3 columns for x0, x1, x2
