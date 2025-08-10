/// Configuration trait for constraint system constants
pub trait ConstraintConfig {
    const N_LOG_INSTANCES: u32;
    const N_CONSTRAINTS: u32;
    const N_COLUMNS: u32;
    const N_INTERMEDIATES: u32;
    const N_EXT_INTERMEDIATES: u32;
    const N_LOOKUP_ELEMENTS: u32;
    
    // Derived constants
    const N_ROWS: u32 = 1 << Self::N_LOG_INSTANCES;
    const N_LANES: u32 = 16;
    const N_EXTENDED_ROWS: u32 = Self::N_ROWS * 2;
    const N_ORIGINAL_ROWS: u32 = Self::N_ROWS;
    const N_PACKED_ROWS: u32 = Self::N_EXTENDED_ROWS / Self::N_LANES;
}

/// Const generic implementation
pub struct Config<
    const N_LOG_INSTANCES: u32,
    const N_CONSTRAINTS: u32, 
    const N_COLUMNS: u32,
    const N_INTERMEDIATES: u32,
    const N_EXT_INTERMEDIATES: u32,
    const N_LOOKUP_ELEMENTS: u32,
>;

impl<
    const N_LOG_INSTANCES: u32,
    const N_CONSTRAINTS: u32,
    const N_COLUMNS: u32,
    const N_INTERMEDIATES: u32,
    const N_EXT_INTERMEDIATES: u32,
    const N_LOOKUP_ELEMENTS: u32,
> ConstraintConfig for Config<N_LOG_INSTANCES, N_CONSTRAINTS, N_COLUMNS, N_INTERMEDIATES, N_EXT_INTERMEDIATES, N_LOOKUP_ELEMENTS> {
    const N_LOG_INSTANCES: u32 = N_LOG_INSTANCES;
    const N_CONSTRAINTS: u32 = N_CONSTRAINTS;
    const N_COLUMNS: u32 = N_COLUMNS;
    const N_INTERMEDIATES: u32 = N_INTERMEDIATES;
    const N_EXT_INTERMEDIATES: u32 = N_EXT_INTERMEDIATES;
    const N_LOOKUP_ELEMENTS: u32 = N_LOOKUP_ELEMENTS;
}

/// Default configuration type alias
pub type DefaultConfig = Config<5, 1, 3, 1, 1, 3>;

// For backward compatibility, re-export default values as constants
pub const N_LOG_INSTANCES: u32 = DefaultConfig::N_LOG_INSTANCES;
pub const N_ROWS: u32 = DefaultConfig::N_ROWS;
pub const N_CONSTRAINTS: u32 = DefaultConfig::N_CONSTRAINTS;
pub const N_LANES: u32 = DefaultConfig::N_LANES;
pub const N_EXTENDED_ROWS: u32 = DefaultConfig::N_EXTENDED_ROWS;
pub const N_ORIGINAL_ROWS: u32 = DefaultConfig::N_ORIGINAL_ROWS;
pub const N_PACKED_ROWS: u32 = DefaultConfig::N_PACKED_ROWS;
pub const N_COLUMNS: u32 = DefaultConfig::N_COLUMNS;
pub const N_INTERMEDIATES: u32 = DefaultConfig::N_INTERMEDIATES;
pub const N_EXT_INTERMEDIATES: u32 = DefaultConfig::N_EXT_INTERMEDIATES;
pub const N_LOOKUP_ELEMENTS: u32 = DefaultConfig::N_LOOKUP_ELEMENTS;