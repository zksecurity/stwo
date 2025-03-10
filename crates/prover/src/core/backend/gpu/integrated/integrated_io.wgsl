
struct BaseColumn {
    data: array<array<M31, N_LANES>, N_ROWS>,
    length: u32,
}

struct Twiddles {
    circle_twiddles: array<M31, N_CIRCLE_TWIDDLES_SIZE>,
    circle_twiddles_size: u32,
    line_twiddles_flat: array<M31, N_LINE_TWIDDLES_FLAT_SIZE>,
    line_twiddles_layer_count: u32,
    line_twiddles_sizes: array<u32, N_LINE_TWIDDLES_SIZE>,
    line_twiddles_offsets: array<u32, N_LINE_TWIDDLES_SIZE>,
    mod_inv: M31,
}

struct IntegratedInput {
    log_size: u32,
    twiddles: Twiddles,
    lookup_elements: LookupElements,
    denom_inv: array<M31, 4>,
    random_coeff_powers: array<QM31, N_CONSTRAINTS>,
    trace_domain_log_size: u32,
    eval_domain_log_size: u32,
}

struct LookupData {
    initial_state: array<array<BaseColumn, N_STATE>, N_INSTANCES_PER_ROW>,
    final_state: array<array<BaseColumn, N_STATE>, N_INSTANCES_PER_ROW>,
}

struct LookupElements {
    z: QM31,
    alpha: QM31,
    alpha_powers: array<QM31, N_STATE>,
}

struct QM31Column {
    data: array<QM31, N_ORIGINAL_COLUMN_SIZE>,
    length: u32
}

struct OriginalColumn {
    data: array<M31, N_ORIGINAL_COLUMN_SIZE>,
}

struct Extended1DColumn {
    data: array<M31, N_EXTENDED_COLUMN_SIZE>,
}

struct GenTraceOutput {
    original_trace: array<OriginalColumn, N_ORIGINAL_TRACE_COLUMNS>,
    trace: array<BaseColumn, N_COLUMNS>,
    lookup_data: LookupData,
}

struct GenInteractionTraceOutput {
    // chunk 
    interaction_trace_qm31: array<QM31Column, N_INSTANCES_PER_ROW>,
    interaction_trace_buffers: array<QM31Column, 4>,
    total_sum: QM31,
    // chunk ends
}

struct Results {
    values: array<M31, N_FLAT_MAX_ARRAY_SIZE>,
}

struct ComputeCompositionPolynomialOutput {
    poly: array<array<QM31, N_LANES>, N_EXTENDED_ROWS>,
    extended_trace: array<Extended1DColumn, N_ORIGINAL_TRACE_COLUMNS>,
}

@group(0) @binding(0)
var<storage, read> input: IntegratedInput;

// Intermediate buffer
@group(0) @binding(1)
var<storage, read_write> gen_trace_output: GenTraceOutput;

@group(0) @binding(2)
var<storage, read_write> interpolate_output: Results;

@group(0) @binding(3)
var<storage, read_write> gen_interaction_trace_output: GenInteractionTraceOutput;
