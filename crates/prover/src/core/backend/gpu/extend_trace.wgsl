// Note: depends on qm31.wgsl, fraction.wgsl, utils.wgsl
// Define constants
const N_ROWS: u32 = 64;
const N_EXTENDED_ROWS: u32 = N_ROWS * 4;
const N_STATE: u32 = 16;
const N_INSTANCES_PER_ROW: u32 = 8;
const N_COLUMNS: u32 = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
const N_INTERACTION_COLUMNS: u32 = N_INSTANCES_PER_ROW * 4;
const N_HALF_FULL_ROUNDS: u32 = 4;
const FULL_ROUNDS: u32 = 2u * N_HALF_FULL_ROUNDS;
const N_PARTIAL_ROUNDS: u32 = 14;
const N_LANES: u32 = 16;
const N_COLUMNS_PER_REP: u32 = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
const LOG_N_LANES: u32 = 4;
const N_WORKGROUPS: u32 = N_EXTENDED_ROWS * N_LANES / THREADS_PER_WORKGROUP;
const THREADS_PER_WORKGROUP: u32 = 256;
const MAX_ARRAY_LOG_SIZE: u32 = 20;
const MAX_ARRAY_SIZE: u32 = 1u << MAX_ARRAY_LOG_SIZE;
const N_CONSTRAINTS: u32 = 1144;
const R: CM31 = CM31(M31(2u), M31(1u));
const ONE = QM31(CM31(M31(1u), M31(0u)), CM31(M31(0u), M31(0u)));
const DUMMY: u32 = 1004;
const N_ORIGINAL_TRACE_COLUMNS: u32 = 1 + N_COLUMNS + N_INTERACTION_COLUMNS;

fn butterfly(v0: ptr<function, u32>, v1: ptr<function, u32>, twid: u32) {
    let tmp = mod_mul(*v1, twid);
    *v1 = partial_reduce(*v0 + P - tmp);
    *v0 = partial_reduce(*v0 + tmp);
}

struct BaseColumn {
    data: array<array<M31, N_LANES>, N_EXTENDED_ROWS>,
    length: u32,
}

struct OriginalColumn {
    data: array<M31, N_LANES * N_ROWS>,
    length: u32,
}

struct ComputeCompositionPolynomialInput {
    extended_preprocessed_trace: BaseColumn,
    extended_trace: array<BaseColumn, N_COLUMNS>,
    extended_interaction_trace: array<BaseColumn, N_INTERACTION_COLUMNS>,
    denom_inv: array<M31, 4>,
    random_coeff_powers: array<QM31, N_CONSTRAINTS>,
    lookup_elements: LookupElements,
    trace_domain_log_size: u32,
    eval_domain_log_size: u32,
    total_sum: QM31,
}

struct Twiddles {
    circle_twiddles: array<u32, MAX_ARRAY_SIZE>,
    circle_twiddles_size: u32,
    line_twiddles_flat: array<u32, MAX_ARRAY_SIZE>,
    line_twiddles_layer_count: u32,
    line_twiddles_sizes: array<u32, MAX_ARRAY_SIZE>,
    line_twiddles_offsets: array<u32, MAX_ARRAY_SIZE>,
}

struct ExtendTraceInput {
    original_trace: array<OriginalColumn, N_ORIGINAL_TRACE_COLUMNS>,
    twiddles: Twiddles,
}

struct ExtendTraceOutput {
    extended_trace: array<BaseColumn, N_ORIGINAL_TRACE_COLUMNS>,
}

@group(0) @binding(0)
var<storage, read> input: ComputeCompositionPolynomialInput;

@group(0) @binding(1)
var<storage, read_write> output: ComputeCompositionPolynomialOutput;

@group(0) @binding(2)
var<storage, read> trace_input: ExtendTraceInput;

@group(0) @binding(3)
var<storage, read_write> trace_output: ExtendTraceOutput;

@compute @workgroup_size(256)
fn evaluate_line_twiddle(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // dispatch = 1
    let thread_size = 256u;

    let size = 1u << input.log_size;
    let thread_id = global_id.x;
    let chunk_size = (size + thread_size - 1u) / thread_size;
    let chunk_start = thread_id * chunk_size;
    let chunk_end = min(chunk_start + chunk_size, size);

    // copy input.coeffs to output.evals
    for (var i = chunk_start; i < chunk_end; i = i + 1u) {
        output.evals[i] = input.coeffs[i];
    }

    storageBarrier();

    // Process line_twiddles
    var layer = input.line_twiddles_layer_count - 1u;
    loop {
        let layer_size = input.line_twiddles_sizes[layer];
        let layer_offset = input.line_twiddles_offsets[layer];
        let step = 1u << (layer + 1u);
        
        for (var h = 0u; h < layer_size; h = h + 1u) {
            let t = input.line_twiddles_flat[layer_offset + h];
            let idx0_offset = (h << (layer + 2u));

            for (var l = thread_id; l < step; l = l + thread_size) {
                let idx0 = idx0_offset + l;
                let idx1 = idx0 + step;
                
                var val0 = output.evals[idx0];
                var val1 = output.evals[idx1];
                
                butterfly(&val0, &val1, t);
                
                output.evals[idx0] = val0;
                output.evals[idx1] = val1;
            }

            storageBarrier();
        }

        if (layer == 0u) { break; }  
        layer = layer - 1u;
    }
}

@compute @workgroup_size(64)
fn evaluate_circle_twiddle(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let workgroup_dispatch = 256u;
    let workgroup_size = 64u;
    let thread_size = workgroup_dispatch * workgroup_size;
    let size = 1u << (input.log_size - 1u);
    
    let workgroup_id = global_id.y;
    let local_id = global_id.x;

    let workgroup_chunk_size = (size + workgroup_dispatch - 1u) / workgroup_dispatch;
    let thread_chunk_size = (workgroup_chunk_size + workgroup_size - 1u) / workgroup_size;
    let start_idx = workgroup_id * workgroup_chunk_size + local_id * thread_chunk_size;
    let end_idx = min(start_idx + thread_chunk_size, size);

    // store_debug_value(thread_id, global_id.y);
    for (var i = start_idx; i < end_idx; i = i + 1u) {
        let idx0 = i << 1u;
        let idx1 = idx0 + 1u;

        var val0 = output.evals[idx0];
        var val1 = output.evals[idx1];

        butterfly(&val0, &val1, input.circle_twiddles[i]);

        output.evals[idx0] = val0;
        output.evals[idx1] = val1;
    }
}
