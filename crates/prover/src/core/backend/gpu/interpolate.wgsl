// Note: depends on gen_trace_interpolate_columns_constants.wgsl

const MODULUS_BITS: u32 = 31u;
const HALF_BITS: u32 = 16u;
// Mersenne prime P = 2^31 - 1
const P: u32 = 2147483647u;

fn partial_reduce(val: u32) -> u32 {
    let reduced = val - P;
    return select(val, reduced, reduced < val);
}

fn mod_mul(a: u32, b: u32) -> u32 {
    // Split into 16-bit parts
    let a1 = a >> HALF_BITS;
    let a0 = a & 0xFFFFu;
    let b1 = b >> HALF_BITS;
    let b0 = b & 0xFFFFu;
    
    // Compute partial products
    let m0 = partial_reduce(a0 * b0);
    let m1 = partial_reduce(a0 * b1);
    let m2 = partial_reduce(a1 * b0);
    let m3 = partial_reduce(a1 * b1);
    
    // Combine middle terms with reduction
    let mid = partial_reduce(m1 + m2);
    
    // Combine parts with partial reduction
    let shifted_mid = partial_reduce(mid << HALF_BITS);
    let low = partial_reduce(m0 + shifted_mid);
    
    let high_part = partial_reduce(m3 + (mid >> HALF_BITS));
    
    // Final combination using Mersenne prime property
    let result = partial_reduce(
        partial_reduce((high_part << 1u)) + 
        partial_reduce((low >> MODULUS_BITS)) + 
        partial_reduce(low & P)
    );
    
    return result;
}

fn ibutterfly(v0: ptr<function, u32>, v1: ptr<function, u32>, itwid: u32) {
    let tmp = *v0;
    *v0 = partial_reduce(tmp + *v1);
    *v1 = mod_mul(partial_reduce(tmp + P - *v1), itwid);
}

struct GenTraceInput {
    log_size: u32,
    circle_twiddles: array<u32, N_CIRCLE_TWIDDLES_SIZE>,
    circle_twiddles_size: u32,
    line_twiddles_flat: array<u32, N_LINE_TWIDDLES_FLAT_SIZE>,
    line_twiddles_layer_count: u32,
    line_twiddles_sizes: array<u32, N_LINE_TWIDDLES_SIZE>,
    line_twiddles_offsets: array<u32, N_LINE_TWIDDLES_SIZE>,
    mod_inv: u32,
    current_layer: u32,
}

struct BaseColumn {
    data: array<array<M31, N_LANES>, N_ROWS>,
    length: u32,
}

struct M31 {
    data: u32,
}

struct LookupData {
    initial_state: array<array<BaseColumn, N_STATE>, N_INSTANCES_PER_ROW>,
    final_state: array<array<BaseColumn, N_STATE>, N_INSTANCES_PER_ROW>,
}

struct GenTraceOutput {
    trace: array<BaseColumn, N_COLUMNS>,
    lookup_data: LookupData,
}

struct Results {
    values: array<u32, MAX_ARRAY_SIZE>,
}

@group(0) @binding(0)
var<storage, read> input: GenTraceInput;

@group(0) @binding(1)
var<storage, read_write> gen_trace_output: GenTraceOutput;

@group(0) @binding(2)
var<storage, read_write> output: Results;

var<workgroup> shared_values: array<u32, MAX_SHARED_SIZE>;

@compute @workgroup_size(INTERPOLATE_THREADS_PER_WORKGROUP)
fn interpolate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let workgroups_size = INTERPOLATE_WORKGROUP_SIZE;
    let threads_per_workgroup = INTERPOLATE_THREADS_PER_WORKGROUP;
    let size = 1u << (input.log_size - 1u);
    
    let workgroup_id = global_id.y;
    let local_thread_id = global_id.x;

    let column_idx = threads_per_workgroup * workgroup_id + local_thread_id;
    if (column_idx >= N_COLUMNS) {
        return;
    }

    for (var i = 0u; i < size; i = i + 1u) {
        let idx0 = i << 1u;
        let idx1 = idx0 + 1u;

        let outer_idx = idx0 / N_LANES;
        let inner_idx = idx0 % N_LANES;
        var val0 = gen_trace_output.trace[column_idx].data[outer_idx][inner_idx].data;
        var val1 = gen_trace_output.trace[column_idx].data[outer_idx][inner_idx + 1u].data;

        ibutterfly(&val0, &val1, input.circle_twiddles[i]);

        output.values[column_idx * (1u << input.log_size) + idx0] = val0;
        output.values[column_idx * (1u << input.log_size) + idx1] = val1;
    }

    interpolate_compute(column_idx);
}

fn interpolate_compute(column_idx: u32) {
    // Process line_twiddles
    var layer = 0u;
    loop {
        let layer_size = input.line_twiddles_sizes[layer];
        let layer_offset = input.line_twiddles_offsets[layer];
        let step = 1u << (layer + 1u);
        
        for (var h = 0u; h < layer_size; h += 1u) {
            let t = input.line_twiddles_flat[layer_offset + h];
            let idx0_offset = (h << (layer + 2u));
            
            for (var l = 0u; l < step; l += 1u) {
                let idx0 = idx0_offset + l;
                let idx1 = idx0 + step;
                
                var val0 = output.values[column_idx * (1u << input.log_size) + idx0];
                var val1 = output.values[column_idx * (1u << input.log_size) + idx1];
                
                ibutterfly(&val0, &val1, t);
                
                output.values[column_idx * (1u << input.log_size) + idx0] = val0;
                output.values[column_idx * (1u << input.log_size) + idx1] = val1;
            }
        }

        layer = layer + 1u;
        if (layer >= input.line_twiddles_layer_count) { break; }
    }

    mod_mul_compute(column_idx);
}

fn mod_mul_compute(column_idx: u32) {
    for (var i = 0u; i < (1u << input.log_size); i += 1u) {
        output.values[column_idx * (1u << input.log_size) + i] = mod_mul(output.values[column_idx * (1u << input.log_size) + i], input.mod_inv);
    }
}
