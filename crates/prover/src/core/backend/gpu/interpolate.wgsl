// Note: depends on gen_trace_interpolate_columns_constants.wgsl

fn ibutterfly(v0: ptr<function, M31>, v1: ptr<function, M31>, itwid: M31) {
    let tmp = *v0;
    *v0 = m31_add(tmp, *v1);
    *v1 = m31_mul(m31_sub(tmp, *v1), itwid);
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

struct GenTraceInput {
    log_size: u32,
    twiddles: Twiddles,
}

struct BaseColumn {
    data: array<array<M31, N_LANES>, N_ROWS>,
    length: u32,
}

struct LookupData {
    initial_state: array<array<BaseColumn, N_STATE>, N_INSTANCES_PER_ROW>,
    final_state: array<array<BaseColumn, N_STATE>, N_INSTANCES_PER_ROW>,
}

struct OriginalColumn {
    data: array<M31, N_ORIGINAL_COLUMN_SIZE>,
}

struct GenTraceOutput {
    original_trace: array<OriginalColumn, N_ORIGINAL_TRACE_COLUMNS>,
    trace: array<BaseColumn, N_COLUMNS>,
    lookup_data: LookupData,
}

struct Results {
    values: array<M31, N_FLAT_MAX_ARRAY_SIZE>,
}

@group(0) @binding(0)
var<storage, read> input: GenTraceInput;

@group(0) @binding(1)
var<storage, read_write> gen_trace_output: GenTraceOutput;

@group(0) @binding(2)
var<storage, read_write> output: Results;

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
        var val0 = gen_trace_output.trace[column_idx].data[outer_idx][inner_idx];
        var val1 = gen_trace_output.trace[column_idx].data[outer_idx][inner_idx + 1u];

        ibutterfly(&val0, &val1, input.twiddles.circle_twiddles[i]);

        output.values[column_idx * (1u << input.log_size) + idx0] = val0;
        output.values[column_idx * (1u << input.log_size) + idx1] = val1;
    }

    interpolate_compute(column_idx);
}

fn interpolate_compute(column_idx: u32) {
    // Process line_twiddles
    var layer = 0u;
    loop {
        let layer_size = input.twiddles.line_twiddles_sizes[layer];
        let layer_offset = input.twiddles.line_twiddles_offsets[layer];
        let step = 1u << (layer + 1u);
        
        for (var h = 0u; h < layer_size; h += 1u) {
            let t = input.twiddles.line_twiddles_flat[layer_offset + h];
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
        if (layer >= input.twiddles.line_twiddles_layer_count) { break; }
    }

    mod_mul_compute(column_idx);
}

fn mod_mul_compute(column_idx: u32) {
    for (var i = 0u; i < (1u << input.log_size); i += 1u) {
        output.values[column_idx * (1u << input.log_size) + i] = m31_mul(output.values[column_idx * (1u << input.log_size) + i], input.twiddles.mod_inv);
    }
}
