// Note: depends on gen_trace_interpolate_columns_constants.wgsl

// Initialize EXTERNAL_ROUND_CONSTS with explicit values
var<private> EXTERNAL_ROUND_CONSTS: array<array<u32, N_STATE>, FULL_ROUNDS> = array<array<u32, N_STATE>, FULL_ROUNDS>(
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
    array<u32, N_STATE>(1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u, 1234u),
);

// Initialize INTERNAL_ROUND_CONSTS with explicit values
var<private> INTERNAL_ROUND_CONSTS: array<u32, N_PARTIAL_ROUNDS> = array<u32, N_PARTIAL_ROUNDS>(
    1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234
);

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

struct GenTraceInput {
    log_size: u32,
    twiddles: Twiddles,
    lookup_elements: LookupElements,
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

struct OriginalColumn {
    data: array<M31, N_ORIGINAL_COLUMN_SIZE>,
}

struct GenTraceOutput {
    original_trace: array<OriginalColumn, N_ORIGINAL_TRACE_COLUMNS>,
    trace: array<BaseColumn, N_COLUMNS>,
    lookup_data: LookupData,
}


// struct GenInteractionTraceOutput {
//     // chunk 
//     interaction_trace_qm31: array<QM31Column, N_INSTANCES_PER_ROW>,
//     interaction_trace_buffers: array<QM31Column, 4>,
//     total_sum: QM31,
//     // chunk ends
// }

struct Results {
    values: array<M31, N_FLAT_MAX_ARRAY_SIZE>,
}

@group(0) @binding(0)
var<storage, read> input: GenTraceInput;

// Intermediate buffer
@group(0) @binding(1)
var<storage, read_write> gen_trace_output: GenTraceOutput;

@group(0) @binding(2)
var<storage, read_write> interpolate_output: Results;

// @group(0) @binding(3)
// var<storage, read_write> gen_interaction_trace_output: GenInteractionTraceOutput;

@compute @workgroup_size(GEN_TRACE_THREADS_PER_WORKGROUP)
fn gen_trace_interpolate_columns(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let workgroup_index =  
        workgroup_id.x +
        workgroup_id.y * num_workgroups.x +
        workgroup_id.z * num_workgroups.x * num_workgroups.y;

    let global_invocation_index = workgroup_index * GEN_TRACE_THREADS_PER_WORKGROUP + local_invocation_index;

    gen_trace_output.original_trace[0].data[0] = M31(1u);

    for (var i = 0u; i < N_COLUMNS; i++) {
        gen_trace_output.trace[i].length = N_ROWS * N_LANES;
    }

    for (var i = 0u; i < N_INSTANCES_PER_ROW; i++) {
        for (var j = 0u; j < N_STATE; j++) {
            gen_trace_output.lookup_data.initial_state[i][j].length = N_ROWS * N_LANES;
            gen_trace_output.lookup_data.final_state[i][j].length = N_ROWS * N_LANES;
        }
    }

    let log_size = input.log_size;

    var vec_index = global_invocation_index / N_LANES;
    var inner_vec_index = global_invocation_index % N_LANES;
        var col_index = 0u;

        // var rep_i = instance_index;
        for (var rep_i = 0u; rep_i < N_INSTANCES_PER_ROW; rep_i++) {
            var state: array<M31, N_STATE> = initialize_state(vec_index, inner_vec_index, rep_i);

            for (var i = 0u; i < N_STATE; i++) {
                gen_trace_output.trace[col_index].data[vec_index][inner_vec_index] = state[i];
                gen_trace_output.original_trace[col_index + N_PREPROCESSED_COLUMNS].data[global_invocation_index] = state[i];
                col_index += 1u;
            }

            for (var i = 0u; i < N_STATE; i++) {
                gen_trace_output.lookup_data.initial_state[rep_i][i].data[vec_index][inner_vec_index] = state[i];
                gen_trace_output.lookup_data.initial_state[rep_i][i].length = N_ROWS * N_LANES;
            }

            // 4 full rounds
            for (var i = 0u; i < N_HALF_FULL_ROUNDS; i++) {
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = m31_add(state[j], M31(EXTERNAL_ROUND_CONSTS[i][j]));
                }
                state = apply_external_round_matrix(state);
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = pow5(state[j]);
                }
                for (var j = 0u; j < N_STATE; j++) {
                    gen_trace_output.trace[col_index].data[vec_index][inner_vec_index] = state[j];
                    gen_trace_output.original_trace[col_index + N_PREPROCESSED_COLUMNS].data[global_invocation_index] = state[j];
                    col_index += 1u;
                }
            }
            // Partial rounds
            for (var i = 0u; i < N_PARTIAL_ROUNDS; i++) {
                state[0] = m31_add(state[0], M31(INTERNAL_ROUND_CONSTS[i]));
                state = apply_internal_round_matrix(state);
                state[0] = pow5(state[0]);
                gen_trace_output.trace[col_index].data[vec_index][inner_vec_index] = state[0];
                gen_trace_output.original_trace[col_index + N_PREPROCESSED_COLUMNS].data[global_invocation_index] = state[0];
                col_index += 1u;
            }
            // 4 full rounds
            for (var i = 0u; i < N_HALF_FULL_ROUNDS; i++) {
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = m31_add(state[j], M31(EXTERNAL_ROUND_CONSTS[i + N_HALF_FULL_ROUNDS][j]));
                }
                state = apply_external_round_matrix(state);
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = pow5(state[j]);
                }
                for (var j = 0u; j < N_STATE; j++) {
                    gen_trace_output.trace[col_index].data[vec_index][inner_vec_index] = state[j];
                    gen_trace_output.original_trace[col_index + N_PREPROCESSED_COLUMNS].data[global_invocation_index] = state[j];
                    col_index += 1u;
                }
            }

            for (var j = 0u; j < N_STATE; j++) {
                gen_trace_output.lookup_data.final_state[rep_i][j].data[vec_index][inner_vec_index] = state[j];
            }
        }
    // }
}

// Function to initialize the state array
fn initialize_state(vec_index: u32, inner_vec_index: u32, rep_i: u32) -> array<M31, N_STATE> {
    var state: array<M31, N_STATE>;

    for (var state_i = 0u; state_i < N_STATE; state_i++) {
        state[state_i] = M31(vec_index * 16u + inner_vec_index + state_i + rep_i);
    }

    return state;
}

// Function to apply pow5 operation
fn pow5(x: M31) -> M31 {
    return m31_mul(m31_mul(m31_mul(x, x), m31_mul(x, x)), x);
}

/// Applies the external round matrix.
/// See <https://eprint.iacr.org/2023/323.pdf> 5.1 and Appendix B.
fn apply_external_round_matrix(state: array<M31, N_STATE>) -> array<M31, N_STATE> {
    // Applies circ(2M4, M4, M4, M4).
    var modified_state = state;
    for (var i = 0u; i < 4u; i++) {
        let partial_state = array<M31, 4>(
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        );
        let modified_partial_state = apply_m4(partial_state);
        modified_state[4 * i] = modified_partial_state[0];
        modified_state[4 * i + 1] = modified_partial_state[1];
        modified_state[4 * i + 2] = modified_partial_state[2];
        modified_state[4 * i + 3] = modified_partial_state[3];
    }
    for (var j = 0u; j < 4u; j++) {
        let s = m31_add(m31_add(modified_state[j], modified_state[j + 4]), m31_add(modified_state[j + 8], modified_state[j + 12]));
        for (var i = 0u; i < 4u; i++) {
            modified_state[4 * i + j] = m31_add(modified_state[4 * i + j], s);
        }
    }
    return modified_state;
}

// Applies the internal round matrix.
//   mu_i = 2^{i+1} + 1.
// See <https://eprint.iacr.org/2023/323.pdf> 5.2.
fn apply_internal_round_matrix(state: array<M31, N_STATE>) -> array<M31, N_STATE> {
    var sum = state[0];
    for (var i = 1u; i < N_STATE; i++) {
        sum = m31_add(sum, state[i]);
    }

    var result = array<M31, N_STATE>();
    for (var i = 0u; i < N_STATE; i++) {
        let factor = partial_reduce(1u << (i + 1));
        result[i] = m31_add(m31_mul(M31(factor), state[i]), sum);
    }

    return result;
}

/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
fn apply_m4(x: array<M31, 4>) -> array<M31, 4> {
    let t0 = m31_add(x[0], x[1]);
    let t02 = m31_add(t0, t0);
    let t1 = m31_add(x[2], x[3]);
    let t12 = m31_add(t1, t1);
    let t2 = m31_add(m31_add(x[1], x[1]), t1);
    let t3 = m31_add(m31_add(x[3], x[3]), t0);
    let t4 = m31_add(m31_add(t12, t12), t3);
    let t5 = m31_add(m31_add(t02, t02), t2);
    let t6 = m31_add(t3, t5);
    let t7 = m31_add(t2, t4);
    return array<M31, 4>(t6, t5, t7, t4);
}
