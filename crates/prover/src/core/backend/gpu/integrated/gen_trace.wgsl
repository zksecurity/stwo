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

fn combine(values: array<M31, N_STATE>) -> QM31 {
    var result: QM31 = QM31(
        CM31(M31(0u), M31(0u)),
        CM31(M31(0u), M31(0u))
    );

    for (var j: u32 = 0u; j < N_STATE; j = j + 1u) {
        let value_q: QM31 = QM31(
            CM31(values[j], M31(0u)),
            CM31(M31(0u), M31(0u))
        );
        result = qm31_add(result, qm31_mul(input.lookup_elements.alpha_powers[j], value_q));
    }

    return qm31_sub(result, input.lookup_elements.z);
}

@compute @workgroup_size(GEN_INTERACTION_TRACE_THREADS_PER_WORKGROUP)
fn compute_interaction_trace(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    var total_rows: u32 = 1u << input.log_size;

    var num_threads: u32 = GEN_INTERACTION_TRACE_THREADS_PER_WORKGROUP;
    var thread_id: u32 = global_id.x;
    var chunk_size: u32 = (total_rows + num_threads - 1u) / num_threads;
    var chunk_start: u32 = thread_id * chunk_size;
    var chunk_end: u32 = min(chunk_start + chunk_size, total_rows);

    for (var rep_i: u32 = 0u; rep_i < N_INSTANCES_PER_ROW; rep_i = rep_i + 1u) {
        for (var row: u32 = chunk_start; row < chunk_end; row = row + 1u) {
            var initial_values: array<M31, N_STATE>;
            var final_values: array<M31, N_STATE>;

            for (var j: u32 = 0u; j < N_STATE; j = j + 1u) {
                var vec_index = row / N_LANES;
                var inner_vec_index = row % N_LANES;
                initial_values[j] = gen_trace_output.lookup_data.initial_state[rep_i][j].data[vec_index][inner_vec_index];
                final_values[j]   = gen_trace_output.lookup_data.final_state[rep_i][j].data[vec_index][inner_vec_index];
            }
            var denom0: QM31 = combine(initial_values);
            var denom1: QM31 = combine(final_values);

            // (1/denom1 - 1/denom0) = (denom1 - denom0) / (denom0 * denom1)
            var num: QM31 = qm31_sub(denom1, denom0);
            var den: QM31 = qm31_mul(denom0, denom1);

            // fraction = num / den
            var fraction: QM31 = qm31_mul(num,qm31_inverse(den));
            gen_interaction_trace_output.interaction_trace_qm31[rep_i].data[row] = fraction;
        }
        workgroupBarrier();

        for (var row: u32 = chunk_start; row < chunk_end; row = row + 1u) {
            var prev_value: QM31;
            if (rep_i == 0u) {
                prev_value = QM31(
                    CM31(M31(0u), M31(0u)),
                    CM31(M31(0u), M31(0u))
                );
            } else {
                prev_value = gen_interaction_trace_output.interaction_trace_qm31[rep_i - 1].data[row];
            }
            gen_interaction_trace_output.interaction_trace_qm31[rep_i].data[row] = qm31_add(gen_interaction_trace_output.interaction_trace_qm31[rep_i].data[row], prev_value);
        }
        gen_interaction_trace_output.interaction_trace_qm31[rep_i].length = N_LANES * N_ROWS;
        workgroupBarrier();
    }

    // finalize_last
    if (global_id.x == 0u) {
        var log_size = input.log_size;
        var last_rep: u32 = N_INSTANCES_PER_ROW - 1u;

        // copy last col to interaction_trace_buffers[0]
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            gen_interaction_trace_output.interaction_trace_buffers[0].data[r] = gen_interaction_trace_output.interaction_trace_qm31[last_rep].data[r];
        }
        // last col is in bit-reversed order

        // bit-reversed order to normal order(circle domain order)
        var target_index: u32 = 1u;
        var source_index: u32 = 0u;
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            gen_interaction_trace_output.interaction_trace_buffers[target_index].data[r] = gen_interaction_trace_output.interaction_trace_buffers[source_index].data[bit_reverse_index(r, log_size)];
        }

        // circle domain order to coset order
        target_index = 0u;
        source_index = 1u;
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            gen_interaction_trace_output.interaction_trace_buffers[target_index].data[circle_domain_index_to_coset_index(r, total_rows)] = gen_interaction_trace_output.interaction_trace_buffers[source_index].data[r];
        }

        // calculate prefix sum in coset order
        target_index = 0u;
        for (var r: u32 = 1u; r < total_rows; r = r + 1u) {
            gen_interaction_trace_output.interaction_trace_buffers[target_index].data[r] = qm31_add(gen_interaction_trace_output.interaction_trace_buffers[target_index].data[r - 1u], gen_interaction_trace_output.interaction_trace_buffers[target_index].data[r]);
        }

        // coset order to circle domain order
        target_index = 1u;
        source_index = 0u;
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            gen_interaction_trace_output.interaction_trace_buffers[target_index].data[coset_index_to_circle_domain_index(r, log_size)] = gen_interaction_trace_output.interaction_trace_buffers[source_index].data[r];
        }

        // normal order(circle domain order) to bit-reversed order
        target_index = 0u;
        source_index = 1u;
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            gen_interaction_trace_output.interaction_trace_buffers[target_index].data[r] = gen_interaction_trace_output.interaction_trace_buffers[source_index].data[bit_reverse_index(r, log_size)];
        }

        var coset_index: u32 = coset_index_to_circle_domain_index(total_rows - 1u, log_size);
        var fixed_index: u32 = bit_reverse_index(coset_index, log_size);

        var total_sum: QM31 = gen_interaction_trace_output.interaction_trace_buffers[target_index].data[fixed_index];
        gen_interaction_trace_output.total_sum = total_sum;

        // copy interaction_trace_buffers[0] to last column
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            gen_interaction_trace_output.interaction_trace_qm31[last_rep].data[r] = gen_interaction_trace_output.interaction_trace_buffers[target_index].data[r];
        }
    }
}

@compute @workgroup_size(GEN_INTERACTION_TRACE_THREADS_PER_WORKGROUP)
fn interaction_trace_to_original_column(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    var total_rows: u32 = 1u << input.log_size;

    var num_threads: u32 = GEN_INTERACTION_TRACE_THREADS_PER_WORKGROUP;
    var thread_id: u32 = global_id.x;
    var chunk_size: u32 = (total_rows + num_threads - 1u) / num_threads;
    var chunk_start: u32 = thread_id * chunk_size;
    var chunk_end: u32 = min(chunk_start + chunk_size, total_rows);

    for (var rep_i: u32 = 0u; rep_i < N_INSTANCES_PER_ROW; rep_i = rep_i + 1u) {
        for (var row: u32 = chunk_start; row < chunk_end; row = row + 1u) {
            let interaction_trace_column_index = rep_i * 4u + N_INTERACTION_TRACE_COLUMN_OFFSET;
            gen_trace_output.original_trace[interaction_trace_column_index].data[row] = gen_interaction_trace_output.interaction_trace_qm31[rep_i].data[row].a.a;
            gen_trace_output.original_trace[interaction_trace_column_index + 1u].data[row] = gen_interaction_trace_output.interaction_trace_qm31[rep_i].data[row].a.b;
            gen_trace_output.original_trace[interaction_trace_column_index + 2u].data[row] = gen_interaction_trace_output.interaction_trace_qm31[rep_i].data[row].b.a;
            gen_trace_output.original_trace[interaction_trace_column_index + 3u].data[row] = gen_interaction_trace_output.interaction_trace_qm31[rep_i].data[row].b.b;
        }
    }
}
