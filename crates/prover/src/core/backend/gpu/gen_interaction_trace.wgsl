// will include gen_trace_interpolate_columns_constants.wgsl

struct QM31Column {
    data: array<QM31, N_ROWS>,
    length: u32
}

struct BaseColumn {
    data: array<array<M31, N_LANES>, N_ROWS>,
    length: u32,
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

struct GenInteractionTraceInput {
    log_size: u32,
    lookup_data: LookupData,
    lookup_elements: LookupElements,
};

struct GenInteractionTraceOutput {
    interaction_trace_qm31: array<QM31Column, N_INSTANCES_PER_ROW>,
    interaction_trace_buffers: array<QM31Column, 2>,
    total_sum: QM31
    // interaction_trace: array<BaseColumn, N_INTERACTION_COLUMNS>,
    // interaction_trace: array<OriginalColumn, N_INTERACTION_COLUMNS>,
}

@group(0) @binding(0)
var<storage, read> input: GenInteractionTraceInput;

@group(0) @binding(1)
var<storage, read_write> output: GenInteractionTraceOutput;

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
                initial_values[j] = input.lookup_data.initial_state[rep_i][j].data[vec_index][inner_vec_index];
                final_values[j]   = input.lookup_data.final_state[rep_i][j].data[vec_index][inner_vec_index];
            }
            var denom0: QM31 = combine(initial_values);
            var denom1: QM31 = combine(final_values);

            // (1/denom1 - 1/denom0) = (denom1 - denom0) / (denom0 * denom1)
            var num: QM31 = qm31_sub(denom1, denom0);
            var den: QM31 = qm31_mul(denom0, denom1);

            // fraction = num / den
            var fraction: QM31 = qm31_mul(num,qm31_inverse(den));
            output.interaction_trace_qm31[rep_i].data[row] = fraction;
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
                prev_value = output.interaction_trace_qm31[rep_i - 1].data[row];
            }
            output.interaction_trace_qm31[rep_i].data[row] = qm31_add(output.interaction_trace_qm31[rep_i].data[row], prev_value);
        }
        workgroupBarrier();
    }

    // finalize_last
    if (global_id.x == 0u) {
        var log_size = input.log_size;
        var last_rep: u32 = N_INSTANCES_PER_ROW - 1u;

        // copy last col to interaction_trace_buffers[0]
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            output.interaction_trace_buffers[0].data[r] = output.interaction_trace_qm31[last_rep].data[r];
        }
        // last col is in bit-reversed order

        // bit-reversed order to normal order(circle domain order)
        var target_index: u32 = 1u;
        var source_index: u32 = 0u;
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            output.interaction_trace_buffers[target_index].data[r] = output.interaction_trace_buffers[source_index].data[bit_reverse_index(r, log_size)];
        }

        // circle domain order to coset order
        target_index = 0u;
        source_index = 1u;
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            output.interaction_trace_buffers[target_index].data[r] = output.interaction_trace_buffers[source_index].data[circle_domain_index_to_coset_index(r, total_rows)];
        }

        // calculate prefix sum in coset order
        target_index = 0u;
        for (var r: u32 = 1u; r < total_rows; r = r + 1u) {
            output.interaction_trace_buffers[target_index].data[r] = qm31_add(output.interaction_trace_buffers[target_index].data[r - 1u], output.interaction_trace_buffers[source_index].data[r]);
        }

        // coset order to circle domain order
        target_index = 1u;
        source_index = 0u;
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            output.interaction_trace_buffers[target_index].data[r] = output.interaction_trace_buffers[source_index].data[coset_index_to_circle_domain_index(r, log_size)];
        }

        // normal order(circle domain order) to bit-reversed order
        target_index = 0u;
        source_index = 1u;
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            output.interaction_trace_buffers[target_index].data[r] = output.interaction_trace_buffers[source_index].data[bit_reverse_index(r, log_size)];
        }

        var coset_index: u32 = coset_index_to_circle_domain_index(total_rows - 1u, log_size);
        var fixed_index: u32 = bit_reverse_index(coset_index, log_size);

        var total_sum: QM31 = output.interaction_trace_buffers[target_index].data[fixed_index];
        output.total_sum = total_sum;

        // copy interaction_trace_buffers[0] to last column
        for (var r: u32 = 0u; r < total_rows; r = r + 1u) {
            output.interaction_trace_qm31[last_rep].data[r] = output.interaction_trace_buffers[target_index].data[r];
        }
    }
}
