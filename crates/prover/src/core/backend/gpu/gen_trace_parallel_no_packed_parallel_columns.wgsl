const MODULUS_BITS: u32 = 31u;
const P: u32 = 2147483647u;

// Define constants
const N_ROWS: u32 = 32;
const N_STATE: u32 = 16;
const N_INSTANCES_PER_ROW: u32 = 8;
const N_COLUMNS: u32 = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
const N_HALF_FULL_ROUNDS: u32 = 4;
const FULL_ROUNDS: u32 = 2u * N_HALF_FULL_ROUNDS;
const N_PARTIAL_ROUNDS: u32 = 14;
const N_LANES: u32 = 16;
const N_COLUMNS_PER_REP: u32 = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
const LOG_N_LANES: u32 = 4;
const THREADS_PER_WORKGROUP: u32 = 256;
const WORKGROUP_SIZE: u32 = 16;
const TOTAL_THREAD_SIZE: u32 = THREADS_PER_WORKGROUP * WORKGROUP_SIZE;

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

struct M31 {
    data: u32,
}

struct GenTraceInput {
    log_n_rows: u32,
}

struct StateData {
    data: array<M31, N_STATE>,
}

struct LookupData {
    initial_state: array<array<BaseColumn, N_STATE>, N_INSTANCES_PER_ROW>,
    final_state: array<array<BaseColumn, N_STATE>, N_INSTANCES_PER_ROW>,
}

struct GenTraceOutput {
    trace: array<BaseColumn, N_COLUMNS>,
    lookup_data: LookupData,
}

struct ShaderResult {
    values: array<Ids, TOTAL_THREAD_SIZE>,
}

struct Ids {
    workgroup_id_x: u32,
    workgroup_id_y: u32,
    workgroup_id_z: u32,
    local_invocation_id_x: u32,
    local_invocation_id_y: u32,
    local_invocation_id_z: u32,
    global_invocation_id_x: u32,
    global_invocation_id_y: u32,
    global_invocation_id_z: u32,
    local_invocation_index: u32,
    num_workgroups_x: u32,
    num_workgroups_y: u32,
    num_workgroups_z: u32,
    workgroup_index: u32,
    global_invocation_index: u32,
}

@group(0) @binding(0)
var<storage, read> input: GenTraceInput;

// Output buffer
@group(0) @binding(1)
var<storage, read_write> output: GenTraceOutput;

@group(0) @binding(2)
var<storage, read_write> state_data: StateData;

@group(0) @binding(3)
var<storage, read_write> shader_result: ShaderResult;

@compute @workgroup_size(256)
fn gen_trace(
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

    let global_invocation_index = workgroup_index * THREADS_PER_WORKGROUP + local_invocation_index;

    shader_result.values[global_invocation_index] = Ids(
        workgroup_id.x,
        workgroup_id.y,
        workgroup_id.z,
        local_invocation_id.x,
        local_invocation_id.y,
        local_invocation_id.z,
        global_invocation_id.x,
        global_invocation_id.y,
        global_invocation_id.z,
        local_invocation_index,
        num_workgroups.x,
        num_workgroups.y,
        num_workgroups.z,
        workgroup_index,
        global_invocation_index,
    );

    for (var i = 0u; i < N_COLUMNS; i++) {
        output.trace[i].length = N_ROWS * N_LANES;
    }

    for (var i = 0u; i < N_INSTANCES_PER_ROW; i++) {
        for (var j = 0u; j < N_STATE; j++) {
            output.lookup_data.initial_state[i][j].length = N_ROWS * N_LANES;
            output.lookup_data.final_state[i][j].length = N_ROWS * N_LANES;
        }
    }

    let log_size = input.log_n_rows;

    var instance_index = global_invocation_index / (N_ROWS * N_LANES);
    var instance_rep_index = global_invocation_index % (N_ROWS * N_LANES);
    var vec_index = instance_rep_index / N_LANES;
    var inner_vec_index = instance_rep_index % N_LANES;
    // for (var vec_index = 0u; vec_index < (1u << (log_size - LOG_N_LANES)); vec_index++) {
        var col_index = instance_index * N_COLUMNS_PER_REP;

        var rep_i = instance_index;
        // for (var rep_i = 0u; rep_i < N_INSTANCES_PER_ROW; rep_i++) {
            var state: array<M31, N_STATE> = initialize_state(vec_index, inner_vec_index, rep_i);

            for (var i = 0u; i < N_STATE; i++) {
                output.trace[col_index].data[vec_index][inner_vec_index] = state[i];
                col_index += 1u;
            }

            for (var i = 0u; i < N_STATE; i++) {
                output.lookup_data.initial_state[rep_i][i].data[vec_index][inner_vec_index] = state[i];
                output.lookup_data.initial_state[rep_i][i].length = N_ROWS * N_LANES;
            }

            // 4 full rounds
            for (var i = 0u; i < N_HALF_FULL_ROUNDS; i++) {
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = add(state[j], M31(EXTERNAL_ROUND_CONSTS[i][j]));
                }
                state = apply_external_round_matrix(state);
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = pow5(state[j]);
                }
                for (var j = 0u; j < N_STATE; j++) {
                    output.trace[col_index].data[vec_index][inner_vec_index] = state[j];
                    col_index += 1u;
                }
            }
            // Partial rounds
            for (var i = 0u; i < N_PARTIAL_ROUNDS; i++) {
                state[0] = add(state[0], M31(INTERNAL_ROUND_CONSTS[i]));
                state = apply_internal_round_matrix(state);
                state[0] = pow5(state[0]);
                output.trace[col_index].data[vec_index][inner_vec_index] = state[0];
                col_index += 1u;
            }
            // 4 full rounds
            for (var i = 0u; i < N_HALF_FULL_ROUNDS; i++) {
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = add(state[j], M31(EXTERNAL_ROUND_CONSTS[i + N_HALF_FULL_ROUNDS][j]));
                }
                state = apply_external_round_matrix(state);
                for (var j = 0u; j < N_STATE; j++) {
                    state[j] = pow5(state[j]);
                }
                for (var j = 0u; j < N_STATE; j++) {
                    output.trace[col_index].data[vec_index][inner_vec_index] = state[j];
                    col_index += 1u;
                }
            }

            for (var j = 0u; j < N_STATE; j++) {
                output.lookup_data.final_state[rep_i][j].data[vec_index][inner_vec_index] = state[j];
            }
        // }
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

fn add(a: M31, b: M31) -> M31 {
    return M31(partial_reduce(a.data + b.data));
}

fn mul(a: M31, b: M31) -> M31 {
    return M31(full_reduce(u64(a.data) * u64(b.data)));
}

// Partial reduce for values in [0, 2P)
fn partial_reduce(val: u32) -> u32 {
    let reduced = val - P;
    return select(val, reduced, reduced < val);
}

fn full_reduce(val: u64) -> u32 {
    let first_shift = val >> MODULUS_BITS;
    let first_sum = first_shift + val + 1;
    let second_shift = first_sum >> MODULUS_BITS;
    let final_sum = second_shift + val;
    return u32(final_sum & u64(P));
}

// Function to apply pow5 operation
fn pow5(x: M31) -> M31 {
    return mul(mul(mul(x, x), mul(x, x)), x);
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
        let s = add(add(modified_state[j], modified_state[j + 4]), add(modified_state[j + 8], modified_state[j + 12]));
        for (var i = 0u; i < 4u; i++) {
            modified_state[4 * i + j] = add(modified_state[4 * i + j], s);
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
        sum = add(sum, state[i]);
    }

    var result = array<M31, N_STATE>();
    for (var i = 0u; i < N_STATE; i++) {
        let factor = partial_reduce(1u << (i + 1));
        result[i] = add(mul(M31(factor), state[i]), sum);
    }

    return result;
}

/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
fn apply_m4(x: array<M31, 4>) -> array<M31, 4> {
    let t0 = add(x[0], x[1]);
    let t02 = add(t0, t0);
    let t1 = add(x[2], x[3]);
    let t12 = add(t1, t1);
    let t2 = add(add(x[1], x[1]), t1);
    let t3 = add(add(x[3], x[3]), t0);
    let t4 = add(add(t12, t12), t3);
    let t5 = add(add(t02, t02), t2);
    let t6 = add(t3, t5);
    let t7 = add(t2, t4);
    return array<M31, 4>(t6, t5, t7, t4);
}
