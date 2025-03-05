// ---------------------------------------------------------------------------
// Constants and Data Structure Definitions (modify as needed for your application)
// ---------------------------------------------------------------------------
const MAX_TOTAL_COLUMNS: u32 = 256 * 4;
// Maximum flattened output size: for max layers, the flattened array will hold
// sum_{i=0}^{max_log_size} (1 << i) nodes. Adjust as needed.
const MAX_FLAT_SIZE: u32 = 65535;

// Input buffer for commit() containing pre-sorted (in descending order of length) columns.
struct CommitInput {
    total_columns: u32,
    columns: array<Column, MAX_TOTAL_COLUMNS>,
    columns_len: array<u32, MAX_TOTAL_COLUMNS>,
};

// Common input data structure for commit_on_layer and layer computation.
// This structure contains layer parameters, previous layer hash data, and columns for the current layer.
struct InputData {
    log_size: u32,           // Logarithmic size of the current layer (log2 of node count)
    num_columns: u32,        // Number of columns used in the current layer
    node_count: u32,         // Number of nodes in the current layer (1 << log_size)
    prev_layer_present: u32, // Flag indicating whether the previous layer exists (1 if exists, 0 otherwise)
    prev_layer: array<Blake2sHash, MAX_PREV_LAYER_WORDS>,
    columns: array<Column, MAX_COLUMNS>,
};

struct CommitOutput {
    flat_layers: array<Blake2sHash, MAX_FLAT_SIZE>,
    input_data: InputData,
    out_layer: array<Blake2sHash, MAX_PREV_LAYER_WORDS>,
}

struct DebugOutput {
    debugs: array<u32, 1024>,
    count: u32,
}

// Input buffer that includes sorted column data.
@group(0) @binding(0)
var<storage, read> commitInput: CommitInput;

@group(0) @binding(1)
var<storage, read_write> commitOutput: CommitOutput;

@group(0) @binding(2)
var<storage, read_write> debugOutput: DebugOutput;

// Flattened output buffer: All layers are concatenated into this 1D array.
// For a layer with log_size L, the final flat offset is (1 << L) - 1.
// For example, if log_size=0, offset=0; log_size=1, offset=1; log_size=2, offset=3; etc.


// ---------------------------------------------------------------------------
// Helper Function: Integer ilog2 (assumes input is a power of 2)
// ---------------------------------------------------------------------------
fn ilog2(x: u32) -> u32 {
    return u32(log2(f32(x)));
}

// ---------------------------------------------------------------------------
// commit_on_layer Function: Reads from globalInput to compute the hash for each node.
// ---------------------------------------------------------------------------
fn commit_on_layer() {
    for (var i: u32 = 0u; i < commitOutput.input_data.node_count; i = i + 1u) {
        var left: Blake2sHash;
        var right: Blake2sHash;

        if (commitOutput.input_data.prev_layer_present != 0u) {
            let left_index = 2u * i;
            let right_index = left_index + 1u;
            left = commitOutput.input_data.prev_layer[left_index];
            right = commitOutput.input_data.prev_layer[right_index];
        }

        var local_column_values: array<u32, MAX_COLUMNS>;
        // Retrieve the i-th element from each column.
        for (var c: u32 = 0u; c < commitOutput.input_data.num_columns; c = c + 1u) {
            local_column_values[c] = commitOutput.input_data.columns[c].column[i];
        }

        // hash_node: Computes the hash using (prev_layer_present, left, right, column data array, number of columns).
        let result: Blake2sHash =
            hash_node(commitOutput.input_data.prev_layer_present, left, right, &local_column_values, commitOutput.input_data.num_columns);

        commitOutput.out_layer[i] = result;
    }
}

// ---------------------------------------------------------------------------
// WGSL Version of the commit Function: Sequentially compute each layer and
// produce the full commit output as a flattened 1D array.
// For a layer with log_size L, the starting index in flat_layers is (1 << L) - 1.
// ---------------------------------------------------------------------------
@compute @workgroup_size(1)
fn main() {
    // If there are no columns, compute a single layer with log_size 0.
    if (commitInput.total_columns == 0u) {
        commitOutput.input_data.log_size = 0u;
        commitOutput.input_data.num_columns = 0u;
        commitOutput.input_data.node_count = 1u; // 1 node
        commitOutput.input_data.prev_layer_present = 0u;
        // No column data available, so call commit_on_layer directly.
        commit_on_layer();
        // For log_size 0, offset = (1 << 0) - 1 = 0.
        let flat_offset = (1u << 0u) - 1u;
        commitOutput.flat_layers[flat_offset] = commitOutput.out_layer[0];
        return;
    }

    // Compute maximum log_size based on the length of the first sorted column.
    let max_log_size = ilog2(commitInput.columns_len[0]);
    var prev_layer_flag: u32 = 0u;
    var ls: u32 = max_log_size;
    loop {
        // Populate globalInput.columns with columns that match the current log_size.
        var layer_num_columns: u32 = 0u;
        for (var i: u32 = 0u; i < commitInput.total_columns; i = i + 1u) {
            if (ilog2(commitInput.columns_len[i]) == ls) {
                commitOutput.input_data.columns[layer_num_columns] = commitInput.columns[i];
                layer_num_columns = layer_num_columns + 1u;
            }
        }
        commitOutput.input_data.num_columns = layer_num_columns;
        commitOutput.input_data.log_size = ls;
        commitOutput.input_data.node_count = 1u << ls;
        commitOutput.input_data.prev_layer_present = prev_layer_flag;

        // If a previous layer exists, copy its result from outLayer to globalInput.prev_layer.
        if (prev_layer_flag != 0u) {
            let num_prev = commitOutput.input_data.node_count * 2u;
            for (var j: u32 = 0u; j < num_prev; j = j + 1u) {
                commitOutput.input_data.prev_layer[j] = commitOutput.out_layer[j];
            }
        }

        // Compute the current layer.
        commit_on_layer();

        // Write the computed layer into the flattened output buffer.
        // For a layer with log_size L, the flat offset is: (1 << L) - 1.
        let flat_offset = (1u << ls) - 1u;
        for (var k: u32 = 0u; k < commitOutput.input_data.node_count; k = k + 1u) {
            commitOutput.flat_layers[flat_offset + k] = commitOutput.out_layer[k];
        }

        // Mark that subsequent layers have a previous layer.
        prev_layer_flag = 1u;

        if (ls == 0u) {
            break;
        }
        ls = ls - 1u;
    }
}
