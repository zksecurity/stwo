// Define maximum sizes for the fixed-size arrays.
// Adjust these values to match your application's expected maximum sizes.
const MAX_PREV_LAYER_WORDS: u32 = 1024; // Maximum number of u32 words for previous layer data.
const MAX_COLUMNS_WORDS: u32 = 1024;      // Maximum number of u32 words for column data.

struct Blake2sHash {
    h: array<u32, 8>,
}

// Combined input storage struct containing layer parameters,
// previous layer hash data, and column data.
struct Input {
    log_size: u32,         // Logarithmic size of the layer.
    num_columns: u32,      // Number of columns.
    node_count: u32,       // Number of nodes = 1 << log_size.
    prevLayerPresent: u32, // Flag indicating whether the previous layer exists (0 or 1).
    // Previous layer hash data:
    // Each hash consists of 8 u32, and two child hashes (16 words) per node are stored consecutively.
    prevLayer: array<u32, MAX_PREV_LAYER_WORDS>,
    // Column data:
    // Each column is stored as an array of length node_count, stored consecutively per column.
    columns: array<u32, MAX_COLUMNS_WORDS>,
};

@group(0) @binding(0)
var<storage, read> input: Input;

// Output layer buffer: stores the hash result for each node (each hash consists of 8 u32).
@group(0) @binding(1)
var<storage, read_write> outLayer: array<u32, MAX_COLUMNS_WORDS>;

@compute @workgroup_size(1)
fn main() {
    // Process all nodes sequentially.
    for (var i: u32 = 0u; i < input.node_count; i = i + 1u) {

        // Local arrays to store left and right child hash values.
        var left: array<u32, 8>;
        var right: array<u32, 8>;

        if (input.prevLayerPresent != 0u) {
            // If the previous layer exists, two child hashes (each 8 words) are stored consecutively.
            let left_index = 8u * (2u * i);
            let right_index = left_index + 8u;
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                left[j] = input.prevLayer[left_index + j];
                right[j] = input.prevLayer[right_index + j];
            }
        } else {
            // If the previous layer does not exist, fill with 0.
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                left[j] = 0u;
                right[j] = 0u;
            }
        }

        // For each column, store the value corresponding to the current node (i) into a local array.
        var local_column_values: array<u32, MAX_COLUMN_VALUES>;
        for (var c: u32 = 0u; c < input.num_columns; c = c + 1u) {
            // Assumes each column is stored as an array of length node_count.
            let idx = c * input.node_count + i;
            local_column_values[c] = input.columns[idx];
        }

        // Call hash_node: pass the child hash flag, left/right hashes, column values array, and number of columns.
        let result: array<u32, 8> =
            hash_node(input.prevLayerPresent, left, right, &local_column_values, input.num_columns);

        // Write the resulting hash into the output buffer (each node hash is 8 words).
        let out_offset = 8u * i;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            outLayer[out_offset + j] = result[j];
        }
    }
}
