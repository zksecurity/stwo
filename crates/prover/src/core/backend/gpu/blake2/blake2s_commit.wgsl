// Define maximum sizes for the fixed-size arrays.
// Adjust these values to match your application's expected maximum sizes.

// Combined input storage struct containing layer parameters,
// previous layer hash data, and column data.
struct Input {
    log_size: u32,         // Logarithmic size of the layer.
    num_columns: u32,      // Number of columns.
    node_count: u32,       // Number of nodes = 1 << log_size.
    prevLayerPresent: u32, // Flag indicating whether the previous layer exists (0 or 1).
    // Previous layer hash data:
    // Each hash consists of 8 u32, and two child hashes (16 words) per node are stored consecutively.
    prevLayer: array<Blake2sHash, MAX_PREV_LAYER_WORDS>,
    // Column data:
    // Each column is stored as an array of length node_count.
    columns: array<Columns, MAX_COLUMNS>,
};

@group(0) @binding(0)
var<storage, read> input: Input;

// Output layer buffer: stores the hash result for each node (each hash consists of 8 u32).
@group(0) @binding(1)
var<storage, read_write> outLayer: array<Blake2sHash, MAX_COLUMNS>;

@compute @workgroup_size(1)
fn main() {
    for (var i: u32 = 0u; i < input.node_count; i = i + 1u) {
        // Local Blake2sHash structs to store left and right child hash values.
        var left: Blake2sHash;
        var right: Blake2sHash;

        if (input.prevLayerPresent != 0u) {
            // If the previous layer exists, two child hashes (each 8 words) are stored consecutively.
            let left_index = 2u * i;
            let right_index = left_index + 1u;
            left = input.prevLayer[left_index];
            right = input.prevLayer[right_index];
        }

        // For each column, store the value corresponding to the current node (i) into a local array.
        var local_column_values: array<u32, MAX_COLUMNS>;
        for (var c: u32 = 0u; c < input.num_columns; c = c + 1u) {
            // Each column is stored as an array of length node_count.
            local_column_values[c] = input.columns[c].columns[i];
        }

        // Call hash_node: This function should take the input parameters as before,
        // but return the result as a Blake2sHash struct.
        let result: Blake2sHash =
            hash_node(input.prevLayerPresent, left, right, &local_column_values, input.num_columns);

        outLayer[i] = result;
    }
}
