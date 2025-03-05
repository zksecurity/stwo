// Define maximum sizes for the fixed-size arrays.
// Adjust these values to match your application's expected maximum sizes.

// Combined input storage struct containing layer parameters,
// previous layer hash data, and column data.
struct Input {
    log_size: u32,         // Logarithmic size of the layer.
    num_columns: u32,      // Number of columns.
    node_count: u32,       // Number of nodes = 1 << log_size.
    prev_layer_present: u32, // Flag indicating whether the previous layer exists (0 or 1).
    prev_layer: array<Blake2sHash, MAX_PREV_LAYER_WORDS>, // Previous layer hash data.
    columns: array<Column, MAX_COLUMNS>, // Column data.
};

@group(0) @binding(0)
var<storage, read> input: Input;

// Output layer buffer: stores the hash result for each node (each hash consists of 8 u32).
@group(0) @binding(1)
var<storage, read_write> out_layer: array<Blake2sHash, MAX_COLUMNS>;

fn commit_on_layer(
    log_size: u32,
    num_columns: u32,
    node_count: u32,
    prev_layer_present: u32,
) {
    for (var i: u32 = 0u; i < node_count; i = i + 1u) {
        var left: Blake2sHash;
        var right: Blake2sHash;

        if (prev_layer_present != 0u) {
            let left_index = 2u * i;
            let right_index = left_index + 1u;
            left = input.prev_layer[left_index];
            right = input.prev_layer[right_index];
        }

        var local_column_values: array<u32, MAX_COLUMNS>;
        for (var c: u32 = 0u; c < num_columns; c = c + 1u) {
            local_column_values[c] = input.columns[c].column[i];
        }

        let result: Blake2sHash =
            hash_node(prev_layer_present, left, right, &local_column_values, num_columns);

        out_layer[i] = result;
    }
}

@compute @workgroup_size(1)
fn main() {
    commit_on_layer(
        input.log_size,
        input.num_columns,
        input.node_count,
        input.prev_layer_present,
    );
}
