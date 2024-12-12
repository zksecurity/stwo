const MODULUS_BITS: u32 = 31u;
const HALF_BITS: u32 = 16u;
// Mersenne prime P = 2^31 - 1
const P: u32 = 2147483647u;
const MAX_ARRAY_LOG_SIZE: u32 = 20;
const MAX_ARRAY_SIZE: u32 = 1u << MAX_ARRAY_LOG_SIZE;
const MAX_DEBUG_SIZE: u32 = 32;
const MAX_SHARED_SIZE: u32 = 1u << 12;

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

// const product_mask: u64 = 0x7FFFFFFF;

// fn mod_mul(a: u32, b: u32) -> u32 {
//     let product: u64 = u64(a) * u64(b);
//     let low = (product & product_mask) + (product >> 31);
//     let reduced = low - u64(P);
//     return select(u32(low), u32(reduced), u32(reduced) < u32(low));
// }

fn ibutterfly(v0: ptr<function, u32>, v1: ptr<function, u32>, itwid: u32) {
    let tmp = *v0;
    *v0 = partial_reduce(tmp + *v1);
    *v1 = mod_mul(partial_reduce(tmp + P - *v1), itwid);
}

struct InterpolateData {
    values: array<u32, MAX_ARRAY_SIZE>,
    initial_x: u32,
    initial_y: u32,
    log_size: u32,
    circle_twiddles: array<u32, MAX_ARRAY_SIZE>,
    circle_twiddles_size: u32,
    line_twiddles_flat: array<u32, MAX_ARRAY_SIZE>,
    line_twiddles_layer_count: u32,
    line_twiddles_sizes: array<u32, MAX_ARRAY_SIZE>,
    line_twiddles_offsets: array<u32, MAX_ARRAY_SIZE>,
    mod_inv: u32,
    current_layer: u32,
}

struct Results {
    values: array<u32, MAX_ARRAY_SIZE>,
}

struct DebugData {
    index: array<u32, MAX_DEBUG_SIZE>,
    values: array<u32, MAX_DEBUG_SIZE>,
    counter: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> input: InterpolateData;
@group(0) @binding(1) var<storage, read_write> output: Results;
@group(0) @binding(2) var<storage, read_write> debug_buffer: DebugData;

var<workgroup> shared_values: array<u32, MAX_SHARED_SIZE>;

fn store_debug_value(index: u32, value: u32) {
    let debug_idx = atomicAdd(&debug_buffer.counter, 1u);
    debug_buffer.index[debug_idx] = index;
    debug_buffer.values[debug_idx] = value;
}

@compute @workgroup_size(64)
fn interpolate_first_circle_twiddle(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let workgroup_dispatch = 256u;
    let workgroup_size = 64u;
    let thread_size = workgroup_dispatch * workgroup_size;
    let size = 1u << (input.log_size - 1u);
    
    let workgroup_id = global_id.y;
    let local_id = global_id.x;

    let workgroup_chunk_size = (size + workgroup_dispatch - 1u) / workgroup_dispatch;
    let thread_chunk_size = (workgroup_chunk_size + workgroup_size - 1u) / workgroup_size;
    let start_idx = workgroup_id * workgroup_chunk_size + local_id * thread_chunk_size;
    let end_idx = min(start_idx + thread_chunk_size, size);

    // store_debug_value(thread_id, global_id.y);
    for (var i = start_idx; i < end_idx; i = i + 1u) {
        let idx0 = i << 1u;
        let idx1 = idx0 + 1u;

        var val0 = input.values[idx0];
        var val1 = input.values[idx1];

        ibutterfly(&val0, &val1, input.circle_twiddles[i]);

        output.values[idx0] = val0;
        output.values[idx1] = val1;
    }
}

@compute @workgroup_size(64)
fn interpolate_big_line_twiddle(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let workgroup_dispatch = 256u;
    let workgroup_size = 64u;
    let thread_size = workgroup_dispatch * workgroup_size;

    let workgroup_id = global_id.y;
    let local_id = global_id.x;
    var workgroup_offset = 0u;

    var workgroup_loop_count = 1u;
    var loop_size_offset = 0u;
    // if (input.log_size > 20u) {
    //     workgroup_loop_count = 1u << (input.log_size - 20u);
    //     loop_size_offset = 1u << (input.log_size - 1u);
    // }
    //store_debug_value(global_id.y, workgroup_loop_count);

    for (var k = 0u; k < workgroup_loop_count; k = k + 1u) {
        var copy_start_idx = 0u;
        var copy_end_idx = 0u;

        {
            var size = 1u << input.log_size;
            // if (input.log_size > 20u) {
            //     size = 1u << 20u;
            // }
            let workgroup_chunk_size = (size + workgroup_dispatch - 1u) / workgroup_dispatch;
            let thread_chunk_size = (workgroup_chunk_size + workgroup_size - 1u) / workgroup_size;

            workgroup_offset = workgroup_id * workgroup_chunk_size;
            copy_start_idx = workgroup_offset + local_id * thread_chunk_size;
            copy_end_idx = min(copy_start_idx + thread_chunk_size, size);

        }

        var first_layer_size = input.line_twiddles_sizes[0];
        // if (input.log_size > 20u) {
        //     first_layer_size = 1u << 19u;
        //     //store_debug_value(global_id.y, first_layer_size);
        // }

        let workgroup_chunk_size = (first_layer_size + workgroup_dispatch - 1u) / workgroup_dispatch;
        let thread_chunk_size = (workgroup_chunk_size + workgroup_size - 1u) / workgroup_size;

        var start_idx = workgroup_id * workgroup_chunk_size + local_id * thread_chunk_size;
        var end_idx = min(start_idx + thread_chunk_size, first_layer_size);


        for (var i = copy_start_idx; i < copy_end_idx; i = i + 1u) {
            shared_values[i - workgroup_offset] = output.values[i];
        }

        workgroupBarrier();

        // Process line_twiddles
        var layer = 0u;
        loop {
            let layer_size = input.line_twiddles_sizes[layer];
            let layer_offset = input.line_twiddles_offsets[layer];
            let step = 1u << (layer + 1u);

            for (var h = start_idx; h < end_idx; h = h + 1u) {
                let t = input.line_twiddles_flat[layer_offset + h];
                let idx0_offset = (h << (layer + 2u)) - workgroup_offset;

                for (var l = 0u; l < step; l = l + 1u) {
                    let idx0 = idx0_offset + l;
                    let idx1 = idx0 + step;

                    var val0 = shared_values[idx0];
                    var val1 = shared_values[idx1];

                    ibutterfly(&val0, &val1, t);

                    shared_values[idx0] = val0;
                    shared_values[idx1] = val1;
                }
            }

            workgroupBarrier();

            layer = layer + 1u;
            start_idx = start_idx >> 1u;
            end_idx = end_idx >> 1u;

            if (layer >= 6) { break; }
        }

        // copy values from shared memory to storage
        for (var i = copy_start_idx; i < copy_end_idx; i = i + 1u) {
            output.values[i] = shared_values[i - workgroup_offset];
        }
    }
}

@compute @workgroup_size(256)
fn interpolate_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // dispatch = 1
    let thread_size = 256u;

    let size = 1u << input.log_size;
    let thread_id = global_id.x;

    // Process line_twiddles
    var layer = 6u;
    loop {
        let layer_size = input.line_twiddles_sizes[layer];
        let layer_offset = input.line_twiddles_offsets[layer];
        let step = 1u << (layer + 1u);
        
        for (var h = 0u; h < layer_size; h = h + 1u) {
            let t = input.line_twiddles_flat[layer_offset + h];
            let idx0_offset = (h << (layer + 2u));
            
            for (var l = thread_id; l < step; l = l + thread_size) {
                let idx0 = idx0_offset + l;
                let idx1 = idx0 + step;
                
                var val0 = output.values[idx0];
                var val1 = output.values[idx1];
                
                ibutterfly(&val0, &val1, t);
                
                output.values[idx0] = val0;
                output.values[idx1] = val1;
            }

            storageBarrier();
        }

        layer = layer + 1u;
        if (layer >= input.line_twiddles_layer_count) { break; }
    }

    // for (var i = thread_id; i < size; i = i + thread_size) {
    //     output.values[i] = mod_mul(output.values[i], input.mod_inv);
    // }
}

@compute @workgroup_size(64)
fn mod_mul_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let workgroup_dispatch = 256u;
    let workgroup_size = 64u;
    let thread_size = workgroup_dispatch * workgroup_size;

    let workgroup_id = global_id.y;
    let local_id = global_id.x;
    let size = 1u << input.log_size;
    
    let workgroup_chunk_size = (size + workgroup_dispatch - 1u) / workgroup_dispatch;
    let thread_chunk_size = (workgroup_chunk_size + workgroup_size - 1u) / workgroup_size;
    let start_idx = workgroup_id * workgroup_chunk_size + local_id * thread_chunk_size;
    let end_idx = min(start_idx + thread_chunk_size, size);

    for (var i = start_idx; i < end_idx; i = i + 1u) {
        output.values[i] = mod_mul(output.values[i], input.mod_inv);
    }
}
