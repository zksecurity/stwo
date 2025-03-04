//! A reference implementation of the BLAKE2s compression function, in pure Rust.
//! Based on <https://github.com/oconnor663/blake2_simd/blob/master/blake2s/src/avx2.rs>.
// Blake2s constants and internal functions

// Blake2s IV
const BLAKE2S_IV: array<u32, 8> = array<u32, 8>(
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
);

// Blake2s Sigma - permutation constants for 10 rounds, 16 values each (total 160 values)
const SIGMA: array<u32, 160> = array<u32, 160>(
    // Round 0
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u,
    // Round 1
    14u, 10u, 4u, 8u, 9u, 15u, 13u, 6u, 1u, 12u, 0u, 2u, 11u, 7u, 5u, 3u,
    // Round 2
    11u, 8u, 12u, 0u, 5u, 2u, 15u, 13u, 10u, 14u, 3u, 6u, 7u, 1u, 9u, 4u,
    // Round 3
    7u, 9u, 3u, 1u, 13u, 12u, 11u, 14u, 2u, 6u, 5u, 10u, 4u, 0u, 15u, 8u,
    // Round 4
    9u, 0u, 5u, 7u, 2u, 4u, 10u, 15u, 14u, 1u, 11u, 12u, 6u, 8u, 3u, 13u,
    // Round 5
    2u, 12u, 6u, 10u, 0u, 11u, 8u, 3u, 4u, 13u, 7u, 5u, 15u, 14u, 1u, 9u,
    // Round 6
    12u, 5u, 1u, 15u, 14u, 13u, 4u, 10u, 0u, 7u, 6u, 3u, 9u, 2u, 8u, 11u,
    // Round 7
    13u, 11u, 7u, 14u, 12u, 1u, 3u, 9u, 5u, 0u, 15u, 4u, 8u, 6u, 2u, 10u,
    // Round 8
    6u, 15u, 14u, 9u, 11u, 3u, 0u, 8u, 12u, 2u, 13u, 7u, 1u, 4u, 10u, 5u,
    // Round 9
    10u, 2u, 8u, 4u, 7u, 6u, 1u, 5u, 15u, 11u, 9u, 14u, 3u, 12u, 13u, 0u
);

// Right rotation function for 32-bit integers
fn rotr(x: u32, n: u32) -> u32 {
  return (x >> n) | (x << (32u - n));
}

// Blake2s mixing function G: mixes four state words with two message words
fn G(v: ptr<function, array<u32, 16>>, a: u32, b: u32, c: u32, d: u32, x: u32, y: u32) {
  (*v)[a] = (*v)[a] + (*v)[b] + x;
  (*v)[d] = rotr((*v)[d] ^ (*v)[a], 16u);
  (*v)[c] = (*v)[c] + (*v)[d];
  (*v)[b] = rotr((*v)[b] ^ (*v)[c], 12u);
  (*v)[a] = (*v)[a] + (*v)[b] + y;
  (*v)[d] = rotr((*v)[d] ^ (*v)[a], 8u);
  (*v)[c] = (*v)[c] + (*v)[d];
  (*v)[b] = rotr((*v)[b] ^ (*v)[c], 7u);
}

// Blake2s compression function
// state: current hash state (8 u32 words)
// block: message block of 16 u32 words (64 bytes)
// t0, t1: counter values; f0, f1: flags (set to 0 in our usage)
fn compress(
  state: array<u32, 8>,
  block: array<u32, 16>,
  t0: u32, t1: u32,
  f0: u32, f1: u32
) -> array<u32, 8> {
  var v: array<u32, 16>;
  // Initialize v[0..7] with the state and v[8..15] with the IV constants.
  for (var i = 0u; i < 8u; i = i + 1u) {
    v[i] = state[i];
    v[i + 8u] = BLAKE2S_IV[i];
  }
  // Apply counter and flag (both are 0 in our case)
  v[12] = v[12] ^ t0;
  v[13] = v[13] ^ t1;
  v[14] = v[14] ^ f0;
  v[15] = v[15] ^ f1;

  // Execute 10 rounds of the compression function
  for (var r = 0u; r < 10u; r = r + 1u) {
    let s = r * 16u;
    // Column step
    G(&v, 0u, 4u, 8u, 12u, block[SIGMA[s + 0u]], block[SIGMA[s + 1u]]);
    G(&v, 1u, 5u, 9u, 13u, block[SIGMA[s + 2u]], block[SIGMA[s + 3u]]);
    G(&v, 2u, 6u, 10u, 14u, block[SIGMA[s + 4u]], block[SIGMA[s + 5u]]);
    G(&v, 3u, 7u, 11u, 15u, block[SIGMA[s + 6u]], block[SIGMA[s + 7u]]);
    // Diagonal step
    G(&v, 0u, 5u, 10u, 15u, block[SIGMA[s + 8u]], block[SIGMA[s + 9u]]);
    G(&v, 1u, 6u, 11u, 12u, block[SIGMA[s + 10u]], block[SIGMA[s + 11u]]);
    G(&v, 2u, 7u, 8u, 13u, block[SIGMA[s + 12u]], block[SIGMA[s + 13u]]);
    G(&v, 3u, 4u, 9u, 14u, block[SIGMA[s + 14u]], block[SIGMA[s + 15u]]);
  }

  var newState: array<u32, 8>;
  // Finalize by XORing the original state with parts of v
  for (var i = 0u; i < 8u; i = i + 1u) {
    newState[i] = state[i] ^ v[i] ^ v[i + 8u];
  }
  return newState;
}

// Maximum input length (adjust as needed)
const MAX_COLUMN_VALUES: u32 = 256;

// hash_node function
// - children_hashes_present: 1 if child nodes are provided, 0 otherwise
// - left, right: child hashes (each an array of 8 u32 words)
// - column_values: input data array (e.g., column values)
// - column_values_len: actual number of input values
//
// This function mimics the behavior of Blake2sMerkleHasher as follows:
// 1. Initialize the state to all zeros.
// 2. If child hashes are provided, perform an initial compress using them.
// 3. Pad column_values to a multiple of 16 words and process each 16-word block using compress.
fn hash_node(
  children_hashes_present: u32,
  left: array<u32, 8>,
  right: array<u32, 8>,
  column_values: ptr<function, array<u32, MAX_COLUMN_VALUES>>,
  column_values_len: u32
) -> array<u32, 8> {
  var state: array<u32, 8>;
  // Initialize state to 0
  for (var i = 0u; i < 8u; i = i + 1u) {
    state[i] = 0u;
  }

  // If child hashes exist, perform an initial compress with them
  if (children_hashes_present != 0u) {
    var children: array<u32, 16>;
    for (var i = 0u; i < 8u; i = i + 1u) {
      children[i] = left[i];
      children[i + 8u] = right[i];
    }
    state = compress(state, children, 0u, 0u, 0u, 0u);
  }

  // Pad column_values so that the total length is a multiple of 16.
  // In the Rust implementation, rem = 15 - ((len + 15) % 16)
  // Here, we compute the number of padding words (between 0 and 15).
  let rem = (15u - ((column_values_len + 15u) % 16u)) % 16u;
  let total = column_values_len + rem;

  var offset = 0u;
  // Process the padded input in 16-word blocks
  while (offset < total) {
    var block: array<u32, 16>;
    for (var j = 0u; j < 16u; j = j + 1u) {
      if (offset + j < column_values_len) {
        block[j] = (*column_values)[offset + j];
      } else {
        block[j] = 0u;
      }
    }
    state = compress(state, block, 0u, 0u, 0u, 0u);
    offset = offset + 16u;
  }
  return state;
}
