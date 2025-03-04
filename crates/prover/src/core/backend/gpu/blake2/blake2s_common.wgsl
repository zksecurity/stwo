// Maximum input length (adjust as needed)
const MAX_COLUMN_VALUES: u32 = 256;
const MAX_PREV_LAYER_WORDS: u32 = 1024; // Maximum number of u32 words for previous layer data.

struct Blake2sHash {
    h: array<u32, 8>,
};
