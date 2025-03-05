
const MAX_COLUMN_LENGTH: u32 = 256;
const MAX_COLUMNS: u32 = 256;
// Maximum number of u32 words for previous layer data.
const MAX_PREV_LAYER_WORDS: u32 = 1024; 

struct Blake2sHash {
    h: array<u32, 8>,
};

struct Column {
    column: array<u32, MAX_COLUMN_LENGTH>,
}