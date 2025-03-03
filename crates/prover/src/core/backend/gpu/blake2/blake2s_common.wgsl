
struct Blake2sCore {
    h: array<u32, 8>,
    // XX : should be u64?
    t: u32,
}

struct Blake2sBuffer {
    buffer: array<u8, 32>,
    pos: u32,
}

struct Blake2s {
    core: Blake2sCore,
    buffer: Blake2sBuffer,
}

// // finalize_variable_core
// fn finalize_variable_core(
//     &mut self,
//     buffer: &mut Buffer<Self>,
//     out: &mut Output<Self>,
// ) {
//     self.t += buffer.get_pos() as u64;
//     let block = buffer.pad_with_zeros();
//     self.finalize_with_flag(block, 0, out);
// }

// fn finalize_with_flag(
//     &mut self,
//     final_block: &GenericArray<u8, $block_size>,
//     flag: $word,
//     out: &mut Output<Self>,
// ) {
//     self.compress(final_block, !0, flag);
//     let buf = [self.h[0].to_le(), self.h[1].to_le()];
//     out.copy_from_slice(buf.as_bytes())
// }
