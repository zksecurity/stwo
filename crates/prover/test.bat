:: run_test.bat
@echo off
set RUST_BACKTRACE=1
set RUST_LOG=info
cargo test --package stwo-prover --lib -- examples::poseidon::tests::test_simd_poseidon_prove --exact --show-output -- --nocapture
pause
