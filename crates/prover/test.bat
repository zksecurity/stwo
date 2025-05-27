:: run_test.bat
@echo off
set RUST_BACKTRACE=1
set RUST_LOG=info
set RUST_MIN_STACK_SIZE=500077216
cargo test --package stwo-prover --lib -- examples::poseidon::tests::test_web_poseidon_prove --exact --show-output -- --nocapture
pause
