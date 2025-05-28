:: run_test.bat
@echo off
setlocal

pushd %~dp0
set "DLL_PATH=%CD%\..\..\dll"
popd

set "PATH=%DLL_PATH%;%PATH%"

set RUST_BACKTRACE=1
set RUST_LOG=info,wgpu_core=warn,wgpu_hal=warn
set RUST_MIN_STACK_SIZE=500077216

set RUST_LOG_SPAN_EVENTS=enter,close
set RUSTFLAGS=-C target-cpu=native -C opt-level=3

cargo test --package stwo-prover --release --lib -- examples::poseidon::tests::test_web_poseidon_prove --exact --show-output -- --nocapture
pause
