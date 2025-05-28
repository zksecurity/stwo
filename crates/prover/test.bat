:: run_test.bat
@echo off
setlocal

:: 상대 경로를 절대 경로로 변환
pushd %~dp0
set "DLL_PATH=%CD%\..\..\dll"
popd

:: DLL 경로를 PATH에 추가
set "PATH=%DLL_PATH%;%PATH%"

set RUST_BACKTRACE=1
set RUST_LOG=info,wgpu_core=trace,wgpu_hal=trace
set RUST_MIN_STACK_SIZE=500077216

cargo test --package stwo-prover --lib -- examples::poseidon::tests::test_web_poseidon_prove --exact --show-output -- --nocapture
pause
