use std::sync::Arc;

use js_sys::{Atomics, Int32Array};
use web_sys::console;

use super::eval_composition_poly::{compute_composition_polynomial_wgpu, init_wgpu_instance};
use super::{ByteSerialize, ComputeCompositionPolynomialInput};

pub async fn runner_eval_composition_polynomial(
    input_data_sab: &js_sys::SharedArrayBuffer,
    output_data_sab: &js_sys::SharedArrayBuffer,
    receiver_sab: &js_sys::SharedArrayBuffer,
    sender_sab: &js_sys::SharedArrayBuffer,
) {
    let instance = init_wgpu_instance().await;
    console::log_1(&"runner: init_wgpu_instance".into());

    let request_flag = Int32Array::new(sender_sab);
    let response_flag = Int32Array::new(receiver_sab);

    // byte‐level view of the SABs
    let input_view = js_sys::Uint8Array::new(input_data_sab);
    let output_view = js_sys::Uint8Array::new(output_data_sab);

    // preallocate once
    let mut input_buf = vec![0u8; std::mem::size_of::<ComputeCompositionPolynomialInput>()];

    loop {
        console::log_1(&"runner: Atomics.wait on receiver_state".into());
        let outcome = Atomics::wait(&request_flag, 0, 0).unwrap();
        console::log_1(&format!("runner: Atomics.wait returned {:?}", outcome).into());

        // start timer here wasm
        input_view.copy_to(&mut input_buf);
        let input_data = Arc::new(ComputeCompositionPolynomialInput::from_bytes(&input_buf));

        let output_data = compute_composition_polynomial_wgpu(input_data, &instance).await;
        output_view.copy_from(&output_data.as_bytes());

        Atomics::store(&request_flag, 0, 0).unwrap();
        console::log_1(&"runner: request_flag set to 0".into());

        // Reset receiver state and notify sender
        Atomics::store(&response_flag, 0, 1).unwrap();
        console::log_1(&"runner: response_flag set to 1".into());
        Atomics::notify(&response_flag, 0).unwrap();
        console::log_1(&"runner: response_flag notified".into());
    }
}
