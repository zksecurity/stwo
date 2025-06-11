use web_sys::console;

use super::eval_composition_poly::{compute_composition_polynomial_wgpu, init_wgpu_instance};
use super::{ComputeCompositionPolynomialInput, ComputeCompositionPolynomialOutput};

pub async fn runner_eval_composition_polynomial(
    request_rx: flume::Receiver<Box<ComputeCompositionPolynomialInput>>,
    response_tx: flume::Sender<Box<ComputeCompositionPolynomialOutput>>,
) {
    let instance = init_wgpu_instance().await;
    console::log_1(&"runner: init_wgpu_instance".into());

    console::log_1(&"runner: running".into());
    let input_data = request_rx.recv_async().await.unwrap();

    console::time_with_label("runner-timer");

    let output_data = compute_composition_polynomial_wgpu(input_data, &instance).await;

    response_tx.send(output_data).unwrap();
    console::time_end_with_label("runner-timer");
}
