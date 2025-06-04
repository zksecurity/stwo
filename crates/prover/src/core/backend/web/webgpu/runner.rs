use std::sync::Arc;

use web_sys::console;

use super::eval_composition_poly::{compute_composition_polynomial_wgpu, init_wgpu_instance};
use super::{ComputeCompositionPolynomialInput, ComputeCompositionPolynomialOutput};

pub async fn runner_eval_composition_polynomial(
    request_rx: flume::Receiver<Arc<ComputeCompositionPolynomialInput>>,
    response_tx: flume::Sender<Arc<ComputeCompositionPolynomialOutput>>,
) {
    let instance = init_wgpu_instance().await;
    console::log_1(&"runner: init_wgpu_instance".into());

    console::log_1(&"runner: running".into());
    let outcome = request_rx.recv_async().await.unwrap();
    console::log_1(&format!("runner: received {:?}", outcome).into());

    console::time_with_label("runner-timer");

    let output_data = compute_composition_polynomial_wgpu(outcome, &instance).await;
    response_tx.send(output_data).unwrap();
    console::time_end_with_label("runner-timer");
}
