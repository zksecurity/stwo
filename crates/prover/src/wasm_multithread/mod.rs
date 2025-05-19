#[cfg(all(feature = "parallel", target_family = "wasm", not(target_os = "wasi")))]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(all(feature = "parallel", target_family = "wasm", not(target_os = "wasi")))]
fn get_hardware_concurrency() -> usize {
    web_sys::window()
        .map(|w| w.navigator().hardware_concurrency() as usize)
        .unwrap_or_else(|| {
            web_sys::console::warn_1(
                &"navigator.hardwareConcurrency unavailable; defaulting to 1".into(),
            );
            1
        })
}

#[cfg(all(feature = "parallel", target_family = "wasm", not(target_os = "wasi")))]
pub async fn rayon_init_thread_pool() {
    let num_threads = get_hardware_concurrency();
    let promise = init_thread_pool(num_threads);
    if let Err(err) = wasm_bindgen_futures::JsFuture::from(promise).await {
        web_sys::console::error_1(&format!("Failed to start pool: {:?}", err).into());
    } else {
        web_sys::console::info_1(
            &format!("Rayon pool started with {} threads", num_threads).into(),
        );
    }
}

#[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
pub async fn init_wasm_mt() {
    #[cfg(feature = "parallel")]
    rayon_init_thread_pool().await;
}
