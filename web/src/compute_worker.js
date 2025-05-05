import init, { run } from '../pkg/wgpu_examples.js';

// Handle web worker messages
onmessage = async (e) => {
    if (e.data.type === 'init') {
        try {
            // Initialize WebAssembly module
            console.log('Initializing WebGPU compute worker');
            await init();

            const { input_data_sab, output_data_sab, receiver_sab, sender_sab } = e.data;

            // Execute WebGPU work
            console.log('Running WebGPU compute worker');
            await run(input_data_sab, output_data_sab, receiver_sab, sender_sab);
        } catch (error) {
            console.error('Error:', error);
        }
    }
}; 