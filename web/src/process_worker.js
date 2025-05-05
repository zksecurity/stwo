import init, { process_data } from '../pkg/wgpu_examples.js';

// Handle messages from main thread
self.onmessage = async (e) => {
    if (e.data.type === 'init') {
        await init();
        console.log('Process worker initialized');
    }
    if (e.data.type === 'process') {
        try {
            const { input_values, input_data_sab, output_data_sab, receiver_sab, sender_sab } = e.data;
            
            console.log('Processing data in process worker');
            console.log(input_values);
            // Process data using WASM
            process_data(
                input_values.input1,
                input_values.input2,
                input_data_sab,
                output_data_sab,
                receiver_sab,
                sender_sab
            );

            // get the output_data_sab
            const output_data_bytes = new Uint32Array(output_data_sab);
            const result = output_data_bytes[0];

            console.log('Process worker finished');
            // e.data.type is 'result'
            // e.data.result is the result
            self.postMessage({ type: 'result', result: result });
        } catch (error) {
            console.error('Process worker error', error);
            self.postMessage({ type: 'error', error: error });
        }
    }
}; 