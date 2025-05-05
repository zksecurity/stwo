import init, { test_poseidon_web } from '../pkg/stwo_prover.js';

// Handle messages from main thread
self.onmessage = async (e) => {
    if (e.data.type === 'init') {
        await init();
        console.log('Process worker initialized');
    }
    if (e.data.type === 'process') {
        try {
            const { input_data_sab, output_data_sab, receiver_sab, sender_sab } = e.data;
            
            console.log('Processing data in process worker');
            test_poseidon_web(
                input_data_sab,
                output_data_sab,
                receiver_sab,
                sender_sab
            );

            console.log('Process worker finished');
            self.postMessage({ type: 'result', result: 0 });
        } catch (error) {
            console.error('Process worker error', error);
            self.postMessage({ type: 'error', error: error });
        }
    }
}; 