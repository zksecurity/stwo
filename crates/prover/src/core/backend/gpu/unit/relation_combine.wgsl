const N_STATE: u32 = 16;
const N_LANES: u32 = 16;

struct LookupElements {
    z: QM31,
    alpha: QM31,
    alpha_powers: array<QM31, N_STATE>,
}

fn combine(values: array<M31, N_STATE>) -> QM31 {
    var result: QM31 = QM31(
        CM31(M31(0u), M31(0u)),
        CM31(M31(0u), M31(0u))
    );

    for (var j: u32 = 0u; j < N_STATE; j = j + 1u) {
        let value_q: QM31 = QM31(
            CM31(values[j], M31(0u)),
            CM31(M31(0u), M31(0u))
        );
        result = qm31_add(result, qm31_mul(input.lookup_elements.alpha_powers[j], value_q));
    }
    
    return qm31_sub(result, input.lookup_elements.z);
}

fn combine_simd(values: array<array<M31, N_LANES>, N_STATE>) -> array<QM31, N_LANES> {
    var results: array<QM31, N_LANES>;

    for (var lane: u32 = 0u; lane < N_LANES; lane = lane + 1u) {
        results[lane] = QM31(
            CM31(M31(0u), M31(0u)),
            CM31(M31(0u), M31(0u))
        );
    }
    
    for (var j: u32 = 0u; j < N_STATE; j = j + 1u) {
        let alpha_power: QM31 = input.lookup_elements.alpha_powers[j];
        for (var lane: u32 = 0u; lane < N_LANES; lane = lane + 1u) {
            let value_q: QM31 = QM31(
                CM31(values[j][lane], M31(0u)),
                CM31(M31(0u), M31(0u))
            );
            results[lane] = qm31_add(results[lane], qm31_mul(alpha_power, value_q));
        }
    }
    
    for (var lane: u32 = 0u; lane < N_LANES; lane = lane + 1u) {
        results[lane] = qm31_sub(results[lane], input.lookup_elements.z);
    }
    
    return results;
}
