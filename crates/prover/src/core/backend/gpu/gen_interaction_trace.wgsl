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

fn combine_simd(values: array<array<M31, N_LANES>, N_LANES>) -> QM31 {
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
