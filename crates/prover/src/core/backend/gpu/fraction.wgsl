// This shader contains implementations for fraction operations.
// It is stateless and can be used as a library in other shaders.

struct Fraction {
    numerator: QM31,
    denominator: QM31,
}

// Add two fractions: (a/b + c/d) = (ad + bc)/(bd)
fn fraction_add(a: Fraction, b: Fraction) -> Fraction {
    let numerator = qm31_add(
        qm31_mul(a.numerator, b.denominator),
        qm31_mul(b.numerator, a.denominator)
    );
    let denominator = qm31_mul(a.denominator, b.denominator);
    return Fraction(numerator, denominator);
}