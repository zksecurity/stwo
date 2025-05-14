// This shader contains implementations for fraction operations.
// It is stateless and can be used as a library in other shaders.

const ZERO_FRACTION: Fraction =
    Fraction(vec4<u32>(0u), vec4<u32>(1u, 0u, 0u, 0u));

struct Fraction {
    numerator  : QM31,      // vec4<u32>
    denominator: QM31,
}

// Add two fractions: (a/b + c/d) = (ad + bc)/(bd)
fn fraction_add(x: Fraction, y: Fraction) -> Fraction {
    let num = qm31_add(
        qm31_mul(x.numerator,   y.denominator),
        qm31_mul(y.numerator,   x.denominator)
    );
    let den = qm31_mul(x.denominator, y.denominator);
    return Fraction(num, den);
}

fn fraction_eq(x: Fraction, y: Fraction) -> bool {
    return all(x.numerator   == y.numerator) &&
           all(x.denominator == y.denominator);
}
