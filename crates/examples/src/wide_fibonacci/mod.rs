use itertools::Itertools;
use num_traits::One;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, LogupTraceGenerator, relation, Relation, RelationEntry};

pub type WideFibonacciComponent<const N: usize> = FrameworkComponent<WideFibonacciEval<N>>;

// Define a relation for Fibonacci value lookups: (prev_value, curr_value, next_value)
relation!(FibonacciRelation, 3);

pub struct FibInput {
    pub a: PackedBaseField,
    pub b: PackedBaseField,
}

/// A component that enforces the Fibonacci sequence.
/// Each row contains a seperate Fibonacci sequence of length `N`.
#[derive(Clone)]
pub struct WideFibonacciEval<const N: usize> {
    pub log_n_rows: u32,
    pub fibonacci_relation: FibonacciRelation,
}
impl<const N: usize> FrameworkEval for WideFibonacciEval<N> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        
        // Add regular Fibonacci constraints
        for _ in 2..N {
            let c = eval.next_trace_mask();
            eval.add_constraint(c.clone() - (a.square() + b.square()));
            
            // Add logup relation entry for Fibonacci sequence (a, b, c)
            eval.add_to_relation(RelationEntry::new(
                &self.fibonacci_relation,
                E::EF::one(),
                &[a.clone(), b.clone(), c.clone()],
            ));
            
            a = b;
            b = c;
        }
        
        // Finalize logup
        eval.finalize_logup();
        eval
    }
}

pub struct FibLookupData<const N: usize> {
    pub fibonacci_triplets: Vec<[Col<SimdBackend, BaseField>; 3]>,
}

pub fn generate_trace<const N: usize>(
    log_size: u32,
    inputs: &[FibInput],
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    FibLookupData<N>,
) {
    let mut trace = (0..N)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    
    let mut lookup_data = FibLookupData {
        fibonacci_triplets: Vec::new(),
    };
    
    for (vec_index, input) in inputs.iter().enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].data[vec_index] = a;
        trace[1].data[vec_index] = b;
        
        // Store lookup data for each Fibonacci triplet (a, b, c)
        for i in 2..N {
            let c = a.square() + b.square();
            trace[i].data[vec_index] = c;
            
            // Store the triplet for lookup
            if lookup_data.fibonacci_triplets.len() <= (i - 2) {
                lookup_data.fibonacci_triplets.push([
                    Col::<SimdBackend, BaseField>::zeros(1 << log_size),
                    Col::<SimdBackend, BaseField>::zeros(1 << log_size),
                    Col::<SimdBackend, BaseField>::zeros(1 << log_size),
                ]);
            }
            
            lookup_data.fibonacci_triplets[i - 2][0].data[vec_index] = a;
            lookup_data.fibonacci_triplets[i - 2][1].data[vec_index] = b;
            lookup_data.fibonacci_triplets[i - 2][2].data[vec_index] = c;
            
            a = b;
            b = c;
        }
    }
    
    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec();
    
    (trace, lookup_data)
}

pub fn generate_interaction_trace<const N: usize>(
    log_size: u32,
    lookup_data: FibLookupData<N>,
    fibonacci_relation: &FibonacciRelation,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let mut logup_gen = unsafe { LogupTraceGenerator::uninitialized(log_size) };
    
    // For each Fibonacci triplet, add fractions to logup
    for triplet_cols in lookup_data.fibonacci_triplets {
        let frac_at_row = |vec_row: usize| {
            let values = [
                triplet_cols[0].data[vec_row],
                triplet_cols[1].data[vec_row], 
                triplet_cols[2].data[vec_row],
            ];
            
            let denom: PackedSecureField = fibonacci_relation.combine(
                &values.each_ref().map(|s| *s),
            );
            
            // Return (numerator, denominator) for logup
            (PackedSecureField::one(), denom)
        };
        
        let range = 0..1 << (log_size - LOG_N_LANES);
        
        #[cfg(not(feature = "parallel"))]
        logup_gen.col_from_iter(range.map(frac_at_row));
        
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            logup_gen.col_from_par_iter(range.into_par_iter().map(frac_at_row));
        }
    }
    
    logup_gen.finalize_last()
}

#[cfg(test)]
pub mod tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};
    use crate::wide_fibonacci::FibonacciRelation;
    use stwo::core::air::Component;
    use stwo::core::channel::Blake2sChannel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::channel::Poseidon252Channel;
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::SecureField;
    use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
    use stwo::core::poly::circle::CanonicCoset;
    use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    #[cfg(not(target_arch = "wasm32"))]
    use stwo::core::vcs::poseidon252_merkle::Poseidon252MerkleChannel;
    use stwo::core::verifier::verify;
    use stwo::core::ColumnVec;
    use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::{cpu, Column};
    use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
    use stwo::prover::poly::BitReversedOrder;
    use stwo::prover::{prove, CommitmentSchemeProver};
    use stwo_constraint_framework::{
        assert_constraints_on_polys, AssertEvaluator, FrameworkEval, TraceLocationAllocator,
    };

    use super::WideFibonacciEval;
    use crate::wide_fibonacci::{generate_trace, generate_interaction_trace, FibInput, FibLookupData, WideFibonacciComponent};
    use stwo_constraint_framework::expr::evaluator::ExprEvaluator;
    use stwo_constraint_framework::expr::wgsl_gen::WgslGenerator;

    const FIB_SEQUENCE_LENGTH: usize = 10;

    pub fn generate_test_trace(
        log_n_instances: u32,
    ) -> (
        ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
        FibLookupData<FIB_SEQUENCE_LENGTH>,
    ) {
        if log_n_instances < LOG_N_LANES {
            let n_instances = 1 << log_n_instances;
            let inputs = vec![FibInput {
                a: PackedBaseField::from_array(std::array::from_fn(|j| {
                    if j < n_instances {
                        BaseField::one()
                    } else {
                        BaseField::zero()
                    }
                })),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    if j < n_instances {
                        BaseField::from_u32_unchecked((j) as u32)
                    } else {
                        BaseField::zero()
                    }
                })),
            }];
            return generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs);
        }
        let inputs = (0..(1 << (log_n_instances - LOG_N_LANES)))
            .map(|i| FibInput {
                a: PackedBaseField::one(),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    BaseField::from_u32_unchecked((i * 16 + j) as u32)
                })),
            })
            .collect_vec();
        generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs)
    }

    fn fibonacci_constraint_evaluator<const N: u32>(eval: AssertEvaluator<'_>) {
        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> { 
            log_n_rows: N,
            fibonacci_relation: FibonacciRelation::dummy(),
        }.evaluate(eval);
    }

    #[test]
    fn test_wide_fibonacci_constraints() {
        const LOG_N_INSTANCES: u32 = 6;
        let (trace, lookup_data) = generate_test_trace(LOG_N_INSTANCES);
        
        // Generate interaction trace with lookup elements
        let fibonacci_relation = FibonacciRelation::dummy();
        let (interaction_trace, claimed_sum) = generate_interaction_trace(
            LOG_N_INSTANCES, 
            lookup_data, 
            &fibonacci_relation
        );
        
        let traces = TreeVec::new(vec![vec![], trace, interaction_trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            claimed_sum,
        );
    }

    #[test]
    #[should_panic]
    fn test_wide_fibonacci_constraints_fails() {
        const LOG_N_INSTANCES: u32 = 6;

        let (mut trace, lookup_data) = generate_test_trace(LOG_N_INSTANCES);
        // Modify the trace such that a constraint fail.
        trace[2].values.set(2, BaseField::one());  // Change index from 17 to 2 (valid for 3 columns)
        
        // Generate interaction trace with the modified trace
        let fibonacci_relation = FibonacciRelation::dummy();
        let (interaction_trace, claimed_sum) = generate_interaction_trace(
            LOG_N_INSTANCES, 
            lookup_data, 
            &fibonacci_relation
        );
        
        let traces = TreeVec::new(vec![vec![], trace, interaction_trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            claimed_sum,
        );
    }

    #[test_log::test]
    fn test_wide_fibonacci_constraints_eval() {
        const LOG_N_INSTANCES: u32 = 5;

        let widefib = WideFibonacciEval::<FIB_SEQUENCE_LENGTH> { 
            log_n_rows: LOG_N_INSTANCES,
            fibonacci_relation: FibonacciRelation::dummy(),
        };
        let eval = widefib.evaluate(ExprEvaluator::new());

        // print eval
        println!("=== Wide Fibonacci Constraint Expressions ===");
        println!("{}", eval.format_constraints());
    }

    #[test_log::test]
    fn test_wide_fib_prove_with_blake() {
        for log_n_instances in 5..=5 {
            let config = PcsConfig::default();
            // Precompute twiddles.
            let twiddles = SimdBackend::precompute_twiddles(
                CanonicCoset::new(log_n_instances + 1 + config.fri_config.log_blowup_factor)
                    .circle_domain()
                    .half_coset,
            );

            // Setup protocol.
            let prover_channel = &mut Blake2sChannel::default();
            let mut commitment_scheme =
                CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

            // Preprocessed trace
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals([]);
            tree_builder.commit(prover_channel);

            // Trace.
            let (trace, lookup_data) = generate_test_trace(log_n_instances);

            // want to print trace.values
            println!("=== Trace ===");
            for (i, col) in trace.iter().enumerate() {
                println!("Column[{}]: {:?}", i, col.values);
            }
            println!();

            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(trace);
            tree_builder.commit(prover_channel);

            // Draw lookup elements.
            let fibonacci_relation = FibonacciRelation::draw(prover_channel);

            // Interaction trace.
            let (interaction_trace, claimed_sum) = generate_interaction_trace(
                log_n_instances, 
                lookup_data, 
                &fibonacci_relation
            );
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(interaction_trace);
            tree_builder.commit(prover_channel);

            // Prove constraints.
            let component = WideFibonacciComponent::new(
                &mut TraceLocationAllocator::default(),
                WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                    log_n_rows: log_n_instances,
                    fibonacci_relation,
                },
                claimed_sum,
            );

            let proof = prove::<SimdBackend, Blake2sMerkleChannel>(
                &[&component],
                prover_channel,
                commitment_scheme,
            )
            .unwrap();

            // Verify.
            let verifier_channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = component.trace_log_degree_bounds();
            commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
            commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
            // Draw lookup elements.
            let fibonacci_relation = FibonacciRelation::draw(verifier_channel);
            assert_eq!(fibonacci_relation, component.fibonacci_relation);
            // Interaction columns.
            commitment_scheme.commit(proof.commitments[2], &sizes[2], verifier_channel);
            verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_wide_fib_prove_with_poseidon() {
        const LOG_N_INSTANCES: u32 = 6;
        let config = PcsConfig::default();
        // Precompute twiddles.
        let twiddles = SimdBackend::precompute_twiddles(
            CanonicCoset::new(LOG_N_INSTANCES + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Poseidon252Channel::default();
        let mut commitment_scheme =
            CommitmentSchemeProver::<SimdBackend, Poseidon252MerkleChannel>::new(config, &twiddles);

        // TODO(ilya): remove the following once preproccessed columns are not mandatory.
        // Preprocessed trace
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals([]);
        tree_builder.commit(prover_channel);

        // Trace.
        let (trace, lookup_data) = generate_test_trace(LOG_N_INSTANCES);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        // Draw lookup elements.
        let fibonacci_relation = FibonacciRelation::draw(prover_channel);

        // Interaction trace.
        let (interaction_trace, claimed_sum) = generate_interaction_trace(
            LOG_N_INSTANCES, 
            lookup_data, 
            &fibonacci_relation
        );
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(interaction_trace);
        tree_builder.commit(prover_channel);

        // Prove constraints.
        let component = WideFibonacciComponent::new(
            &mut TraceLocationAllocator::default(),
            WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                log_n_rows: LOG_N_INSTANCES,
                fibonacci_relation,
            },
            claimed_sum,
        );
        let proof = prove::<SimdBackend, Poseidon252MerkleChannel>(
            &[&component],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Poseidon252Channel::default();
        let commitment_scheme =
            &mut CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(proof.config);

        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);
        // Draw lookup elements.
        let fibonacci_relation = FibonacciRelation::draw(verifier_channel);
        assert_eq!(fibonacci_relation, component.fibonacci_relation);
        // Interaction columns.
        commitment_scheme.commit(proof.commitments[2], &sizes[2], verifier_channel);
        verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
    }

    #[test]
    fn test_wide_fibonacci_wgsl_generation_with_logup() {
        const LOG_N_INSTANCES: u32 = 6;
        const SMALL_FIB_SEQUENCE_LENGTH: usize = 3;

        // Create ExprEvaluator and evaluate constraints to get expressions
        let fibonacci_eval = WideFibonacciEval::<SMALL_FIB_SEQUENCE_LENGTH> { 
            log_n_rows: LOG_N_INSTANCES,
            fibonacci_relation: FibonacciRelation::dummy(),
        };
        
        let expr_evaluator = fibonacci_eval.evaluate(ExprEvaluator::new());
        
        // Print constraint expressions (human-readable format)
        println!("=== Fibonacci Constraint Expressions with Logup ===");
        println!("{}", expr_evaluator.format_constraints());
        println!();
        
        // Build IR from the expressions
        let ir_instructions = expr_evaluator.build_ir();
        
        println!("=== IR Instructions ===");
        for (i, instr) in ir_instructions.iter().enumerate() {
            println!("{:02}: {:?}", i, instr);
        }
        println!();
        
        // Generate WGSL code from IR
        let mut wgsl_generator = stwo_constraint_framework::expr::wgsl_gen::DefaultWgslGenerator::new();
        let wgsl_code = wgsl_generator.generate_wgsl(&ir_instructions, true);
        
        println!("=== Generated WGSL Code with Logup ===");
        println!("{}", wgsl_code);
        
        // Check that WGSL code contains logup parameters
        assert!(wgsl_code.contains("claimed_sum: QM31,"));
        assert!(wgsl_code.contains("column_size: u32,"));
        assert!(wgsl_code.contains("@compute"));
        assert!(wgsl_code.contains("fn main"));
    }

    #[test]
    fn test_wide_fibonacci_wgsl_generation() {
        const LOG_N_INSTANCES: u32 = 6;
        const SMALL_FIB_SEQUENCE_LENGTH: usize = 3;

        let (test_trace, _lookup_data) = generate_test_trace(LOG_N_INSTANCES);
        let traces = TreeVec::new(vec![vec![], test_trace]);

        // want to print traces[0]
        println!("=== Traces ===");
        for (i, trace) in traces[1].iter().enumerate() {
            println!("Trace[{}]: {:?}", i, trace);
        }
        println!();

        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints_on_polys(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
            SecureField::zero(),
        );
        
        // Create ExprEvaluator and evaluate constraints to get expressions
        let fibonacci_eval = WideFibonacciEval::<SMALL_FIB_SEQUENCE_LENGTH> { 
            log_n_rows: LOG_N_INSTANCES,
            fibonacci_relation: FibonacciRelation::dummy(),
        };
        
        let expr_evaluator = fibonacci_eval.evaluate(ExprEvaluator::new());
        
        // Print constraint expressions (human-readable format)
        println!("=== Fibonacci Constraint Expressions ===");
        println!("{}", expr_evaluator.format_constraints());
        println!();
        
        // Build IR from the expressions
        let ir_instructions = expr_evaluator.build_ir();
        
        println!("=== IR Instructions ===");
        for (i, instr) in ir_instructions.iter().enumerate() {
            println!("{:02}: {:?}", i, instr);
        }
        println!();
        
        // Generate WGSL code from IR
        let mut wgsl_generator = stwo_constraint_framework::expr::wgsl_gen::DefaultWgslGenerator::new();
        let wgsl_code = wgsl_generator.generate_wgsl(&ir_instructions, true);
        
        println!("=== Generated WGSL Code ===");
        println!("{}", wgsl_code);
        
        // Basic sanity checks that WGSL code contains expected elements
        assert!(wgsl_code.contains("@compute"));
        assert!(wgsl_code.contains("fn main"));
    }
}
