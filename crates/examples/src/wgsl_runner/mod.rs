use std::mem;

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;
use stwo::core::fields::FieldExpOps;
use stwo::prover::backend::Column;
use stwo_constraint_framework::expr::gpu_common::{ByteSerialize, GpuComputeInstance};
use stwo_constraint_framework::expr::gpu_types::*;
use stwo_constraint_framework::expr::qm31::{GpuM31, GpuQM31};
use stwo_constraint_framework::expr::wgsl_parser::*;
use stwo_constraint_framework::expr::evaluator::ExprEvaluator;
use stwo_constraint_framework::{EvalAtRow, FrameworkEval};

pub struct SumEvalExample {
    pub log_n_rows: u32,
}

impl FrameworkEval for SumEvalExample {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let x0 = eval.next_trace_mask();
        let x1 = eval.next_trace_mask();
        let x2 = eval.next_trace_mask();
        eval.add_constraint(x2 - x0 - x1);
        eval
    }
}

pub struct FiveFibonacciEval;
impl FrameworkEval for FiveFibonacciEval {
    fn log_size(&self) -> u32 {
        5
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        6
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        for _ in 2..3 {
            let c = eval.next_trace_mask();
            eval.add_constraint(c.clone() - (a.square() + b.square()));
            a = b;
            b = c;
        }
        eval
    }
}

/// A wrapper struct that is local to this crate to avoid orphan rule issues
#[derive(Clone)]
pub struct LocalInput(pub ComputeCompositionPolynomialInput);

#[derive(Clone)]  
pub struct LocalOutput(pub ComputeCompositionPolynomialOutput);

impl ByteSerialize for LocalInput {}
impl ByteSerialize for LocalOutput {}

pub struct WgslComputeRunner {
    shader_source: String,
}

impl WgslComputeRunner {
    pub fn new_from_evaluator(evaluator: &ExprEvaluator) -> Self {
        // Generate WGSL code from the evaluator with proper dimensions
        let mut parser = WgslParser::new();
        parser.generator().set_dimensions(
            stwo_constraint_framework::expr::constants::N_ROWS,
            stwo_constraint_framework::expr::constants::N_CONSTRAINTS
        );
        let shader_source = parser.parse_constraints_to_wgsl(evaluator, false);
        println!("Generated WGSL shader:\n{}", shader_source);
        
        Self { shader_source }
    }
    
    pub async fn run_with_parameters(
        &self, 
        random_coeff_powers: &[QM31], 
        denom_inv: &[M31]
    ) -> LocalOutput {
        println!("=== Running WGSL Computation ===");
        
        // Set up input data with the provided parameters
        println!("Random Coeff Powers: {:?}", random_coeff_powers);
        println!("Denom Inv: {:?}", denom_inv);
        
        // Create input data structure
        let input_data = LocalInput(ComputeCompositionPolynomialInput {
            extended_trace: [GpuExtendedColumn { 
                data: [GpuM31(0); stwo_constraint_framework::expr::constants::N_EXTENDED_ROWS as usize] 
            }; stwo_constraint_framework::expr::constants::N_COLUMNS as usize],
            denom_inv: [
                GpuM31(denom_inv[0].into()), 
                GpuM31(denom_inv[1].into()),
                GpuM31(0), 
                GpuM31(0)
            ],
            random_coeff_powers: [
                GpuQM31::from(random_coeff_powers[0]); 
                stwo_constraint_framework::expr::constants::N_CONSTRAINTS as usize
            ],
        });

        // Fill extended_trace with some example data (Sum constraint)
        let mut input_data_mut = input_data;
        for col_idx in 0..stwo_constraint_framework::expr::constants::N_COLUMNS as usize {
            for row_idx in 0..stwo_constraint_framework::expr::constants::N_EXTENDED_ROWS as usize {
                let value = match col_idx {
                    0 => (row_idx + 1) as u32,           // x0
                    1 => ((row_idx * 2) + 1) as u32,     // x1
                    2 => ((row_idx + 1) + (row_idx * 2) + 1) as u32, // x2 = x0 + x1
                    _ => 0
                };
                input_data_mut.0.extended_trace[col_idx].data[row_idx] = GpuM31(value);
            }
        }

        let output_size = mem::size_of::<ComputeCompositionPolynomialOutput>();
        let instance = GpuComputeInstance::new(&input_data_mut, output_size).await;
        
        let (pipeline, bind_group) = instance.create_pipeline(&self.shader_source, "main");
        let workgroup_count = (1, 1, 1); // Single workgroup for simplicity
        
        let result: LocalOutput = instance
            .run_computation(&pipeline, &bind_group, workgroup_count)
            .await;
            
        println!("WGSL computation completed successfully!");
        result
    }

    pub async fn run_with_fibonacci_trace(
        &self, 
        random_coeff_powers: &[QM31], 
        denom_inv: &[M31]
    ) -> LocalOutput {
        println!("=== Running Fibonacci WGSL Computation ===");
        
        // Set up input data with the provided parameters
        println!("Random Coeff Powers: {:?}", random_coeff_powers);
        println!("Denom Inv: {:?}", denom_inv);
        
        // Create input data structure
        let input_data = LocalInput(ComputeCompositionPolynomialInput {
            extended_trace: [GpuExtendedColumn { 
                data: [GpuM31(0); stwo_constraint_framework::expr::constants::N_EXTENDED_ROWS as usize] 
            }; stwo_constraint_framework::expr::constants::N_COLUMNS as usize],
            denom_inv: [
                GpuM31(denom_inv[0].into()), 
                GpuM31(denom_inv[1].into()),
                GpuM31(0), 
                GpuM31(0)
            ],
            random_coeff_powers: [
                GpuQM31::from(random_coeff_powers[0]); 
                stwo_constraint_framework::expr::constants::N_CONSTRAINTS as usize
            ],
        });

        // Manually create the trace with the provided values for 64 rows
        let mut input_data_mut = input_data;
        
        // Column 0: All 1s (64 rows)
        let col0_data = vec![1u32; 64];
        
        // Column 1: The provided packed values
        let col1_packed_values = [
            [1830472794, 1830505562, 1622241907, 1622274675, 336576155, 336608923, 1544545479, 1544578247, 926722771, 926755539, 933835206, 933867974, 730652753, 730685521, 1863173674, 1863206442],
            [5706326, 5739094, 978939810, 978972578, 511558018, 511590786, 1128418600, 1128451368, 1685264856, 1685297624, 685590168, 685622936, 566490957, 566523725, 1829417806, 1829450574],
            [588803054, 588770286, 1681201769, 1681169001, 522611957, 522579189, 1878253168, 1878220400, 1789121466, 1789088698, 593800358, 593767590, 538132102, 538099334, 505497029, 505464261],
            [1516327261, 1516294493, 1434540735, 1434507967, 1976487899, 1976455131, 155776196, 155743428, 1780538011, 1780505243, 191600178, 191567410, 506735182, 506702414, 1520705203, 1520672435]
        ];
        
        // Column 2: The provided packed values
        let col2_packed_values = [
            [634803516, 155848103, 101122856, 177485162, 1699433438, 69804761, 1967031064, 15262190, 238954811, 832834962, 2115017394, 59942817, 1741678288, 1404236793, 238895790, 572487654],
            [1411571672, 517735650, 1029503368, 148134178, 1775518817, 1785122543, 225145177, 394898774, 1057307543, 1146746806, 549516318, 1961010150, 1716924779, 1831331694, 669323105, 1820151333],
            [1165023483, 1227620794, 1833990019, 2072757058, 1777854982, 1864570851, 1159878479, 1756707927, 440477032, 21563038, 1520081105, 370762331, 1119639520, 1170252410, 286516578, 1060450876],
            [701889098, 702604925, 593433666, 242026594, 174876388, 863572976, 420912473, 1009540337, 35596828, 612159206, 683825277, 1700184476, 1391337142, 1502681010, 1735185147, 994293127]
        ];
        
        // Fill the trace data
        // Column 0: All 1s
        for row_idx in 0..64.min(stwo_constraint_framework::expr::constants::N_EXTENDED_ROWS as usize) {
            input_data_mut.0.extended_trace[0].data[row_idx] = GpuM31(col0_data[row_idx]);
        }
        
        // Column 1: Unpack the provided values
        for (packed_idx, packed_row) in col1_packed_values.iter().enumerate() {
            for (val_idx, &value) in packed_row.iter().enumerate() {
                let row_idx = packed_idx * 16 + val_idx;
                if row_idx < 64.min(stwo_constraint_framework::expr::constants::N_EXTENDED_ROWS as usize) {
                    input_data_mut.0.extended_trace[1].data[row_idx] = GpuM31(value);
                }
            }
        }
        
        // Column 2: Unpack the provided values
        for (packed_idx, packed_row) in col2_packed_values.iter().enumerate() {
            for (val_idx, &value) in packed_row.iter().enumerate() {
                let row_idx = packed_idx * 16 + val_idx;
                if row_idx < 64.min(stwo_constraint_framework::expr::constants::N_EXTENDED_ROWS as usize) {
                    input_data_mut.0.extended_trace[2].data[row_idx] = GpuM31(value);
                }
            }
        }

        println!("Created manual trace with 64 rows for Circle STARK");
        println!("Input data for WGSL computation: {:?}", input_data_mut.0.extended_trace);

        let output_size = mem::size_of::<ComputeCompositionPolynomialOutput>();
        let instance = GpuComputeInstance::new(&input_data_mut, output_size).await;
        
        let (pipeline, bind_group) = instance.create_pipeline(&self.shader_source, "main");
        let workgroup_count = (1, 1, 1);
        
        let result: LocalOutput = instance
            .run_computation(&pipeline, &bind_group, workgroup_count)
            .await;
            
        println!("Fibonacci WGSL computation completed successfully!");
        result
    }
}

pub async fn run_wgsl_example() {
    println!("=== WGSL Runner Example ===");
    
    // Create the sum evaluation example
    let eval = SumEvalExample { log_n_rows: 5 };
    let evaluator = eval.evaluate(ExprEvaluator::new());
    
    // Create WGSL runner
    let runner = WgslComputeRunner::new_from_evaluator(&evaluator);
    
    // Set up input data with the provided parameters
    let random_coeff_powers = vec![QM31::from_u32_unchecked(1, 0, 0, 0)]; // (1 + 0i) + (0 + 0i)u
    let denom_inv = vec![M31::from(65536), M31::from(2147418111)];
    
    // Run the computation
    let result = runner.run_with_parameters(&random_coeff_powers, &denom_inv).await;
    
    println!("Result computed! Output has {} polynomial lanes", 
             result.0.poly.len());
    
    println!("Example completed successfully!");
}

pub async fn run_five_fibonacci_wgsl_example() {
    println!("=== Five Fibonacci WGSL Runner Example ===");
    
    // Create the Five Fibonacci evaluation example
    let eval = FiveFibonacciEval;
    let evaluator = eval.evaluate(ExprEvaluator::new());
    
    // Create WGSL runner
    let runner = WgslComputeRunner::new_from_evaluator(&evaluator);
    
    // Set up input data with the provided parameters
    let random_coeff_powers = vec![QM31::from_u32_unchecked(1, 0, 0, 0)]; // (1 + 0i) + (0 + 0i)u
    let denom_inv = vec![M31::from(65536), M31::from(2147418111)];
    
    // Run the computation with actual Fibonacci trace data
    let result = runner.run_with_fibonacci_trace(&random_coeff_powers, &denom_inv).await;
    
    println!("Five Fibonacci result computed! Output has {} polynomial lanes", 
             result.0.poly.len());

    // Print the output polynomial
    for (i, row) in result.0.poly.iter().enumerate() {
        println!("Row {}: {:?}", i, row); 
    }
    
    println!("Five Fibonacci example completed successfully!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wgsl_runner() {
        run_wgsl_example().await;
    }

    #[tokio::test]
    async fn test_five_fibonacci_wgsl_runner() {
        run_five_fibonacci_wgsl_example().await;
    }
}