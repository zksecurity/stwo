use super::evaluator::ExprEvaluator;
use super::ir::IRInstr;
use super::wgsl_gen::{WgslGenerator, DefaultWgslGenerator};
use super::constants::ConstraintConfig;
use std::marker::PhantomData;

/// Main parser that converts constraint expressions to WGSL compute shaders
pub struct WgslParser<C> 
where
    C: ConstraintConfig,
{
    generator: WgslGenerator<C>,
    _phantom: PhantomData<C>,
}

impl<C> WgslParser<C> 
where
    C: ConstraintConfig,
{
    pub fn new() -> Self {
        Self {
            generator: WgslGenerator::new(),
            _phantom: PhantomData,
        }
    }

    /// Convert an ExprEvaluator's constraints to WGSL compute shader code
    pub fn parse_constraints_to_wgsl(&mut self, evaluator: &ExprEvaluator, is_debug: bool) -> String {
        // Build IR from the evaluator's constraints and intermediates
        let ir_instructions = evaluator.build_ir();
        
        // Generate WGSL code from the IR
        self.generator.generate_wgsl(&ir_instructions, !is_debug)
    }

    /// Convert IR instructions directly to WGSL
    pub fn parse_ir_to_wgsl(&mut self, instructions: &[IRInstr], is_debug: bool) -> String {
        self.generator.generate_wgsl(instructions, !is_debug)
    }

    /// Get a reference to the internal generator for advanced usage
    pub fn generator(&mut self) -> &mut WgslGenerator<C> {
        &mut self.generator
    }
}

// Type alias for default configuration
use super::constants::DefaultConfig;
pub type DefaultWgslParser = WgslParser<DefaultConfig>;

impl Default for DefaultWgslParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to convert constraints to WGSL in one call
pub fn constraints_to_wgsl(evaluator: &ExprEvaluator, is_debug: bool) -> String {
    let mut parser = DefaultWgslParser::new();
    parser.parse_constraints_to_wgsl(evaluator, is_debug)
}

/// Convenience function to convert IR to WGSL in one call  
pub fn ir_to_wgsl(instructions: &[IRInstr], is_debug: bool) -> String {
    let mut parser = DefaultWgslParser::new();
    parser.parse_ir_to_wgsl(instructions, is_debug)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{ColumnExpr, ExprEvaluator};
    use crate::{EvalAtRow, FrameworkEval};
    use stwo::core::fields::m31::BaseField;

    pub struct SumTestEval {
        pub log_n_rows: u32,
    }
    impl FrameworkEval for SumTestEval {
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

    #[test]
    fn test_sumeval_to_wgsl() {
        let eval = SumTestEval { log_n_rows: 5 };
        let evaluator = eval.evaluate(ExprEvaluator::new());
        
        // Convert to WGSL
        let wgsl_code = constraints_to_wgsl(&evaluator, true);
        
        println!("Generated WGSL for basic constraint:\n{}", wgsl_code);
    }


    #[test]
    fn test_basic_constraint_to_wgsl() {
        // Create a simple constraint evaluator
        let mut evaluator = ExprEvaluator::new();
        
        // Add a simple constraint: x0 + x1 = 0
        let x0 = evaluator.next_trace_mask();
        let x1 = evaluator.next_trace_mask();
        evaluator.add_constraint(x0 + x1);
        
        // Convert to WGSL
        let wgsl_code = constraints_to_wgsl(&evaluator, true);
        
        println!("Generated WGSL for basic constraint:\n{}", wgsl_code);
    }

    #[test] 
    fn test_constraint_with_intermediate() {
        let mut evaluator = ExprEvaluator::new();
        
        // Create constraint with intermediate: 
        // intermediate = x0 * x1
        // constraint = intermediate - x2 = 0
        let x0 = evaluator.next_trace_mask();
        let x1 = evaluator.next_trace_mask(); 
        let x2 = evaluator.next_trace_mask();
        
        let intermediate = evaluator.add_intermediate(x0.clone() * x1.clone());
        evaluator.add_constraint(intermediate - x2);
        
        let wgsl_code = constraints_to_wgsl(&evaluator, true);
        
        println!("Generated WGSL with intermediate:\n{}", wgsl_code);
    }

    #[test]
    fn test_ir_to_wgsl_direct() {
        use crate::expr::ir::{IRInstr, Reg};
        
        // Test direct IR to WGSL conversion
        let instructions = vec![
            IRInstr::LoadCol { 
                dest: Reg(0), 
                col: ColumnExpr::from((0, 0, 0)) 
            },
            IRInstr::LoadConst { 
                dest: Reg(1), 
                value: BaseField::from(42) 
            },
            IRInstr::Mul { 
                dest: Reg(2), 
                lhs: Reg(0), 
                rhs: Reg(1) 
            },
        ];
        
        let wgsl_code = ir_to_wgsl(&instructions, true);
        
        println!("Generated WGSL from IR:\n{}", wgsl_code);
    }
}