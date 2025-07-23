use super::evaluator::ExprEvaluator;
use super::ir::IRInstr;
use super::wgsl_gen::WgslGenerator;

/// Main parser that converts constraint expressions to WGSL compute shaders
pub struct WgslParser {
    generator: WgslGenerator,
}

impl WgslParser {
    pub fn new() -> Self {
        Self {
            generator: WgslGenerator::new(),
        }
    }

    /// Convert an ExprEvaluator's constraints to WGSL compute shader code
    pub fn parse_constraints_to_wgsl(&mut self, evaluator: &ExprEvaluator) -> String {
        // Build IR from the evaluator's constraints and intermediates
        let ir_instructions = evaluator.build_ir();
        
        // Generate WGSL code from the IR
        self.generator.generate_wgsl(&ir_instructions)
    }

    /// Convert IR instructions directly to WGSL
    pub fn parse_ir_to_wgsl(&mut self, instructions: &[IRInstr]) -> String {
        self.generator.generate_wgsl(instructions)
    }

    /// Get a reference to the internal generator for advanced usage
    pub fn generator(&mut self) -> &mut WgslGenerator {
        &mut self.generator
    }
}

impl Default for WgslParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to convert constraints to WGSL in one call
pub fn constraints_to_wgsl(evaluator: &ExprEvaluator) -> String {
    let mut parser = WgslParser::new();
    parser.parse_constraints_to_wgsl(evaluator)
}

/// Convenience function to convert IR to WGSL in one call  
pub fn ir_to_wgsl(instructions: &[IRInstr]) -> String {
    let mut parser = WgslParser::new();
    parser.parse_ir_to_wgsl(instructions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{BaseExpr, ColumnExpr, ExprEvaluator};
    use crate::{EvalAtRow, FrameworkEval};
    use stwo::core::fields::m31::BaseField;

    #[test]
    fn test_basic_constraint_to_wgsl() {
        // Create a simple constraint evaluator
        let mut evaluator = ExprEvaluator::new();
        
        // Add a simple constraint: x0 + x1 = 0
        let x0 = evaluator.next_trace_mask();
        let x1 = evaluator.next_trace_mask();
        evaluator.add_constraint(x0 + x1);
        
        // Convert to WGSL
        let wgsl_code = constraints_to_wgsl(&evaluator);
        
        // Verify the generated code contains expected elements
        assert!(wgsl_code.contains("@compute"));
        assert!(wgsl_code.contains("fn main"));
        assert!(wgsl_code.contains("col_"));
        
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
        
        let wgsl_code = constraints_to_wgsl(&evaluator);
        
        // Should contain multiplication and subtraction
        assert!(wgsl_code.contains(" * "));
        assert!(wgsl_code.contains(" - "));
        assert!(wgsl_code.contains("col_"));
        
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
        
        let wgsl_code = ir_to_wgsl(&instructions);
        
        assert!(wgsl_code.contains("42f"));
        assert!(wgsl_code.contains(" * "));
        assert!(wgsl_code.contains("col_0_0_offset_0"));
        
        println!("Generated WGSL from IR:\n{}", wgsl_code);
    }
}