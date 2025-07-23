use std::collections::HashMap;
use std::fmt::Write;

use super::ir::{IRInstr, Reg, Reg4};
use super::ColumnExpr;

/// WGSL code generator for constraint evaluation
pub struct WgslGenerator {
    /// Generated WGSL compute shader code
    shader_code: String,
    /// Maps register IDs to WGSL variable names
    reg_map: HashMap<usize, String>,
    /// Maps 4-element register IDs to WGSL variable names
    reg4_map: HashMap<usize, String>,
    /// Counter for generating unique variable names
    var_counter: usize,
    /// Column bindings for input data
    column_bindings: HashMap<ColumnExpr, usize>,
    /// Parameter bindings
    param_bindings: HashMap<String, usize>,
    /// Next available binding index
    next_binding: usize,
}

impl WgslGenerator {
    pub fn new() -> Self {
        Self {
            shader_code: String::new(),
            reg_map: HashMap::new(),
            reg4_map: HashMap::new(),
            var_counter: 0,
            column_bindings: HashMap::new(),
            param_bindings: HashMap::new(),
            next_binding: 0,
        }
    }

    /// Generate WGSL code from IR instructions
    pub fn generate_wgsl(&mut self, instructions: &[IRInstr]) -> String {
        self.generate_header();
        self.analyze_bindings(instructions);
        self.generate_bindings();
        self.generate_compute_function();
        self.generate_instructions(instructions);
        self.generate_footer();
        
        self.shader_code.clone()
    }

    fn generate_header(&mut self) {
        writeln!(self.shader_code, "// WGSL Constraint Evaluation Shader").unwrap();
        writeln!(self.shader_code, "// Generated from IR instructions").unwrap();
        writeln!(self.shader_code).unwrap();
    }

    fn analyze_bindings(&mut self, instructions: &[IRInstr]) {
        // Scan instructions to find all columns and parameters that need bindings
        for instr in instructions {
            match instr {
                IRInstr::LoadCol { col, .. } => {
                    if !self.column_bindings.contains_key(col) {
                        self.column_bindings.insert(col.clone(), self.next_binding);
                        self.next_binding += 1;
                    }
                }
                IRInstr::LoadParam { name, .. } | IRInstr::LoadExtParam { name, .. } => {
                    if !self.param_bindings.contains_key(name) {
                        self.param_bindings.insert(name.clone(), self.next_binding);
                        self.next_binding += 1;
                    }
                }
                _ => {}
            }
        }
    }

    fn generate_bindings(&mut self) {
        // Generate buffer bindings for columns
        for (col, binding) in &self.column_bindings {
            writeln!(
                self.shader_code,
                "@group(0) @binding({}) var<storage, read> col_{}_{}_offset_{}: array<f32>;",
                binding, col.interaction, col.idx, col.offset
            ).unwrap();
        }

        // Generate uniform bindings for parameters
        for (param, binding) in &self.param_bindings {
            writeln!(
                self.shader_code,
                "@group(0) @binding({}) var<uniform> param_{}: f32;",
                binding, param.replace(' ', "_")
            ).unwrap();
        }

        // Output buffer for results
        writeln!(
            self.shader_code,
            "@group(0) @binding({}) var<storage, read_write> output: array<f32>;",
            self.next_binding
        ).unwrap();
        
        writeln!(self.shader_code).unwrap();
    }

    fn generate_compute_function(&mut self) {
        writeln!(self.shader_code, "@compute @workgroup_size(64)").unwrap();
        writeln!(self.shader_code, "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{").unwrap();
        writeln!(self.shader_code, "    let index = global_id.x;").unwrap();
        writeln!(self.shader_code).unwrap();
    }

    fn generate_instructions(&mut self, instructions: &[IRInstr]) {
        for instr in instructions {
            self.generate_instruction(instr);
        }
    }

    fn generate_instruction(&mut self, instr: &IRInstr) {
        match instr {
            IRInstr::LoadCol { dest, col } => {
                let var_name = self.get_reg_var(*dest);
                let binding = self.column_bindings[col];
                writeln!(
                    self.shader_code,
                    "    let {} = col_{}_{}_offset_{}[index];",
                    var_name, col.interaction, col.idx, col.offset
                ).unwrap();
            }
            IRInstr::LoadConst { dest, value } => {
                let var_name = self.get_reg_var(*dest);
                writeln!(
                    self.shader_code,
                    "    let {} = {}f;",
                    var_name, value.0
                ).unwrap();
            }
            IRInstr::LoadParam { dest, name } => {
                let var_name = self.get_reg_var(*dest);
                writeln!(
                    self.shader_code,
                    "    let {} = param_{};",
                    var_name, name.replace(' ', "_")
                ).unwrap();
            }
            IRInstr::Add { dest, lhs, rhs } => {
                let dest_var = self.get_reg_var(*dest);
                let lhs_var = self.get_reg_var(*lhs);
                let rhs_var = self.get_reg_var(*rhs);
                writeln!(
                    self.shader_code,
                    "    let {} = {} + {};",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::Sub { dest, lhs, rhs } => {
                let dest_var = self.get_reg_var(*dest);
                let lhs_var = self.get_reg_var(*lhs);
                let rhs_var = self.get_reg_var(*rhs);
                writeln!(
                    self.shader_code,
                    "    let {} = {} - {};",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::Mul { dest, lhs, rhs } => {
                let dest_var = self.get_reg_var(*dest);
                let lhs_var = self.get_reg_var(*lhs);
                let rhs_var = self.get_reg_var(*rhs);
                writeln!(
                    self.shader_code,
                    "    let {} = {} * {};",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::Neg { dest, op } => {
                let dest_var = self.get_reg_var(*dest);
                let op_var = self.get_reg_var(*op);
                writeln!(
                    self.shader_code,
                    "    let {} = -{};",
                    dest_var, op_var
                ).unwrap();
            }
            IRInstr::Inv { dest, op } => {
                let dest_var = self.get_reg_var(*dest);
                let op_var = self.get_reg_var(*op);
                writeln!(
                    self.shader_code,
                    "    let {} = 1.0 / {};",
                    dest_var, op_var
                ).unwrap();
            }
            IRInstr::LoadExtCol { dest, col } => {
                let var_name = self.get_reg4_var(*dest);
                let col0_var = self.get_reg_var(col[0]);
                let col1_var = self.get_reg_var(col[1]);
                let col2_var = self.get_reg_var(col[2]);
                let col3_var = self.get_reg_var(col[3]);
                writeln!(
                    self.shader_code,
                    "    let {} = vec4<f32>({}, {}, {}, {});",
                    var_name, col0_var, col1_var, col2_var, col3_var
                ).unwrap();
            }
            IRInstr::LoadExtConst { dest, value } => {
                let var_name = self.get_reg4_var(*dest);
                writeln!(
                    self.shader_code,
                    "    let {} = vec4<f32>({}f, {}f, {}f, {}f);",
                    var_name, value.0.0.0, value.0.1.0, value.1.0.0, value.1.1.0
                ).unwrap();
            }
            IRInstr::LoadExtParam { dest, name } => {
                let var_name = self.get_reg4_var(*dest);
                // For now, assume extension parameters are stored as vec4
                writeln!(
                    self.shader_code,
                    "    let {} = vec4<f32>(param_{}, 0.0, 0.0, 0.0);",
                    var_name, name.replace(' ', "_")
                ).unwrap();
            }
            IRInstr::AddExt { dest, lhs, rhs } => {
                let dest_var = self.get_reg4_var(*dest);
                let lhs_var = self.get_reg4_var(*lhs);
                let rhs_var = self.get_reg4_var(*rhs);
                writeln!(
                    self.shader_code,
                    "    let {} = {} + {};",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::SubExt { dest, lhs, rhs } => {
                let dest_var = self.get_reg4_var(*dest);
                let lhs_var = self.get_reg4_var(*lhs);
                let rhs_var = self.get_reg4_var(*rhs);
                writeln!(
                    self.shader_code,
                    "    let {} = {} - {};",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::MulExt { dest, lhs, rhs } => {
                let dest_var = self.get_reg4_var(*dest);
                let lhs_var = self.get_reg4_var(*lhs);
                let rhs_var = self.get_reg4_var(*rhs);
                // Simplified multiplication for vec4 (not true extension field multiplication)
                writeln!(
                    self.shader_code,
                    "    let {} = {} * {};",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::NegExt { dest, op } => {
                let dest_var = self.get_reg4_var(*dest);
                let op_var = self.get_reg4_var(*op);
                writeln!(
                    self.shader_code,
                    "    let {} = -{};",
                    dest_var, op_var
                ).unwrap();
            }
        }
    }

    fn generate_footer(&mut self) {
        writeln!(self.shader_code, "}}").unwrap();
    }

    fn get_reg_var(&mut self, reg: Reg) -> String {
        if let Some(var_name) = self.reg_map.get(&reg.0) {
            var_name.clone()
        } else {
            let var_name = format!("r{}", self.var_counter);
            self.var_counter += 1;
            self.reg_map.insert(reg.0, var_name.clone());
            var_name
        }
    }

    fn get_reg4_var(&mut self, reg: Reg4) -> String {
        if let Some(var_name) = self.reg4_map.get(&reg.0) {
            var_name.clone()
        } else {
            let var_name = format!("r4_{}", self.var_counter);
            self.var_counter += 1;
            self.reg4_map.insert(reg.0, var_name.clone());
            var_name
        }
    }
}

impl Default for WgslGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::ir::{IRBuilder, IRInstr};
    use crate::expr::{BaseExpr, ColumnExpr};
    use stwo::core::fields::m31::BaseField;

    #[test]
    fn test_simple_wgsl_generation() {
        let mut generator = WgslGenerator::new();
        
        // Simple test: r0 = col(0,0,0) + 5
        let instructions = vec![
            IRInstr::LoadCol { 
                dest: super::super::ir::Reg(0), 
                col: ColumnExpr::from((0, 0, 0)) 
            },
            IRInstr::LoadConst { 
                dest: super::super::ir::Reg(1), 
                value: BaseField::from(5) 
            },
            IRInstr::Add { 
                dest: super::super::ir::Reg(2), 
                lhs: super::super::ir::Reg(0), 
                rhs: super::super::ir::Reg(1) 
            },
        ];

        let wgsl_code = generator.generate_wgsl(&instructions);
        
        // Check that the generated code contains expected elements
        assert!(wgsl_code.contains("@compute @workgroup_size(64)"));
        assert!(wgsl_code.contains("fn main"));
        assert!(wgsl_code.contains("col_0_0_offset_0"));
        assert!(wgsl_code.contains("5f"));
        assert!(wgsl_code.contains(" + "));
        
        println!("Generated WGSL:\n{}", wgsl_code);
    }
}