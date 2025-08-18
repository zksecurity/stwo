use std::collections::HashMap;
use std::fmt::Write;
use std::marker::PhantomData;

use super::ir::{IRInstr, Reg, Reg4};
use super::constants::ConstraintConfig;

/// WGSL code generator for constraint evaluation
pub struct WgslGenerator<C> 
where
    C: ConstraintConfig,
{
    /// Generated WGSL compute shader code
    shader_code: String,
    /// Maps register IDs to WGSL variable names
    reg_map: HashMap<usize, String>,
    /// Maps 4-element register IDs to WGSL variable names
    reg4_map: HashMap<usize, String>,
    /// Counter for generating unique base field variable names
    reg_counter: usize,
    /// Counter for generating unique extension field variable names
    reg4_counter: usize,
    /// Parameter bindings
    param_bindings: HashMap<String, usize>,
    /// Next available binding index
    next_binding: usize,
    /// Counter for constraint indices
    constraint_index: usize,
    /// Maps intermediate names to indices for storage
    intermediate_map: HashMap<String, usize>,
    /// Maps extension intermediate names to indices for storage
    ext_intermediate_map: HashMap<String, usize>,
    /// Counter for intermediate indices
    intermediate_counter: usize,
    /// Counter for extension intermediate indices
    ext_intermediate_counter: usize,
    /// Phantom data for config type
    _phantom: PhantomData<C>,
}

impl<C> WgslGenerator<C> 
where
    C: ConstraintConfig,
{
    pub fn new() -> Self {
        Self {
            shader_code: String::new(),
            reg_map: HashMap::new(),
            reg4_map: HashMap::new(),
            reg_counter: 0,
            reg4_counter: 0,
            param_bindings: HashMap::new(),
            next_binding: 0,
            constraint_index: 0,
            intermediate_map: HashMap::new(),
            ext_intermediate_map: HashMap::new(),
            intermediate_counter: 0,
            ext_intermediate_counter: 0,
            _phantom: PhantomData,
        }
    }

    /// Generate WGSL code from IR instructions
    pub fn generate_wgsl(&mut self, instructions: &[IRInstr], header: bool) -> String {
        if header {
            self.generate_header();
        }
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
        
        // Include the qm31.wgsl library with template substitution
        let qm31_code = include_str!("qm31.wgsl");
        let substituted_code = qm31_code
            .replace("${N_ROWS}", &C::N_ROWS.to_string())
            .replace("${N_CONSTRAINTS}", &C::N_CONSTRAINTS.to_string())
            .replace("${N_COLUMNS}", &C::N_COLUMNS.to_string())
            .replace("${N_INTERMEDIATES}", &C::N_INTERMEDIATES.to_string())
            .replace("${N_EXT_INTERMEDIATES}", &C::N_EXT_INTERMEDIATES.to_string())
            .replace("${N_LOOKUP_ELEMENTS}", &C::N_LOOKUP_ELEMENTS.to_string());
        writeln!(self.shader_code, "{}", substituted_code).unwrap();
        writeln!(self.shader_code).unwrap();
    }

    fn analyze_bindings(&mut self, instructions: &[IRInstr]) {
        // Scan instructions to find all interactions and parameters that need bindings
        for instr in instructions {
            match instr {
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
        // Generate all the required struct definitions from the working version
        writeln!(self.shader_code, "struct Extended1DColumn {{").unwrap();
        writeln!(self.shader_code, "    data: array<M31, N_EXTENDED_ROWS>,").unwrap();
        writeln!(self.shader_code, "}}").unwrap();
        writeln!(self.shader_code).unwrap();

        writeln!(self.shader_code, "struct ComputeCompositionPolynomialInput {{").unwrap();
        writeln!(self.shader_code, "    extended_trace: array<Extended1DColumn, N_COLUMNS>,").unwrap();
        writeln!(self.shader_code, "    denom_inv: array<M31, 4>,").unwrap();
        writeln!(self.shader_code, "    random_coeff_powers: array<QM31, N_CONSTRAINTS>,").unwrap();
        writeln!(self.shader_code, "    claimed_sum: QM31,").unwrap();
        writeln!(self.shader_code, "    column_size: u32,").unwrap();
        writeln!(self.shader_code, "    lookup_elements: LookupElements,").unwrap();
        writeln!(self.shader_code, "}}").unwrap();
        writeln!(self.shader_code).unwrap();

        writeln!(self.shader_code, "struct ComputeCompositionPolynomialOutput {{").unwrap();
        writeln!(self.shader_code, "    poly: array<array<QM31, N_LANES>, N_PACKED_ROWS>,").unwrap();
        writeln!(self.shader_code, "    intermediates: array<M31, N_INTERMEDIATES>,").unwrap();
        writeln!(self.shader_code, "    ext_intermediates: array<QM31, N_EXT_INTERMEDIATES>,").unwrap();
        writeln!(self.shader_code, "}}").unwrap();
        writeln!(self.shader_code).unwrap();

        // Generate the proper binding declarations
        writeln!(
            self.shader_code,
            "@group(0) @binding(0)"
        ).unwrap();
        writeln!(
            self.shader_code,
            "var<storage, read> input: ComputeCompositionPolynomialInput;"
        ).unwrap();
        writeln!(self.shader_code).unwrap();

        writeln!(
            self.shader_code,
            "@group(0) @binding(1)"
        ).unwrap();
        writeln!(
            self.shader_code,
            "var<storage, read_write> output: ComputeCompositionPolynomialOutput;"
        ).unwrap();
        writeln!(self.shader_code).unwrap();

        writeln!(self.shader_code).unwrap();
    }

    fn generate_compute_function(&mut self) {
        // Use 16 threads per workgroup to allow more workgroups
        // This will create 4 workgroups for 64 rows (64/16 = 4)
        writeln!(self.shader_code, "@compute @workgroup_size(16, 1, 1)").unwrap();
        writeln!(self.shader_code, "fn main(").unwrap();
        writeln!(self.shader_code, "    @builtin(global_invocation_id) global_id: vec3<u32>,").unwrap();
        writeln!(self.shader_code, "    @builtin(local_invocation_id) local_id: vec3<u32>,").unwrap();
        writeln!(self.shader_code, "    @builtin(workgroup_id) workgroup_id: vec3<u32>").unwrap();
        writeln!(self.shader_code, ") {{").unwrap();
        writeln!(self.shader_code, "    // Each thread processes one row").unwrap();
        writeln!(self.shader_code, "    let index = global_id.x;").unwrap();
        writeln!(self.shader_code, "    ").unwrap();
        writeln!(self.shader_code, "    // Bounds check to ensure we don't process beyond the array").unwrap();
        writeln!(self.shader_code, "    if (index >= N_EXTENDED_ROWS) {{").unwrap();
        writeln!(self.shader_code, "        return;").unwrap();
        writeln!(self.shader_code, "    }}").unwrap();
        writeln!(self.shader_code, "    ").unwrap();
        writeln!(self.shader_code, "    var constraint_sum: QM31 = vec4<u32>(0u, 0u, 0u, 0u);").unwrap();
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
                let index_expr = if col.offset >= 0 {
                    format!("index + {}u", col.offset)
                } else {
                    format!("index - {}u", -col.offset)
                };
                writeln!(
                    self.shader_code,
                    "    let {} = input.extended_trace[{}].data[{}];",
                    var_name, col.idx, index_expr
                ).unwrap();
            }
            IRInstr::LoadConst { dest, value } => {
                let var_name = self.get_reg_var(*dest);
                writeln!(
                    self.shader_code,
                    "    let {}: M31 = {}u;",
                    var_name, value.0
                ).unwrap();
            }
            IRInstr::LoadParam { dest, name } => {
                let dest_var = self.get_reg_var(*dest);
                // Check if this is an intermediate value
                if let Some(&index) = self.intermediate_map.get(name) {
                    writeln!(
                        self.shader_code,
                        "    let {}: M31 = output.intermediates[{}u]; // Load {}",
                        dest_var, index, name
                    ).unwrap();
                } else {
                    // Handle special logup parameters
                    match name.as_str() {
                        "column_size" => {
                            writeln!(
                                self.shader_code,
                                "    let {}: M31 = input.column_size; // Load column_size",
                                dest_var
                            ).unwrap();
                        }
                        _ => {
                            // Regular parameter - not implemented yet
                            writeln!(
                                self.shader_code,
                                "    // TODO: Load parameter {} into {}",
                                name, dest_var
                            ).unwrap();
                        }
                    }
                }
            }
            IRInstr::Add { dest, lhs, rhs } => {
                let dest_var = self.get_reg_var(*dest);
                let lhs_var = self.get_reg_var(*lhs);
                let rhs_var = self.get_reg_var(*rhs);
                writeln!(
                    self.shader_code,
                    "    let {}: M31 = m31_add({}, {});",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::Sub { dest, lhs, rhs } => {
                let dest_var = self.get_reg_var(*dest);
                let lhs_var = self.get_reg_var(*lhs);
                let rhs_var = self.get_reg_var(*rhs);
                writeln!(
                    self.shader_code,
                    "    let {}: M31 = m31_sub({}, {});",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::Mul { dest, lhs, rhs } => {
                let dest_var = self.get_reg_var(*dest);
                let lhs_var = self.get_reg_var(*lhs);
                let rhs_var = self.get_reg_var(*rhs);
                writeln!(
                    self.shader_code,
                    "    let {}: M31 = m31_mul({}, {});",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::Neg { dest, op } => {
                let dest_var = self.get_reg_var(*dest);
                let op_var = self.get_reg_var(*op);
                writeln!(
                    self.shader_code,
                    "    let {}: M31 = m31_neg({});",
                    dest_var, op_var
                ).unwrap();
            }
            IRInstr::Inv { dest, op } => {
                let dest_var = self.get_reg_var(*dest);
                let op_var = self.get_reg_var(*op);
                writeln!(
                    self.shader_code,
                    "    let {}: M31 = m31_inverse({});",
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
                    "    let {}: QM31 = vec4<u32>({}, {}, {}, {});",
                    var_name, col0_var, col1_var, col2_var, col3_var
                ).unwrap();
            }
            IRInstr::LoadExtConst { dest, value } => {
                let var_name = self.get_reg4_var(*dest);
                writeln!(
                    self.shader_code,
                    "    let {}: QM31 = vec4<u32>({}u, {}u, {}u, {}u);",
                    var_name, value.0.0.0, value.0.1.0, value.1.0.0, value.1.1.0
                ).unwrap();
            }
            IRInstr::LoadExtParam { dest, name } => {
                let dest_var = self.get_reg4_var(*dest);
                // Check if this is an extension intermediate value
                if let Some(&index) = self.ext_intermediate_map.get(name) {
                    writeln!(
                        self.shader_code,
                        "    let {}: QM31 = output.ext_intermediates[{}u]; // Load {}",
                        dest_var, index, name
                    ).unwrap();
                } else {
                    // Handle special logup extension parameters
                    match name.as_str() {
                        "claimed_sum" => {
                            writeln!(
                                self.shader_code,
                                "    let {}: QM31 = input.claimed_sum; // Load claimed_sum",
                                dest_var
                            ).unwrap();
                        }
                        name if name.ends_with("_z") => {
                            writeln!(
                                self.shader_code,
                                "    let {}: QM31 = input.lookup_elements.z; // Load lookup z",
                                dest_var
                            ).unwrap();
                        }
                        name if name.ends_with("_alpha") => {
                            writeln!(
                                self.shader_code,
                                "    let {}: QM31 = input.lookup_elements.alpha; // Load lookup alpha",
                                dest_var
                            ).unwrap();
                        }
                        name if name.contains("_alpha") && name.chars().last().map_or(false, |c| c.is_ascii_digit()) => {
                            // Extract alpha power index from parameter name (e.g., "FibonacciRelation_alpha0" -> 0)
                            // Handle multi-digit numbers by parsing the suffix after "_alpha"
                            if let Some(alpha_pos) = name.rfind("_alpha") {
                                let index_str = &name[alpha_pos + 6..]; // Skip "_alpha"
                                if let Ok(index) = index_str.parse::<u32>() {
                                    writeln!(
                                        self.shader_code,
                                        "    let {}: QM31 = input.lookup_elements.alpha_powers[{}u]; // Load alpha power {}",
                                        dest_var, index, index
                                    ).unwrap();
                                } else {
                                    writeln!(
                                        self.shader_code,
                                        "    // TODO: Invalid alpha power index in parameter {} -> {}",
                                        name, dest_var
                                    ).unwrap();
                                }
                            } else {
                                writeln!(
                                    self.shader_code,
                                    "    // TODO: Load alpha power parameter {} into {}",
                                    name, dest_var
                                ).unwrap();
                            }
                        }
                        _ => {
                            // Regular extension parameter - not implemented yet
                            writeln!(
                                self.shader_code,
                                "    // TODO: Load ext parameter {} into {}",
                                name, dest_var
                            ).unwrap();
                        }
                    }
                }
            }
            IRInstr::AddExt { dest, lhs, rhs } => {
                let dest_var = self.get_reg4_var(*dest);
                let lhs_var = self.get_reg4_var(*lhs);
                let rhs_var = self.get_reg4_var(*rhs);
                writeln!(
                    self.shader_code,
                    "    let {}: QM31 = qm31_add({}, {});",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::SubExt { dest, lhs, rhs } => {
                let dest_var = self.get_reg4_var(*dest);
                let lhs_var = self.get_reg4_var(*lhs);
                let rhs_var = self.get_reg4_var(*rhs);
                writeln!(
                    self.shader_code,
                    "    let {}: QM31 = qm31_sub({}, {});",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::MulExt { dest, lhs, rhs } => {
                let dest_var = self.get_reg4_var(*dest);
                let lhs_var = self.get_reg4_var(*lhs);
                let rhs_var = self.get_reg4_var(*rhs);
                writeln!(
                    self.shader_code,
                    "    let {}: QM31 = qm31_mul({}, {});",
                    dest_var, lhs_var, rhs_var
                ).unwrap();
            }
            IRInstr::NegExt { dest, op } => {
                let dest_var = self.get_reg4_var(*dest);
                let op_var = self.get_reg4_var(*op);
                writeln!(
                    self.shader_code,
                    "    let {}: QM31 = qm31_neg({});",
                    dest_var, op_var
                ).unwrap();
            }
            IRInstr::StoreIntermediate { reg, name } => {
                let reg_var = self.get_reg_var(*reg);
                let index = self.intermediate_map.entry(name.clone()).or_insert_with(|| {
                    let idx = self.intermediate_counter;
                    self.intermediate_counter += 1;
                    idx
                });
                writeln!(
                    self.shader_code,
                    "    output.intermediates[{}u] = {}; // Store {}",
                    index, reg_var, name
                ).unwrap();
            }
            IRInstr::StoreExtIntermediate { reg, name } => {
                let reg_var = self.get_reg4_var(*reg);
                let index = self.ext_intermediate_map.entry(name.clone()).or_insert_with(|| {
                    let idx = self.ext_intermediate_counter;
                    self.ext_intermediate_counter += 1;
                    idx
                });
                writeln!(
                    self.shader_code,
                    "    output.ext_intermediates[{}u] = {}; // Store {}",
                    index, reg_var, name
                ).unwrap();
            }
            IRInstr::AssertZero { reg } => {
                let reg_var = self.get_reg4_var(*reg);
                writeln!(
                    self.shader_code,
                    "    constraint_sum = qm31_add(constraint_sum, qm31_mul(input.random_coeff_powers[{}], {}));",
                    self.constraint_index, reg_var
                ).unwrap();
                self.constraint_index += 1;
            }
        }
    }

    fn generate_footer(&mut self) {
        writeln!(self.shader_code, "    // Store constraint_sum in the appropriate position").unwrap();
        writeln!(self.shader_code, "    let packed_index = index / N_LANES;").unwrap();
        writeln!(self.shader_code, "    let lane_index = index % N_LANES;").unwrap();
        writeln!(self.shader_code, "    output.poly[packed_index][lane_index] = constraint_sum;").unwrap();
        writeln!(self.shader_code, "}}").unwrap();
    }

    fn get_reg_var(&mut self, reg: Reg) -> String {
        if let Some(var_name) = self.reg_map.get(&reg.0) {
            var_name.clone()
        } else {
            let var_name = format!("r{}", self.reg_counter);
            self.reg_counter += 1;
            self.reg_map.insert(reg.0, var_name.clone());
            var_name
        }
    }

    fn get_reg4_var(&mut self, reg: Reg4) -> String {
        if let Some(var_name) = self.reg4_map.get(&reg.0) {
            var_name.clone()
        } else {
            let var_name = format!("r4_{}", self.reg4_counter);
            self.reg4_counter += 1;
            self.reg4_map.insert(reg.0, var_name.clone());
            var_name
        }
    }
}

// Type alias for default configuration
use super::constants::DefaultConfig;
pub type DefaultWgslGenerator = WgslGenerator<DefaultConfig>;

impl Default for DefaultWgslGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::ir::{IRInstr, Reg};
    use crate::expr::ColumnExpr;
    use stwo::core::fields::m31::BaseField;

    #[test]
    fn test_simple_wgsl_generation() {
        let mut generator = DefaultWgslGenerator::new();
        
        // Simple test: r0 = col(0,0,0) + 5
        let instructions = vec![
            IRInstr::LoadCol { 
                dest: Reg(0), 
                col: ColumnExpr::from((0, 0, 0)) 
            },
            IRInstr::LoadConst { 
                dest: Reg(1), 
                value: BaseField::from(5) 
            },
            IRInstr::Add { 
                dest: Reg(2), 
                lhs: Reg(0), 
                rhs: Reg(1) 
            },
            IRInstr::LoadExtCol { 
                dest: Reg4(0), 
                col: [Reg(2), Reg(0), Reg(0), Reg(0)] 
            },
            IRInstr::AssertZero { reg: Reg4(0) },
        ];

        let wgsl_code = generator.generate_wgsl(&instructions, true);
        
        println!("Generated WGSL:\n{}", wgsl_code);
    }

    #[test]
    fn test_logup_parameter_generation() {
        let mut generator = DefaultWgslGenerator::new();
        
        // Test logup parameters: claimed_sum / column_size
        let instructions = vec![
            IRInstr::LoadExtParam { 
                dest: Reg4(0), 
                name: "claimed_sum".to_string() 
            },
            IRInstr::LoadParam { 
                dest: Reg(0), 
                name: "column_size".to_string() 
            },
            IRInstr::LoadExtCol { 
                dest: Reg4(1), 
                col: [Reg(0), Reg(0), Reg(0), Reg(0)] 
            },
            IRInstr::AssertZero { reg: Reg4(1) },
        ];

        let wgsl_code = generator.generate_wgsl(&instructions, true);
        
        // Check that claimed_sum and column_size are properly handled
        assert!(wgsl_code.contains("claimed_sum: QM31,"));
        assert!(wgsl_code.contains("column_size: u32,"));
        assert!(wgsl_code.contains("input.claimed_sum"));
        assert!(wgsl_code.contains("input.column_size"));
        
        println!("Generated WGSL with logup params:\n{}", wgsl_code);
    }

    #[test]
    fn test_general_alpha_power_parsing() {
        let mut generator = DefaultWgslGenerator::new();
        
        // Test alpha powers for different ranges (simulating various N_LOOKUP_ELEMENTS)
        let instructions = vec![
            // Single digit alpha powers
            IRInstr::LoadExtParam { dest: Reg4(0), name: "TestRelation_alpha0".to_string() },
            IRInstr::LoadExtParam { dest: Reg4(1), name: "TestRelation_alpha7".to_string() },
            // Multi-digit alpha powers  
            IRInstr::LoadExtParam { dest: Reg4(2), name: "PoseidonElements_alpha10".to_string() },
            IRInstr::LoadExtParam { dest: Reg4(3), name: "PoseidonElements_alpha15".to_string() },
            // Edge cases
            IRInstr::LoadExtParam { dest: Reg4(4), name: "SomeRelation_z".to_string() },
            IRInstr::LoadExtParam { dest: Reg4(5), name: "claimed_sum".to_string() },
        ];
        
        let wgsl_code = generator.generate_wgsl(&instructions, true);
        
        // Verify that all alpha power accesses are generated correctly
        assert!(wgsl_code.contains("input.lookup_elements.alpha_powers[0u]"));
        assert!(wgsl_code.contains("input.lookup_elements.alpha_powers[7u]"));
        assert!(wgsl_code.contains("input.lookup_elements.alpha_powers[10u]"));
        assert!(wgsl_code.contains("input.lookup_elements.alpha_powers[15u]"));
        assert!(wgsl_code.contains("input.lookup_elements.z"));
        assert!(wgsl_code.contains("input.claimed_sum"));
        
        println!("Generated WGSL with general alpha powers:\n{}", wgsl_code);
    }


}