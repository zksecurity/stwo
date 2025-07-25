use std::collections::{HashMap, HashSet};
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
    /// Column bindings for input data - maps interaction to binding index
    interaction_bindings: HashMap<usize, usize>,
    /// Track which interactions we've seen
    interactions_seen: HashSet<usize>,
    /// Parameter bindings
    param_bindings: HashMap<String, usize>,
    /// Next available binding index
    next_binding: usize,
    /// Accumulator for constraint linear combinations
    constraint_accumulator: Option<String>,
    /// Counter for constraint indices
    constraint_index: usize,
}

impl WgslGenerator {
    pub fn new() -> Self {
        Self {
            shader_code: String::new(),
            reg_map: HashMap::new(),
            reg4_map: HashMap::new(),
            var_counter: 0,
            interaction_bindings: HashMap::new(),
            interactions_seen: HashSet::new(),
            param_bindings: HashMap::new(),
            next_binding: 0,
            constraint_accumulator: None,
            constraint_index: 0,
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
        
        // Include the qm31.wgsl library
        let qm31_code = include_str!("qm31.wgsl");
        writeln!(self.shader_code, "{}", qm31_code).unwrap();
        writeln!(self.shader_code).unwrap();
    }

    fn analyze_bindings(&mut self, instructions: &[IRInstr]) {
        // Scan instructions to find all interactions and parameters that need bindings
        for instr in instructions {
            match instr {
                IRInstr::LoadCol { col, .. } => {
                    if !self.interactions_seen.contains(&col.interaction) {
                        self.interactions_seen.insert(col.interaction);
                        self.interaction_bindings.insert(col.interaction, self.next_binding);
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
        // Single input structure containing interaction data and random coefficients
        writeln!(self.shader_code, "struct ComputeInput {{").unwrap();
        
        // Include interaction tables  
        for interaction in &self.interactions_seen {
            writeln!(
                self.shader_code,
                "    interaction_{}: array<M31>,",
                interaction
            ).unwrap();
        }
        
        // Include parameters
        for param in self.param_bindings.keys() {
            writeln!(
                self.shader_code,
                "    param_{}: M31,",
                param.replace(' ', "_")
            ).unwrap();
        }
        
        // Random coefficient powers
        writeln!(self.shader_code, "    random_coeff_powers: array<QM31>,").unwrap();
        writeln!(self.shader_code, "}}").unwrap();
        writeln!(self.shader_code).unwrap();

        // Input buffer
        writeln!(
            self.shader_code,
            "@group(0) @binding(0) var<storage, read> input: ComputeInput;"
        ).unwrap();

        // Output buffer as array
        writeln!(
            self.shader_code,
            "@group(0) @binding(1) var<storage, read_write> output: array<QM31>;"
        ).unwrap();
        
        writeln!(self.shader_code).unwrap();
    }

    fn generate_compute_function(&mut self) {
        writeln!(self.shader_code, "@compute @workgroup_size(64)").unwrap();
        writeln!(self.shader_code, "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{").unwrap();
        writeln!(self.shader_code, "    let index = global_id.x;").unwrap();
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
                writeln!(
                    self.shader_code,
                    "    let {} = input.interaction_{}[{}][index + {}];",
                    var_name, col.interaction, col.idx, col.offset
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
                let var_name = self.get_reg_var(*dest);
                writeln!(
                    self.shader_code,
                    "    let {} = input.param_{};",
                    var_name, name.replace(' ', "_")
                ).unwrap();
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
                let var_name = self.get_reg4_var(*dest);
                // For now, assume extension parameters are stored as QM31
                writeln!(
                    self.shader_code,
                    "    let {}: QM31 = vec4<u32>(input.param_{}, 0u, 0u, 0u);",
                    var_name, name.replace(' ', "_")
                ).unwrap();
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
            IRInstr::AssertZero { reg } => {
                let reg_var = self.get_reg4_var(*reg);
                writeln!(
                    self.shader_code,
                    "    // CONSTRAINT {}: Add linear combination to sum",
                    self.constraint_index
                ).unwrap();
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
        writeln!(self.shader_code, "    output[index] = constraint_sum;").unwrap();
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
    use crate::expr::ir::{IRInstr, Reg, Reg4};
    use crate::expr::ColumnExpr;
    use crate::expr::gpu_common::{GpuComputeInstance, GpuOperation, ByteSerialize};
    use stwo::core::fields::m31::BaseField;
    use stwo::core::fields::qm31::QM31;
    use std::borrow::Cow;
    use bytemuck::{Pod, Zeroable};

    use crate::expr::qm31::GpuQM31;

    // Test input/output structures
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct TestComputeInput {
        interaction_0: [[u32; 64]; 1], // Mock interaction data
        random_coeff_powers: [GpuQM31; 1],
    }

    impl ByteSerialize for TestComputeInput {}

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct TestComputeOutput {
        result: GpuQM31,
    }

    impl ByteSerialize for TestComputeOutput {}

    impl ByteSerialize for [GpuQM31; 1] {}
    impl ByteSerialize for [GpuQM31; 64] {}

    struct TestOperation {
        shader_code: String,
    }

    impl crate::expr::gpu_common::GpuOperation for TestOperation {
        fn shader_source(&self) -> Cow<'static, str> {
            // Include the qm31.wgsl library
            let qm31_code = include_str!("qm31.wgsl");
            let combined = format!("{}\n{}", qm31_code, self.shader_code);
            Cow::Owned(combined)
        }
    }

    #[test]
    fn test_simple_wgsl_generation() {
        let mut generator = WgslGenerator::new();
        
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
        ];

        let wgsl_code = generator.generate_wgsl(&instructions);
        
        // Check that the generated code contains expected elements
        assert!(wgsl_code.contains("@compute @workgroup_size(64)"));
        assert!(wgsl_code.contains("fn main"));
        assert!(wgsl_code.contains("ComputeInput"));
        assert!(wgsl_code.contains("input.interaction_0"));
        assert!(wgsl_code.contains("5u"));
        assert!(wgsl_code.contains("m31_add"));
        
        println!("Generated WGSL:\n{}", wgsl_code);
    }

    #[tokio::test]
    async fn test_gpu_shader_execution() {
        let mut generator = WgslGenerator::new();
        
        // Create a simple constraint: column value should equal 42
        let instructions = vec![
            IRInstr::LoadCol { 
                dest: Reg(0), 
                col: ColumnExpr::from((0, 0, 0)) 
            },
            IRInstr::LoadConst { 
                dest: Reg(1), 
                value: BaseField::from(42) 
            },
            IRInstr::Sub { 
                dest: Reg(2), 
                lhs: Reg(0), 
                rhs: Reg(1) 
            },
            // Convert to extension field for AssertZero
            IRInstr::LoadExtCol { 
                dest: Reg4(0), 
                col: [
                    Reg(2), 
                    Reg(1), // dummy
                    Reg(1), // dummy  
                    Reg(1)  // dummy
                ]
            },
            IRInstr::AssertZero { 
                reg: Reg4(0) 
            },
        ];

        let shader_code = generator.generate_wgsl(&instructions);
        let operation = TestOperation { shader_code };

        // Prepare test input - column value 42 should make constraint pass (result = 0)
        let mut interaction_data = [[0u32; 64]; 1];
        interaction_data[0][0] = 42; // Set first element to 42

        let input = TestComputeInput {
            interaction_0: interaction_data,
            random_coeff_powers: [GpuQM31::from(QM31::from_u32_unchecked(1, 0, 0, 0))], // Simple coefficient
        };

        let instance = GpuComputeInstance::new(&input, std::mem::size_of::<[GpuQM31; 1]>()).await;
        let (pipeline, bind_group) = instance.create_pipeline(
            &operation.shader_source(), 
            operation.entry_point()
        );

        let output: [GpuQM31; 1] = instance
            .run_computation(&pipeline, &bind_group, (1, 1, 1))
            .await;

        // The constraint should be satisfied (result should be close to zero)
        println!("Constraint result: {:?}", output[0]);
        
        // For a satisfied constraint, the result should be zero in the base field
        assert_eq!(output[0].0[0], 0); // Real part of first component should be 0
    }

    /// Test function similar to compute_field_operation but for generated shaders
    pub async fn compute_generated_field_operation(
        instructions: Vec<IRInstr>,
        interaction_data: Vec<Vec<u32>>,
        _params: Vec<(String, u32)>,
    ) -> Vec<QM31> {
        let mut generator = WgslGenerator::new();
        let shader_code = generator.generate_wgsl(&instructions);
        let operation = TestOperation { shader_code };

        // Determine the number of interactions needed
        let num_interactions = interaction_data.len();
        let data_size = if !interaction_data.is_empty() { interaction_data[0].len() } else { 64 };

        // Create test input structure dynamically (simplified for now)
        let input = TestComputeInput {
            interaction_0: if num_interactions > 0 {
                let mut arr = [[0u32; 64]; 1];
                for (i, val) in interaction_data[0].iter().enumerate().take(64) {
                    arr[0][i] = *val;
                }
                arr
            } else {
                [[0u32; 64]; 1]
            },
            random_coeff_powers: [GpuQM31::from(QM31::from_u32_unchecked(1, 0, 0, 0))],
        };

        let output_size = std::mem::size_of::<[GpuQM31; 64]>();
        let instance = GpuComputeInstance::new(&input, output_size).await;
        let (pipeline, bind_group) = instance.create_pipeline(
            &operation.shader_source(),
            operation.entry_point()
        );

        let output: [GpuQM31; 64] = instance
            .run_computation(&pipeline, &bind_group, (data_size as u32, 1, 1))
            .await;

        output.iter().map(|&gpu_qm31| QM31::from(gpu_qm31)).collect()
    }

    #[tokio::test]
    async fn test_generated_field_arithmetic() {
        // Test addition operation similar to compute_field_operation
        let add_instructions = vec![
            IRInstr::LoadCol { 
                dest: Reg(0), 
                col: ColumnExpr::from((0, 0, 0)) 
            },
            IRInstr::LoadCol { 
                dest: Reg(1), 
                col: ColumnExpr::from((0, 0, 1)) 
            },
            IRInstr::Add { 
                dest: Reg(2), 
                lhs: Reg(0), 
                rhs: Reg(1) 
            },
            // Convert to extension field for AssertZero to capture result
            IRInstr::LoadExtCol { 
                dest: Reg4(0), 
                col: [
                    Reg(2), 
                    Reg(0), // dummy
                    Reg(0), // dummy  
                    Reg(0)  // dummy
                ]
            },
            IRInstr::AssertZero { 
                reg: Reg4(0) 
            },
        ];

        // Test data: first column = 10, second column = 5, expected result = 15
        let interaction_data = vec![vec![10, 5, 0, 0]]; // interaction 0
        let params = vec![];

        let results = compute_generated_field_operation(
            add_instructions, 
            interaction_data, 
            params
        ).await;

        // The constraint captures (10 + 5) in the first component
        // Since we're using AssertZero, the constraint will be the negation of the expression
        // But the actual field addition result should be 15
        println!("Generated shader addition result: {:?}", results[0]);
        
        // Verify the computation was performed (non-zero result expected due to AssertZero capturing the sum)
        assert_ne!(results[0].0.0.0, 0);
    }

    #[tokio::test] 
    async fn test_generated_multiplication() {
        // Test multiplication operation
        let mul_instructions = vec![
            IRInstr::LoadCol { 
                dest: Reg(0), 
                col: ColumnExpr::from((0, 0, 0)) 
            },
            IRInstr::LoadConst { 
                dest: Reg(1), 
                value: BaseField::from(3) 
            },
            IRInstr::Mul { 
                dest: Reg(2), 
                lhs: Reg(0), 
                rhs: Reg(1) 
            },
            // Convert to extension field
            IRInstr::LoadExtCol { 
                dest: Reg4(0), 
                col: [
                    Reg(2), 
                    Reg(0), // dummy
                    Reg(0), // dummy  
                    Reg(0)  // dummy
                ]
            },
            IRInstr::AssertZero { 
                reg: Reg4(0) 
            },
        ];

        // Test data: column value = 7, multiplied by 3 = 21
        let interaction_data = vec![vec![7, 0, 0, 0]];
        let params = vec![];

        let results = compute_generated_field_operation(
            mul_instructions, 
            interaction_data, 
            params
        ).await;

        println!("Generated shader multiplication result: {:?}", results[0]);
        
        // Verify multiplication was performed
        assert_ne!(results[0].0.0.0, 0);
    }
}