# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Expression System Architecture

This directory implements a symbolic expression system for the Stwo CSTARK constraint framework. The system operates on two main expression types:

- **BaseExpr**: Represents base field operations (M31 field elements)
- **ExtExpr**: Represents extension field operations (QM31 secure field elements)

### Core Components

- **mod.rs**: Defines the main expression AST types (`BaseExpr`, `ExtExpr`) with reference-counted inner nodes for sharing
- **ir.rs**: Contains the Intermediate Representation (IR) that compiles expressions into instruction sequences with virtual registers
- **wgsl_gen.rs**: WGSL code generator that converts IR instructions to GPU compute shaders
- **evaluator.rs**: Expression evaluator for constraint checking and formal verification
- **optimizer.rs**: Expression tree optimization passes
- **simplify.rs**: Algebraic simplification rules for expressions

### Key Concepts

**Expression Types:**
- `ColumnExpr`: References trace columns by (interaction, index, offset)
- `BaseExpr`: Operations on base field elements (Add, Sub, Mul, Neg, Inv, Col, Const, Param)
- `ExtExpr`: Extension field operations with 4-element secure columns

**IR Compilation:**
- Expressions are compiled to register-based IR instructions
- `Reg` for base field registers, `Reg4` for extension field registers
- Common subexpression elimination through expression caching
- Depth-first traversal builds instruction sequences

**WGSL Generation:**
- IR instructions are translated to WGSL compute shader code
- Handles column bindings and parameter mappings
- Generates GPU-optimized field arithmetic

### Build and Test Commands

From the workspace root (`/Users/jaehunkim/work/stwo_wgpu`):

```bash
# Build the entire workspace
cargo build

# Run tests for constraint-framework
cargo test -p stwo-constraint-framework

# Run benchmarks
cargo bench

# Build with parallel feature
cargo build --features parallel

# Build with prover feature  
cargo build --features prover
```

### Development Workflow

When working with expressions:
1. Understand the dual representation: AST (BaseExpr/ExtExpr) → IR → WGSL
2. Use the evaluator for testing constraint logic
3. Apply optimizations before IR generation for performance
4. The IR uses virtual registers - register allocation happens during WGSL generation

The expression system is central to constraint definition in the CSTARK framework, enabling both CPU evaluation and GPU acceleration through WGSL compilation.