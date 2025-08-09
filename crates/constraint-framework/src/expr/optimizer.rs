use std::collections::{HashMap, HashSet};

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;

use crate::expr::ir::{IRInstr, InstrDest, Reg, Reg4};
use crate::expr::ColumnExpr;

#[derive(Hash, PartialEq, Eq)]
enum IRInstrId {
    // Load
    Col(ColumnExpr),
    Const(BaseField),
    Param(String),
    ExtConst(SecureField),
    ExtParam(String),
    ExtCol([Reg; 4]),
    // arithmetic
    Bin(usize, Reg, Reg), // commutative?
    Un(usize, Reg),
    Bin4(usize, Reg4, Reg4),
    Un4(usize, Reg4),
}

fn order(lhs: Reg, rhs: Reg, is_comm: bool) -> (Reg, Reg) {
    if is_comm && lhs.0 > rhs.0 {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}
fn order4(lhs: Reg4, rhs: Reg4, is_comm: bool) -> (Reg4, Reg4) {
    if is_comm && lhs.0 > rhs.0 {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

pub fn global_cse(old: Vec<IRInstr>) -> Vec<IRInstr> {
    let mut reg_map: HashMap<Reg, Reg> = HashMap::new();
    let mut reg4_map: HashMap<Reg4, Reg4> = HashMap::new();
    let mut id_reg: HashMap<IRInstrId, Reg> = HashMap::new();
    let mut id_reg4: HashMap<IRInstrId, Reg4> = HashMap::new();

    let mut new_ir = Vec::new();

    for instr in old {
        // apply current mapping
        let inst = instr.map_reg(
            |reg| *reg_map.get(&reg).unwrap_or(&reg),
            |reg4| *reg4_map.get(&reg4).unwrap_or(&reg4),
        );

        // new instruction identifier
        let new_id = match inst {
            IRInstr::LoadCol { ref col, .. } => IRInstrId::Col(col.clone()),
            IRInstr::LoadConst { ref value, .. } => IRInstrId::Const(value.clone()),
            IRInstr::LoadParam { ref name, .. } => IRInstrId::Param(name.clone()),
            IRInstr::LoadExtCol { col, .. } => IRInstrId::ExtCol(col),
            IRInstr::LoadExtConst { ref value, .. } => IRInstrId::ExtConst(value.clone()),
            IRInstr::LoadExtParam { ref name, .. } => IRInstrId::ExtParam(name.clone()),
            IRInstr::Add { lhs, rhs, .. }
            | IRInstr::Mul { lhs, rhs, .. }
            | IRInstr::Sub { lhs, rhs, .. } => {
                let (lhs, rhs) = order(lhs, rhs, inst.is_commutative());
                IRInstrId::Bin(instr.opcode(), lhs, rhs)
            }
            IRInstr::Neg { op, .. } | IRInstr::Inv { op, .. } => IRInstrId::Un(instr.opcode(), op),
            IRInstr::AddExt { lhs, rhs, .. }
            | IRInstr::MulExt { lhs, rhs, .. }
            | IRInstr::SubExt { lhs, rhs, .. } => {
                let (lhs, rhs) = order4(lhs, rhs, instr.is_commutative());
                IRInstrId::Bin4(instr.opcode(), lhs, rhs)
            }
            IRInstr::NegExt { op, .. } => IRInstrId::Un4(instr.opcode(), op),
            IRInstr::AssertZero { reg, .. } => IRInstrId::Un4(instr.opcode(), reg),
        };

        // check duplication
        let duplicated = match inst.dest() {
            InstrDest::Reg(reg) => match id_reg.get(&new_id) {
                Some(&prev) => {
                    reg_map.insert(reg, prev); // same register
                    true
                }
                None => {
                    id_reg.insert(new_id, reg);
                    false
                }
            },
            InstrDest::Reg4(reg4) => match id_reg4.get(&new_id) {
                Some(&prev) => {
                    reg4_map.insert(reg4, prev); // same register
                    true
                }
                None => {
                    id_reg4.insert(new_id, reg4);
                    false
                }
            },
            InstrDest::None => false,
        };

        if !duplicated {
            new_ir.push(inst);
        }
    }

    new_ir
}

#[allow(warnings)]
pub fn dce(old: Vec<IRInstr>) -> Vec<IRInstr> {
    let mut live_r: HashSet<Reg> = HashSet::new();
    let mut live_r4: HashSet<Reg4> = HashSet::new();
    let mut new_ir = Vec::new();

    for instr in old.into_iter().rev() {
        // need to know what instrution are using and what is the destination of the instruction

        //
    }

    new_ir
}
