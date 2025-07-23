use std::collections::HashMap;
use std::ops::Deref;
use std::rc::Rc;

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;

use crate::expr::{BaseExpr, ColumnExpr, ExtExpr};

#[derive(Debug)]
// string is to store the name of the variable
pub enum IRInstr {
    // === BASE FIELD OPERATIONS ===
    LoadCol { dest: Reg, col: ColumnExpr },
    LoadConst { dest: Reg, value: BaseField },
    LoadParam { dest: Reg, name: String },
    Add { dest: Reg, lhs: Reg, rhs: Reg },
    Sub { dest: Reg, lhs: Reg, rhs: Reg },
    Mul { dest: Reg, lhs: Reg, rhs: Reg },
    Neg { dest: Reg, op: Reg },
    Inv { dest: Reg, op: Reg },

    // === EXTENSION FIELD OPERATIONS ===
    LoadExtCol { dest: Reg4, col: [Reg; 4] },
    LoadExtConst { dest: Reg4, value: SecureField },
    LoadExtParam { dest: Reg4, name: String },
    AddExt { dest: Reg4, lhs: Reg4, rhs: Reg4 },
    SubExt { dest: Reg4, lhs: Reg4, rhs: Reg4 },
    MulExt { dest: Reg4, lhs: Reg4, rhs: Reg4 },
    NegExt { dest: Reg4, op: Reg4 },

    // === CONSTRAINT ASSERTIONS ===
    /// Assert that the extension field register contains zero (constraint)
    AssertZero { reg: Reg4 },
}

#[derive(Debug)]
pub enum InstrDest {
    Reg(Reg),
    Reg4(Reg4),
    None,
}

impl IRInstr {
    pub fn is_commutative(&self) -> bool {
        match self {
            IRInstr::Add { .. }
            | IRInstr::Mul { .. }
            | IRInstr::AddExt { .. }
            | IRInstr::MulExt { .. } => true,
            _ => false,
        }
    }

    pub fn map_reg<F, G>(&self, f_reg: F, f_reg4: G) -> Self
    where
        F: Fn(Reg) -> Reg,
        G: Fn(Reg4) -> Reg4,
    {
        match *self {
            IRInstr::LoadCol { dest, ref col } => IRInstr::LoadCol {
                dest: f_reg(dest),
                col: col.clone(),
            },
            IRInstr::LoadConst { dest, ref value } => IRInstr::LoadConst {
                dest: f_reg(dest),
                value: value.clone(),
            },
            IRInstr::LoadParam { dest, ref name } => IRInstr::LoadParam {
                dest: f_reg(dest),
                name: name.clone(),
            },
            IRInstr::Add { dest, lhs, rhs } => IRInstr::Add {
                dest: f_reg(dest),
                lhs: f_reg(lhs),
                rhs: f_reg(rhs),
            },
            IRInstr::Sub { dest, lhs, rhs } => IRInstr::Sub {
                dest: f_reg(dest),
                lhs: f_reg(lhs),
                rhs: f_reg(rhs),
            },
            IRInstr::Mul { dest, lhs, rhs } => IRInstr::Mul {
                dest: f_reg(dest),
                lhs: f_reg(lhs),
                rhs: f_reg(rhs),
            },
            IRInstr::Neg { dest, op } => IRInstr::Neg {
                dest: f_reg(dest),
                op: f_reg(op),
            },
            IRInstr::Inv { dest, op } => IRInstr::Inv {
                dest: f_reg(dest),
                op: f_reg(op),
            },
            IRInstr::LoadExtCol { dest, col } => IRInstr::LoadExtCol {
                dest: f_reg4(dest),
                col: [f_reg(col[0]), f_reg(col[1]), f_reg(col[2]), f_reg(col[3])],
            },
            IRInstr::LoadExtConst { dest, ref value } => IRInstr::LoadExtConst {
                dest: f_reg4(dest),
                value: value.clone(),
            },
            IRInstr::LoadExtParam { dest, ref name } => IRInstr::LoadExtParam {
                dest: f_reg4(dest),
                name: name.clone(),
            },
            IRInstr::AddExt { dest, lhs, rhs } => IRInstr::AddExt {
                dest: f_reg4(dest),
                lhs: f_reg4(lhs),
                rhs: f_reg4(rhs),
            },
            IRInstr::SubExt { dest, lhs, rhs } => IRInstr::SubExt {
                dest: f_reg4(dest),
                lhs: f_reg4(lhs),
                rhs: f_reg4(rhs),
            },
            IRInstr::MulExt { dest, lhs, rhs } => IRInstr::MulExt {
                dest: f_reg4(dest),
                lhs: f_reg4(lhs),
                rhs: f_reg4(rhs),
            },
            IRInstr::NegExt { dest, op } => IRInstr::NegExt {
                dest: f_reg4(dest),
                op: f_reg4(op),
            },
            IRInstr::AssertZero { reg } => IRInstr::AssertZero {
                reg: f_reg4(reg),
            },
        }
    }

    pub fn opcode(&self) -> usize {
        match self {
            IRInstr::LoadCol { .. } => 0,
            IRInstr::LoadConst { .. } => 1,
            IRInstr::LoadParam { .. } => 2,
            IRInstr::Add { .. } => 3,
            IRInstr::Sub { .. } => 4,
            IRInstr::Mul { .. } => 5,
            IRInstr::Neg { .. } => 6,
            IRInstr::Inv { .. } => 7,
            IRInstr::LoadExtCol { .. } => 8,
            IRInstr::LoadExtConst { .. } => 9,
            IRInstr::LoadExtParam { .. } => 10,
            IRInstr::AddExt { .. } => 11,
            IRInstr::SubExt { .. } => 12,
            IRInstr::MulExt { .. } => 13,
            IRInstr::NegExt { .. } => 14,
            IRInstr::AssertZero { .. } => 15,
        }
    }

    pub fn dest(&self) -> InstrDest {
        match *self {
            IRInstr::LoadCol { dest, .. } => InstrDest::Reg(dest),
            IRInstr::LoadConst { dest, .. } => InstrDest::Reg(dest),
            IRInstr::LoadParam { dest, .. } => InstrDest::Reg(dest),
            IRInstr::Add { dest, .. } => InstrDest::Reg(dest),
            IRInstr::Sub { dest, .. } => InstrDest::Reg(dest),
            IRInstr::Mul { dest, .. } => InstrDest::Reg(dest),
            IRInstr::Neg { dest, .. } => InstrDest::Reg(dest),
            IRInstr::Inv { dest, .. } => InstrDest::Reg(dest),
            IRInstr::LoadExtCol { dest, .. } => InstrDest::Reg4(dest),
            IRInstr::LoadExtConst { dest, .. } => InstrDest::Reg4(dest),
            IRInstr::LoadExtParam { dest, .. } => InstrDest::Reg4(dest),
            IRInstr::AddExt { dest, .. } => InstrDest::Reg4(dest),
            IRInstr::SubExt { dest, .. } => InstrDest::Reg4(dest),
            IRInstr::MulExt { dest, .. } => InstrDest::Reg4(dest),
            IRInstr::NegExt { dest, .. } => InstrDest::Reg4(dest),
            IRInstr::AssertZero { .. } => InstrDest::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg4(pub(crate) usize);

// the garuentee
#[derive(Eq, PartialEq, Hash, Debug)]
pub enum ExprId {
    Col(ColumnExpr),
    Const(BaseField),
    Param(String),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Neg(usize),
    Inv(usize),
}

impl BaseExpr {
    fn id(&self) -> ExprId {
        match self {
            BaseExpr::Col(c) => ExprId::Col(c.clone()),
            BaseExpr::Const(c) => ExprId::Const(c.clone()),
            BaseExpr::Param(c) => ExprId::Param(c.clone()),
            BaseExpr::Add(l, r) => {
                let lptr = Rc::as_ptr(&l.0);
                let rptr = Rc::as_ptr(&r.0);
                ExprId::Add(lptr as usize, rptr as usize)
            }
            BaseExpr::Sub(l, r) => {
                let lptr = Rc::as_ptr(&l.0);
                let rptr = Rc::as_ptr(&r.0);
                ExprId::Sub(lptr as usize, rptr as usize)
            }
            BaseExpr::Mul(l, r) => {
                let lptr = Rc::as_ptr(&l.0);
                let rptr = Rc::as_ptr(&r.0);
                ExprId::Mul(lptr as usize, rptr as usize)
            }
            BaseExpr::Neg(c) => {
                let ptr = Rc::as_ptr(&c.0);
                ExprId::Neg(ptr as usize)
            }
            BaseExpr::Inv(c) => {
                let ptr = Rc::as_ptr(&c.0);
                ExprId::Inv(ptr as usize)
            }
        }
    }
}

#[derive(Eq, PartialEq, Hash)]
pub enum ExtExprId {
    Col([ExprId; 4]),
    Const(SecureField),
    Param(String),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Neg(usize),
}

impl ExtExpr {
    fn id(&self) -> ExtExprId {
        match self {
            ExtExpr::SecureCol(arr) => ExtExprId::Col([
                arr[0].deref().id(),
                arr[1].deref().id(),
                arr[2].deref().id(),
                arr[3].deref().id(),
            ]),
            ExtExpr::Const(c) => ExtExprId::Const(c.clone()),
            ExtExpr::Param(s) => ExtExprId::Param(s.clone()),
            ExtExpr::Add(l, r) => {
                let lptr = Rc::as_ptr(&l.0) as usize;
                let rptr = Rc::as_ptr(&r.0) as usize;
                ExtExprId::Add(lptr, rptr)
            }
            ExtExpr::Sub(l, r) => {
                let lptr = Rc::as_ptr(&l.0) as usize;
                let rptr = Rc::as_ptr(&r.0) as usize;
                ExtExprId::Sub(lptr, rptr)
            }
            ExtExpr::Mul(l, r) => {
                let lptr = Rc::as_ptr(&l.0) as usize;
                let rptr = Rc::as_ptr(&r.0) as usize;
                ExtExprId::Mul(lptr, rptr)
            }
            ExtExpr::Neg(e) => {
                let ptr = Rc::as_ptr(&e.0) as usize;
                ExtExprId::Neg(ptr)
            }
        }
    }
}

pub struct IRBuilder {
    pub instrs: Vec<IRInstr>,
    pub env: HashMap<ExprId, Reg>,
    pub ext_env: HashMap<ExtExprId, Reg4>,
    reg_id: usize,
    ext_reg_id: usize,
}

impl IRBuilder {
    pub fn new() -> Self {
        Self {
            instrs: vec![],
            env: HashMap::new(),
            ext_env: HashMap::new(),
            reg_id: 0,
            ext_reg_id: 0,
        }
    }

    fn next_reg(&mut self) -> Reg {
        let id = self.reg_id;
        self.reg_id += 1;
        Reg(id)
    }

    fn next_reg4(&mut self) -> Reg4 {
        let id = self.ext_reg_id;
        self.ext_reg_id += 1;
        Reg4(id)
    }

    // depth-first traverse the expression tree
    pub fn build_ir(&mut self, expr: &BaseExpr) -> Reg {
        // query the expression cache
        if let Some(&reg) = self.env.get(&expr.id()) {
            return reg;
        }

        // compile recursively
        let reg = match expr {
            BaseExpr::Col(c) => {
                let reg = self.next_reg();
                self.instrs.push(IRInstr::LoadCol {
                    dest: reg,
                    col: c.clone(),
                });
                reg
            }
            BaseExpr::Const(c) => {
                let reg = self.next_reg();
                self.instrs.push(IRInstr::LoadConst {
                    dest: reg,
                    value: c.clone(),
                });
                reg
            }
            BaseExpr::Param(p) => {
                let reg: Reg = self.next_reg();
                self.instrs.push(IRInstr::LoadParam {
                    dest: reg,
                    name: p.to_string(),
                });
                reg
            }
            BaseExpr::Add(lhs, rhs) => {
                let lhs = self.build_ir(lhs);
                let rhs = self.build_ir(rhs);
                let dest = self.next_reg();
                self.instrs.push(IRInstr::Add { dest, lhs, rhs });
                dest
            }
            BaseExpr::Sub(lhs, rhs) => {
                let lhs = self.build_ir(lhs);
                let rhs = self.build_ir(rhs);
                let dest = self.next_reg();
                self.instrs.push(IRInstr::Sub { dest, lhs, rhs });
                dest
            }
            BaseExpr::Mul(lhs, rhs) => {
                let lhs = self.build_ir(lhs);
                let rhs = self.build_ir(rhs);
                let dest = self.next_reg();
                self.instrs.push(IRInstr::Mul { dest, lhs, rhs });
                dest
            }
            BaseExpr::Neg(op) => {
                let op = self.build_ir(op);
                let dest = self.next_reg();
                self.instrs.push(IRInstr::Neg { dest, op });
                dest
            }
            BaseExpr::Inv(op) => {
                let op = self.build_ir(op);
                let dest = self.next_reg();
                self.instrs.push(IRInstr::Inv { dest, op });
                dest
            }
        };

        // store expression in cache
        self.env.insert(expr.id(), reg);
        return reg;
    }

    pub fn build_ext_ir(&mut self, expr: &ExtExpr) -> Reg4 {
        if let Some(&regs) = self.ext_env.get(&expr.id()) {
            return regs;
        }

        let dest = match expr {
            ExtExpr::SecureCol(arr) => {
                let regs = [
                    self.build_ir(&arr[0]),
                    self.build_ir(&arr[1]),
                    self.build_ir(&arr[2]),
                    self.build_ir(&arr[3]),
                ];
                let out = self.next_reg4();
                self.instrs.push(IRInstr::LoadExtCol {
                    dest: out,
                    col: regs,
                });
                out
            }
            ExtExpr::Const(c) => {
                let out = self.next_reg4();
                self.instrs.push(IRInstr::LoadExtConst {
                    dest: out,
                    value: c.clone(),
                });
                out
            }
            ExtExpr::Param(name) => {
                let out = self.next_reg4();
                self.instrs.push(IRInstr::LoadExtParam {
                    dest: out,
                    name: name.clone(),
                });
                out
            }
            ExtExpr::Add(l, r) => {
                let lreg = self.build_ext_ir(l);
                let rreg = self.build_ext_ir(r);
                let out = self.next_reg4();
                self.instrs.push(IRInstr::AddExt {
                    dest: out,
                    lhs: lreg,
                    rhs: rreg,
                });
                out
            }
            ExtExpr::Sub(l, r) => {
                let lreg = self.build_ext_ir(l);
                let rreg = self.build_ext_ir(r);
                let out = self.next_reg4();
                self.instrs.push(IRInstr::SubExt {
                    dest: out,
                    lhs: lreg,
                    rhs: rreg,
                });
                out
            }
            ExtExpr::Mul(l, r) => {
                let lreg = self.build_ext_ir(l);
                let rreg = self.build_ext_ir(r);
                let out = self.next_reg4();
                self.instrs.push(IRInstr::MulExt {
                    dest: out,
                    lhs: lreg,
                    rhs: rreg,
                });
                out
            }
            ExtExpr::Neg(op) => {
                let oreg = self.build_ext_ir(op);
                let out = self.next_reg4();
                self.instrs.push(IRInstr::NegExt {
                    dest: out,
                    op: oreg,
                });
                out
            }
        };

        self.ext_env.insert(expr.id(), dest);
        dest
    }
}
