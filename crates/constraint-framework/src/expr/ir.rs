use std::collections::HashMap;

use stwo::core::fields::m31::BaseField;

use crate::expr::{BaseExpr, ColumnExpr};

#[derive(Debug)]
// string is to store the name of the variable
pub enum IRInstr {
    LoadCol {
        dest: String,
        col: ColumnExpr,
    },
    LoadConst {
        dest: String,
        value: BaseField,
    },
    LoadParam {
        dest: String,
        name: String,
    },

    Add {
        dest: String,
        lhs: String,
        rhs: String,
    },
    Sub {
        dest: String,
        lhs: String,
        rhs: String,
    },
    Mul {
        dest: String,
        lhs: String,
        rhs: String,
    },
    Neg {
        dest: String,
        op: String,
    },
    Inv {
        dest: String,
        op: String,
    },
    Constraint {
        value: String,
    },
}

pub struct IRBuilder {
    pub instrs: Vec<IRInstr>,
    pub env: HashMap<String, String>,
    tmp_id: usize,
}

impl IRBuilder {
    pub fn new() -> Self {
        Self {
            instrs: vec![],
            env: HashMap::new(),
            tmp_id: 0,
        }
    }

    pub fn next_tmp_var(&mut self) -> String {
        let name = format!("tmp_{}", self.tmp_id);
        self.tmp_id += 1;
        name
    }

    // depth-first traverse the expression tree
    pub fn build_ir(&mut self, expr: &BaseExpr, dest: String) {
        match expr {
            BaseExpr::Col(c) => {
                self.instrs.push(IRInstr::LoadCol {
                    dest,
                    col: c.clone(),
                });
            }
            BaseExpr::Const(c) => {
                self.instrs.push(IRInstr::LoadConst {
                    dest: dest.clone(),
                    value: c.clone(),
                });
                self.env.insert(dest, c.to_string());
            }
            BaseExpr::Param(p) => {
                self.instrs.push(IRInstr::LoadParam {
                    dest: dest.clone(),
                    name: p.to_string(),
                });
                self.env.insert(dest, p.to_string());
            }
            BaseExpr::Add(lhs, rhs) => {
                let l = self.next_tmp_var();
                let r = self.next_tmp_var();
                self.build_ir(lhs, l.clone());
                self.build_ir(rhs, r.clone());
                self.instrs.push(IRInstr::Add {
                    dest: dest.clone(),
                    lhs: l,
                    rhs: r,
                });
            }
            BaseExpr::Sub(lhs, rhs) => {
                let l = self.next_tmp_var();
                let r = self.next_tmp_var();
                self.build_ir(lhs, l.clone());
                self.build_ir(rhs, r.clone());
                self.instrs.push(IRInstr::Sub {
                    dest: dest.clone(),
                    lhs: l,
                    rhs: r,
                });
            }
            BaseExpr::Mul(lhs, rhs) => {
                let l = self.next_tmp_var();
                let r = self.next_tmp_var();
                self.build_ir(lhs, l.clone());
                self.build_ir(rhs, r.clone());
                self.instrs.push(IRInstr::Mul {
                    dest: dest.clone(),
                    lhs: l,
                    rhs: r,
                });
            }
            BaseExpr::Neg(op) => {
                let o = self.next_tmp_var();
                self.build_ir(op, o.clone());
                self.instrs.push(IRInstr::Neg {
                    dest: dest.clone(),
                    op: o,
                });
            }
            BaseExpr::Inv(op) => {
                let o = self.next_tmp_var();
                self.build_ir(op, o.clone());
                self.instrs.push(IRInstr::Inv {
                    dest: dest.clone(),
                    op: o,
                });
            }
        }
    }
}
