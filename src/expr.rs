use std::fmt;

/// A snapped symbolic expression.
///
/// After training, every input selector is one-hot, so the tree collapses to
/// a concrete expression built from the variable `x`, constants, and `eml`.
/// A small set of rewrite rules can further simplify common forms into
/// `exp`, `ln`, and arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Const(f64),
    Var,
    /// The raw EML operator `eml(a, b) = exp(a) - ln(b)`.
    Eml(Box<Expr>, Box<Expr>),
    /// `exp(a)` — produced only by simplification.
    Exp(Box<Expr>),
    /// `ln(a)` — produced only by simplification.
    Ln(Box<Expr>),
    /// `a - b` — produced only by simplification.
    Sub(Box<Expr>, Box<Expr>),
}

impl Expr {
    /// Evaluate the expression at a real input `x`.
    pub fn eval(&self, x: f64) -> f64 {
        match self {
            Expr::Const(c) => *c,
            Expr::Var => x,
            Expr::Eml(a, b) => {
                let left = a.eval(x);
                let right = b.eval(x);
                eml_scalar(left, right)
            }
            Expr::Exp(a) => a.eval(x).exp(),
            Expr::Ln(a) => a.eval(x).max(1e-12).ln(),
            Expr::Sub(a, b) => a.eval(x) - b.eval(x),
        }
    }

    /// Apply a small set of simplification rules so that common recovered
    /// forms look like ordinary math.
    pub fn simplify(&self) -> Expr {
        match self {
            Expr::Eml(a, b) => {
                let a = a.simplify();
                let b = b.simplify();

                // eml(x, 1) = exp(x)
                if matches!(&b, Expr::Const(c) if *c == 1.0) {
                    return Expr::Exp(Box::new(a));
                }

                // eml(1, eml(eml(1, x), 1)) = ln(x)
                if let Expr::Eml(inner, one1) = &b
                    && matches!(one1.as_ref(), Expr::Const(1.0))
                        && let Expr::Eml(one2, x) = inner.as_ref()
                            && matches!(one2.as_ref(), Expr::Const(1.0)) {
                                return Expr::Ln(Box::new(x.simplify()));
                            }

                // eml(1, x) = e - ln(x)
                if matches!(&a, Expr::Const(1.0)) {
                    return Expr::Sub(
                        Box::new(Expr::Const(std::f64::consts::E)),
                        Box::new(Expr::Ln(Box::new(b))),
                    );
                }

                Expr::Eml(Box::new(a), Box::new(b))
            }
            Expr::Exp(a) => Expr::Exp(Box::new(a.simplify())),
            Expr::Ln(a) => Expr::Ln(Box::new(a.simplify())),
            Expr::Sub(a, b) => Expr::Sub(Box::new(a.simplify()), Box::new(b.simplify())),
            other => other.clone(),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(c) => {
                if c.fract() == 0.0 {
                    write!(f, "{}", *c as i64)
                } else {
                    write!(f, "{c}")
                }
            }
            Expr::Var => write!(f, "x"),
            Expr::Eml(a, b) => write!(f, "eml({}, {})", a, b),
            Expr::Exp(a) => write!(f, "exp({})", a),
            Expr::Ln(a) => write!(f, "ln({})", a),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
        }
    }
}

fn eml_scalar(x: f64, y: f64) -> f64 {
    let y = y.abs().max(1e-12);
    x.exp() - y.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eml_exp_identity() {
        // eml(x, 1) == exp(x)
        let e = Expr::Eml(Box::new(Expr::Var), Box::new(Expr::Const(1.0)));
        for x in [-1.0, 0.0, 0.5, 1.0] {
            assert!((e.eval(x) - x.exp()).abs() < 1e-9);
        }
    }

    #[test]
    fn eml_ln_identity() {
        // ln(x) = eml(1, eml(eml(1, x), 1))
        let e = Expr::Eml(
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Eml(
                Box::new(Expr::Eml(Box::new(Expr::Const(1.0)), Box::new(Expr::Var))),
                Box::new(Expr::Const(1.0)),
            )),
        );
        for x in [0.5, 1.0, 1.5, 2.0] {
            assert!((e.eval(x) - x.ln()).abs() < 1e-9);
        }
    }
}
