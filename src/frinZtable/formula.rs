//! Row-wise arithmetic for calculated columns. References are names in brackets
//! (e.g. [Noise-level]), simple names (Amp), or spreadsheet letters (E).
use super::data::{column_letter, number, Column};
use anyhow::{bail, Context, Result};

#[derive(Clone, Debug)]
pub enum Expression {
    Number(f64),
    Column(usize),
    Negate(Box<Self>),
    Binary(char, Box<Self>, Box<Self>),
    Function(String, Box<Self>),
}

impl Expression {
    pub fn parse(input: &str, columns: &[Column]) -> Result<Self> {
        let mut parser = Parser {
            input: input.trim().trim_start_matches('='),
            columns,
            position: 0,
        };
        let expr = parser.expression(0)?;
        parser.whitespace();
        if parser.position != parser.input.len() {
            bail!("Unexpected text at position {}", parser.position + 1);
        }
        Ok(expr)
    }
    pub fn evaluate(&self, row: &[String]) -> Option<f64> {
        let v = match self {
            Self::Number(v) => *v,
            Self::Column(c) => number(row.get(*c)?)?,
            Self::Negate(expr) => -expr.evaluate(row)?,
            Self::Binary(op, a, b) => {
                let (a, b) = (a.evaluate(row)?, b.evaluate(row)?);
                match op {
                    '+' => a + b,
                    '-' => a - b,
                    '*' => a * b,
                    '/' => a / b,
                    '^' => a.powf(b),
                    _ => return None,
                }
            }
            Self::Function(name, expr) => {
                let x = expr.evaluate(row)?;
                match name.as_str() {
                    "abs" => x.abs(),
                    "sqrt" => x.sqrt(),
                    "ln" => x.ln(),
                    "log10" => x.log10(),
                    "exp" => x.exp(),
                    "sin" => x.sin(),
                    "cos" => x.cos(),
                    _ => return None,
                }
            }
        };
        v.is_finite().then_some(v)
    }
}

struct Parser<'a> {
    input: &'a str,
    columns: &'a [Column],
    position: usize,
}
impl Parser<'_> {
    fn whitespace(&mut self) {
        while self
            .input
            .as_bytes()
            .get(self.position)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.position += 1;
        }
    }
    fn peek(&mut self) -> Option<char> {
        self.whitespace();
        self.input[self.position..].chars().next()
    }
    fn expression(&mut self, precedence: u8) -> Result<Expression> {
        let mut left = self.atom()?;
        loop {
            let Some(op) = self.peek() else {
                break;
            };
            let level = match op {
                '+' | '-' => 1,
                '*' | '/' => 2,
                '^' => 4,
                _ => break,
            };
            if level < precedence {
                break;
            }
            self.position += 1;
            let right = self.expression(if op == '^' { level } else { level + 1 })?;
            left = Expression::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }
    fn atom(&mut self) -> Result<Expression> {
        let ch = self.peek().context("Expected a value or column")?;
        match ch {
            '-' | '+' => {
                self.position += 1;
                let value = self.expression(3)?;
                Ok(if ch == '-' {
                    Expression::Negate(Box::new(value))
                } else {
                    value
                })
            }
            '(' => {
                self.position += 1;
                let value = self.expression(0)?;
                if self.peek() != Some(')') {
                    bail!("Missing closing parenthesis");
                }
                self.position += 1;
                Ok(value)
            }
            '[' => {
                self.position += 1;
                let end = self.input[self.position..]
                    .find(']')
                    .context("Missing closing bracket")?
                    + self.position;
                let name = &self.input[self.position..end];
                self.position = end + 1;
                self.reference(name)
            }
            '0'..='9' | '.' => {
                let start = self.position;
                while self
                    .input
                    .as_bytes()
                    .get(self.position)
                    .is_some_and(|c| c.is_ascii_digit() || *c == b'.')
                {
                    self.position += 1;
                }
                if self
                    .input
                    .as_bytes()
                    .get(self.position)
                    .is_some_and(|c| *c == b'e' || *c == b'E')
                {
                    self.position += 1;
                    if self
                        .input
                        .as_bytes()
                        .get(self.position)
                        .is_some_and(|c| *c == b'+' || *c == b'-')
                    {
                        self.position += 1;
                    }
                    while self
                        .input
                        .as_bytes()
                        .get(self.position)
                        .is_some_and(u8::is_ascii_digit)
                    {
                        self.position += 1;
                    }
                }
                let value = self.input[start..self.position]
                    .parse::<f64>()
                    .context("Invalid number")?;
                if !value.is_finite() {
                    bail!("Number must be finite");
                }
                Ok(Expression::Number(value))
            }
            _ => {
                let start = self.position;
                while self
                    .input
                    .as_bytes()
                    .get(self.position)
                    .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
                {
                    self.position += 1;
                }
                if start == self.position {
                    bail!("Unexpected character {ch}");
                }
                let name = self.input[start..self.position].to_owned();
                if self.peek() == Some('(') {
                    let function = name.to_lowercase();
                    if !["abs", "sqrt", "ln", "log10", "exp", "sin", "cos"]
                        .contains(&function.as_str())
                    {
                        bail!("Unknown function {name}");
                    }
                    self.position += 1;
                    let argument = self.expression(0)?;
                    if self.peek() != Some(')') {
                        bail!("Missing closing parenthesis");
                    }
                    self.position += 1;
                    Ok(Expression::Function(function, Box::new(argument)))
                } else {
                    self.reference(&name)
                }
            }
        }
    }
    fn reference(&self, name: &str) -> Result<Expression> {
        let column = self
            .columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(name))
            .or_else(|| {
                (0..self.columns.len()).find(|&i| column_letter(i).eq_ignore_ascii_case(name))
            })
            .with_context(|| format!("Unknown column {name}. Use [column name] or its letter."))?;
        Ok(Expression::Column(column))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn formulas_precedence_references_and_invalid_results() {
        let columns = vec![
            Column {
                name: "Amp".into(),
                unit: "%".into(),
            },
            Column {
                name: "Noise-level".into(),
                unit: "%".into(),
            },
        ];
        let row = vec!["4".into(), "2".into()];
        assert_eq!(
            Expression::parse("=Amp / [Noise-level] + sqrt(A) * 2", &columns)
                .unwrap()
                .evaluate(&row),
            Some(6.0)
        );
        assert_eq!(
            Expression::parse("-2^2 + 1e-3", &columns)
                .unwrap()
                .evaluate(&row),
            Some(-3.999)
        );
        assert_eq!(
            Expression::parse("2^3^2", &columns).unwrap().evaluate(&row),
            Some(512.0)
        );
        assert!(Expression::parse("Amp / 0", &columns)
            .unwrap()
            .evaluate(&row)
            .is_none());
        assert!(Expression::parse("unknown + 1", &columns).is_err());
        assert!(Expression::parse("1 +", &columns).is_err());
    }
}
