use nalgebra::{DMatrix, DVector};
use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct BaselineSolve {
    pub a: usize,     // antenna index p
    pub b: usize,     // antenna index q
    pub tau_s: f64,   // measured tau_pq = tau_p - tau_q
    pub rate_hz: f64, // measured rho_pq = rho_p - rho_q
    pub w_tau: f64,   // weight for tau (1/var)
    pub w_rate: f64,  // weight for rate (1/var)
    // Optional phase info (radians) for phase solving / closure diagnostics
    pub phase_rad: f64, // measured phi_pq in radians (-pi..pi)
    pub w_phase: f64,   // weight for phase
}

#[derive(Debug, Clone)]
pub struct AntennaSolution {
    pub tau_s: Vec<f64>,
    pub rate_hz: Vec<f64>,
}

fn build_b_matrix_with_map(
    m: usize,
    col_map: &HashMap<usize, usize>,
    rows: &[(usize, usize)],
) -> DMatrix<f64> {
    let n = col_map.len();
    let mut b = DMatrix::<f64>::zeros(m, n);
    for (i, &(p, q)) in rows.iter().enumerate() {
        if let Some(&jp) = col_map.get(&p) {
            b[(i, jp)] += 1.0;
        }
        if let Some(&jq) = col_map.get(&q) {
            b[(i, jq)] -= 1.0;
        }
    }
    b
}

fn solve_weighted_ls(b: &DMatrix<f64>, y: &DVector<f64>, w: &DVector<f64>) -> Vec<f64> {
    // Solve (B^T W B) x = B^T W y, W diagonal from w
    let m = b.nrows();
    let mut btw = DMatrix::<f64>::zeros(b.ncols(), m);
    for i in 0..m {
        let wi = w[i];
        // btw col i = wi * b_row_i^T
        let col = b.row(i).transpose() * wi;
        btw.set_column(i, &col);
    }
    let ata = &btw * b; // n x n
    let atb = &btw * y; // n x 1
    if let Some(ch) = ata.clone().cholesky() {
        let sol = ch.solve(&atb);
        sol.as_slice().to_vec()
    } else {
        ata.lu().solve(&atb).unwrap().as_slice().to_vec()
    }
}

pub fn solve_antenna_tau_rate(
    num_ant: usize,
    ref_ant: usize,
    solves: &[BaselineSolve],
) -> AntennaSolution {
    if solves.is_empty() {
        eprintln!("#WARN: No baselines provided to LS; returning zeros.");
        return AntennaSolution {
            tau_s: vec![0.0; num_ant],
            rate_hz: vec![0.0; num_ant],
        };
    }

    // Build connectivity graph and find component containing ref_ant
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for s in solves {
        adj.entry(s.a).or_default().push(s.b);
        adj.entry(s.b).or_default().push(s.a);
    }
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack = vec![ref_ant];
    while let Some(u) = stack.pop() {
        if visited.insert(u) {
            if let Some(nei) = adj.get(&u) {
                for &v in nei {
                    if !visited.contains(&v) {
                        stack.push(v);
                    }
                }
            }
        }
    }

    // Count components (for diagnostics)
    let mut all_ants: BTreeSet<usize> = BTreeSet::new();
    for s in solves {
        all_ants.insert(s.a);
        all_ants.insert(s.b);
    }
    let _total_ants_seen = all_ants.len();
    let mut component_count = 0usize;
    let mut seen_comp: HashSet<usize> = HashSet::new();
    for &a in &all_ants {
        if seen_comp.contains(&a) {
            continue;
        }
        // BFS from a
        component_count += 1;
        let mut st = vec![a];
        while let Some(u) = st.pop() {
            if seen_comp.insert(u) {
                if let Some(nei) = adj.get(&u) {
                    for &v in nei {
                        if !seen_comp.contains(&v) {
                            st.push(v);
                        }
                    }
                }
            }
        }
    }
    if component_count > 1 {
        eprintln!("#WARN: Detected {} disconnected component(s) in baseline graph; using only the one containing reference {}.", component_count, ref_ant);
    }

    // Filter baselines to those fully inside the ref component
    let mut filtered: Vec<&BaselineSolve> = solves
        .iter()
        .filter(|s| visited.contains(&s.a) && visited.contains(&s.b))
        .collect();
    let mut anchor_ant = ref_ant;
    let mut ignored = solves.len().saturating_sub(filtered.len());
    if filtered.is_empty() {
        // Fallback: choose the largest connected component and anchor it to its smallest antenna index
        // Build all components
        let mut comps: Vec<BTreeSet<usize>> = Vec::new();
        let mut seen_all: HashSet<usize> = HashSet::new();
        for &a in &all_ants {
            if seen_all.contains(&a) {
                continue;
            }
            let mut comp: BTreeSet<usize> = BTreeSet::new();
            let mut st = vec![a];
            while let Some(u) = st.pop() {
                if !seen_all.insert(u) {
                    continue;
                }
                comp.insert(u);
                if let Some(nei) = adj.get(&u) {
                    for &v in nei {
                        if !seen_all.contains(&v) {
                            st.push(v);
                        }
                    }
                }
            }
            comps.push(comp);
        }
        if let Some(best) = comps.into_iter().max_by_key(|c| c.len()) {
            let mut best_set: HashSet<usize> = HashSet::new();
            for &x in &best {
                best_set.insert(x);
            }
            filtered = solves
                .iter()
                .filter(|s| best_set.contains(&s.a) && best_set.contains(&s.b))
                .collect();
            anchor_ant = *best.iter().next().unwrap_or(&ref_ant);
            ignored = solves.len().saturating_sub(filtered.len());
            eprintln!("#WARN: Reference {} not connected; re-anchoring to antenna {} in largest component (ignored {} baseline(s)).", ref_ant, anchor_ant, ignored);
        } else {
            eprintln!("#WARN: No usable baselines; returning zeros.");
            return AntennaSolution {
                tau_s: vec![0.0; num_ant],
                rate_hz: vec![0.0; num_ant],
            };
        }
    } else if ignored > 0 {
        eprintln!(
            "#WARN: Ignoring {} baseline(s) not connected to reference {}.",
            ignored, ref_ant
        );
    }

    // Build compact column map for unknown antennas in the ref component (excluding ref)
    let mut col_map: HashMap<usize, usize> = HashMap::new();
    let mut idx = 0usize;
    for &a in &all_ants {
        if a == anchor_ant {
            continue;
        }
        if visited.contains(&a) {
            col_map.insert(a, idx);
            idx += 1;
        }
    }
    let n_unknowns = col_map.len();
    let m_rows = filtered.len();
    if n_unknowns == 0 {
        // Only ref antenna is present
        return AntennaSolution {
            tau_s: vec![0.0; num_ant],
            rate_hz: vec![0.0; num_ant],
        };
    }

    // Build rows (a,b) for filtered set and corresponding y, w
    let rows: Vec<(usize, usize)> = filtered.iter().map(|s| (s.a, s.b)).collect();
    let b = build_b_matrix_with_map(m_rows, &col_map, &rows);

    // tau
    let y_tau = DVector::from_iterator(m_rows, filtered.iter().map(|s| s.tau_s));
    let w_tau = DVector::from_iterator(m_rows, filtered.iter().map(|s| s.w_tau.max(1e-20)));
    let x_tau = solve_weighted_ls(&b, &y_tau, &w_tau);

    // rate
    let y_rate = DVector::from_iterator(m_rows, filtered.iter().map(|s| s.rate_hz));
    let w_rate = DVector::from_iterator(m_rows, filtered.iter().map(|s| s.w_rate.max(1e-20)));
    let x_rate = solve_weighted_ls(&b, &y_rate, &w_rate);

    // Diagnostics: degrees of freedom and condition number (sqrt(W)*B)
    let dof_tau = (m_rows as isize) - (n_unknowns as isize);
    if dof_tau < 0 {
        eprintln!(
            "#WARN: Negative degrees of freedom ({}): m={} < n={}",
            dof_tau, m_rows, n_unknowns
        );
    }
    // Build sqrt(W) * B for cond estimate
    let mut bw = DMatrix::<f64>::zeros(m_rows, n_unknowns);
    for i in 0..m_rows {
        let wi = w_tau[i].sqrt();
        for j in 0..n_unknowns {
            bw[(i, j)] = wi * b[(i, j)];
        }
    }
    {
        let svd = bw.svd(true, false);
        let sv = svd.singular_values;
        if sv.len() >= 1 {
            let mut smax = 0.0f64;
            let mut smin = f64::INFINITY;
            for v in sv.iter().cloned() {
                if v > smax {
                    smax = v;
                }
                if v > 1e-14 && v < smin {
                    smin = v;
                }
            }
            if smin.is_finite() {
                let cond = smax / smin;
                if cond > 1.0e8 {
                    eprintln!("#WARN: Ill-conditioned system: cond(B) ~ {:.3e}", cond);
                }
            } else {
                eprintln!("#WARN: Singular system detected (no positive singular values)");
            }
        }
    }

    // Expand back to full antenna vectors; unknown/outside-component set to 0
    let mut tau = vec![0.0f64; num_ant];
    let mut rate = vec![0.0f64; num_ant];
    for (&ant, &j) in &col_map {
        tau[ant] = x_tau[j];
        rate[ant] = x_rate[j];
    }
    // anchor remains 0 by definition; ants not in component remain 0 but are unconstrained
    AntennaSolution {
        tau_s: tau,
        rate_hz: rate,
    }
}

pub fn solve_antenna_phase(num_ant: usize, ref_ant: usize, solves: &[BaselineSolve]) -> Vec<f64> {
    use num_complex::Complex;
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for s in solves {
        if s.w_phase > 0.0 {
            adj.entry(s.a).or_default().push(s.b);
            adj.entry(s.b).or_default().push(s.a);
        }
    }
    // Component containing reference
    let mut visited: HashSet<usize> = HashSet::new();
    let mut st = vec![ref_ant];
    while let Some(u) = st.pop() {
        if visited.insert(u) {
            if let Some(nei) = adj.get(&u) {
                for &v in nei {
                    if !visited.contains(&v) {
                        st.push(v);
                    }
                }
            }
        }
    }
    if visited.len() <= 1 {
        return vec![0.0f64; num_ant];
    }

    // Map ants in component to [0..k)
    let mut comp_ants: Vec<usize> = visited.iter().cloned().collect();
    comp_ants.sort_unstable();
    let mut idx_of: HashMap<usize, usize> = HashMap::new();
    for (i, &a) in comp_ants.iter().enumerate() {
        idx_of.insert(a, i);
    }
    let k = comp_ants.len();
    // Build Hermitian matrix M (k x k)
    let mut m = DMatrix::<Complex<f64>>::from_element(k, k, Complex::new(0.0, 0.0));
    for s in solves {
        if s.w_phase <= 0.0 {
            continue;
        }
        if let (Some(&ip), Some(&iq)) = (idx_of.get(&s.a), idx_of.get(&s.b)) {
            let w = s.w_phase.max(1e-12);
            let h = Complex::new(0.0, s.phase_rad).exp(); // e^{i phi}
            m[(ip, iq)] += h * w;
            m[(iq, ip)] += h.conj() * w;
        }
    }
    // Power iteration to get principal eigenvector
    let mut v = DVector::<Complex<f64>>::from_element(k, Complex::new(1.0, 0.0));
    for _ in 0..50 {
        v = &m * &v;
        // normalize
        let norm = v
            .iter()
            .map(|z| z.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(1e-12);
        v.unscale_mut(norm);
    }
    // Extract phases, anchor to reference
    let mut out = vec![0.0f64; num_ant];
    let ref_phase = if let Some(&ir) = idx_of.get(&ref_ant) {
        v[ir].arg()
    } else {
        0.0
    };
    let wrap = |x: f64| -> f64 {
        let mut y = (x + std::f64::consts::PI) % (2.0 * std::f64::consts::PI);
        if y < 0.0 {
            y += 2.0 * std::f64::consts::PI;
        }
        y - std::f64::consts::PI
    };
    for (&a, i) in comp_ants.iter().zip(0..) {
        out[a] = wrap(v[i].arg() - ref_phase);
    }
    out
}
