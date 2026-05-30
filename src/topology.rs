//! Topology generation: build adjacency matrices for various graph topologies.

use rand::Rng;
use std::fmt;

/// Supported graph topologies.
#[derive(Debug, Clone)]
pub enum Topology {
    /// All nodes connected to a single hub (node 0).
    Star,
    /// Linear chain: node i connected to i+1.
    Chain,
    /// Circular ring: each node connected to its two neighbors.
    Ring,
    /// Full mesh / complete graph: every node connected to every other.
    Mesh,
    /// Watts–Strogatz-style small-world with rewiring probability `p`.
    SmallWorld(f64),
    /// Erdős–Rényi random graph with edge probability `p`.
    Random(f64),
    /// Custom topology from a provided adjacency matrix.
    Custom(Vec<Vec<f64>>),
}

impl fmt::Display for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Topology::Star => write!(f, "Star"),
            Topology::Chain => write!(f, "Chain"),
            Topology::Ring => write!(f, "Ring"),
            Topology::Mesh => write!(f, "Mesh"),
            Topology::SmallWorld(p) => write!(f, "SmallWorld({p:.2})"),
            Topology::Random(p) => write!(f, "Random({p:.2})"),
            Topology::Custom(_) => write!(f, "Custom"),
        }
    }
}

/// Generate an adjacency matrix for the given topology with `n` nodes.
///
/// Returns a dense `n × n` adjacency matrix where `adj[i][j] > 0.0` indicates an edge.
/// The matrix is symmetric (undirected) with `adj[i][i] == 0.0`.
pub fn generate(topology: &Topology, n: usize) -> Vec<Vec<f64>> {
    match topology {
        Topology::Star => generate_star(n),
        Topology::Chain => generate_chain(n),
        Topology::Ring => generate_ring(n),
        Topology::Mesh => generate_mesh(n),
        Topology::SmallWorld(p) => generate_small_world(n, *p),
        Topology::Random(p) => generate_random(n, *p),
        Topology::Custom(adj) => {
            assert_eq!(adj.len(), n, "Custom adjacency matrix must be {n}×{n}");
            adj.clone()
        }
    }
}

/// Validate that an adjacency matrix is square, symmetric, and has zero diagonal.
pub fn validate(adj: &[Vec<f64>]) -> Result<(), String> {
    let n = adj.len();
    for (i, row) in adj.iter().enumerate() {
        if row.len() != n {
            return Err(format!("Row {i} has length {}, expected {n}", row.len()));
        }
        if row[i] != 0.0 {
            return Err(format!("Diagonal element [{i}][{i}] is {}, expected 0.0", row[i]));
        }
        for (j, &val) in row.iter().enumerate() {
            if val < 0.0 {
                return Err(format!("Element [{i}][{j}] is negative: {val}"));
            }
            if (val - adj[j][i]).abs() > 1e-10 {
                return Err(format!(
                    "Asymmetric at [{i}][{j}]: {val} vs [{j}][{i}]: {}",
                    adj[j][i]
                ));
            }
        }
    }
    Ok(())
}

fn zeros(n: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; n]; n]
}

fn set_sym(adj: &mut [Vec<f64>], i: usize, j: usize, w: f64) {
    adj[i][j] = w;
    adj[j][i] = w;
}

fn generate_star(n: usize) -> Vec<Vec<f64>> {
    let mut adj = zeros(n);
    if n > 1 {
        for i in 1..n {
            set_sym(&mut adj, 0, i, 1.0);
        }
    }
    adj
}

fn generate_chain(n: usize) -> Vec<Vec<f64>> {
    let mut adj = zeros(n);
    if n > 1 {
        for i in 0..n - 1 {
            set_sym(&mut adj, i, i + 1, 1.0);
        }
    }
    adj
}

fn generate_ring(n: usize) -> Vec<Vec<f64>> {
    let mut adj = zeros(n);
    if n > 1 {
        for i in 0..n {
            set_sym(&mut adj, i, (i + 1) % n, 1.0);
        }
    }
    adj
}

fn generate_mesh(n: usize) -> Vec<Vec<f64>> {
    let mut adj = zeros(n);
    for i in 0..n {
        for j in i + 1..n {
            set_sym(&mut adj, i, j, 1.0);
        }
    }
    adj
}

fn generate_small_world(n: usize, p: f64) -> Vec<Vec<f64>> {
    // Start from a ring with K=2 nearest neighbors on each side, then rewire.
    let mut adj = zeros(n);
    if n <= 1 {
        return adj;
    }
    let k = 2usize.min(n / 2);
    // Build initial ring lattice
    for i in 0..n {
        for dk in 1..=k {
            let j = (i + dk) % n;
            set_sym(&mut adj, i, j, 1.0);
        }
    }
    // Rewire edges with probability p
    let mut rng = rand::thread_rng();
    for i in 0..n {
        for j in i + 1..n {
            if adj[i][j] > 0.0 && rng.gen::<f64>() < p {
                // Remove edge (i, j) and add edge (i, random) — avoid self-loops and duplicates
                set_sym(&mut adj, i, j, 0.0);
                let mut target = rng.gen_range(0..n);
                let mut attempts = 0;
                while (target == i || adj[i][target] > 0.0) && attempts < n {
                    target = rng.gen_range(0..n);
                    attempts += 1;
                }
                if target != i && adj[i][target] == 0.0 {
                    set_sym(&mut adj, i, target, 1.0);
                } else {
                    // Fallback: restore original edge
                    set_sym(&mut adj, i, j, 1.0);
                }
            }
        }
    }
    adj
}

fn generate_random(n: usize, p: f64) -> Vec<Vec<f64>> {
    let mut adj = zeros(n);
    let mut rng = rand::thread_rng();
    for i in 0..n {
        for j in i + 1..n {
            if rng.gen::<f64>() < p {
                set_sym(&mut adj, i, j, 1.0);
            }
        }
    }
    adj
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge_count(adj: &[Vec<f64>]) -> usize {
        let n = adj.len();
        let mut count = 0;
        for i in 0..n {
            for j in i + 1..n {
                if adj[i][j] > 0.0 {
                    count += 1;
                }
            }
        }
        count
    }

    #[test]
    fn test_star_edges() {
        let adj = generate(&Topology::Star, 5);
        assert_eq!(edge_count(&adj), 4);
        assert!(validate(&adj).is_ok());
    }

    #[test]
    fn test_chain_edges() {
        let adj = generate(&Topology::Chain, 5);
        assert_eq!(edge_count(&adj), 4);
        assert!(validate(&adj).is_ok());
    }

    #[test]
    fn test_ring_edges() {
        let adj = generate(&Topology::Ring, 5);
        assert_eq!(edge_count(&adj), 5);
        assert!(validate(&adj).is_ok());
    }

    #[test]
    fn test_ring_edges_even() {
        let adj = generate(&Topology::Ring, 6);
        assert_eq!(edge_count(&adj), 6);
    }

    #[test]
    fn test_mesh_edges() {
        let adj = generate(&Topology::Mesh, 4);
        assert_eq!(edge_count(&adj), 6); // C(4,2) = 6
        assert!(validate(&adj).is_ok());
    }

    #[test]
    fn test_mesh_complete() {
        let n = 10;
        let adj = generate(&Topology::Mesh, n);
        assert_eq!(edge_count(&adj), n * (n - 1) / 2);
    }

    #[test]
    fn test_random_expected_density() {
        // With p=1.0, random should be a complete graph
        let adj = generate(&Topology::Random(1.0), 5);
        assert_eq!(edge_count(&adj), 10);
    }

    #[test]
    fn test_random_zero_density() {
        let adj = generate(&Topology::Random(0.0), 5);
        assert_eq!(edge_count(&adj), 0);
    }

    #[test]
    fn test_custom_topology() {
        let custom = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];
        let adj = generate(&Topology::Custom(custom.clone()), 3);
        assert_eq!(adj, custom);
    }

    #[test]
    fn test_single_node() {
        for topo in &[
            Topology::Star,
            Topology::Chain,
            Topology::Ring,
            Topology::Mesh,
        ] {
            let adj = generate(topo, 1);
            assert_eq!(adj, vec![vec![0.0]]);
        }
    }

    #[test]
    fn test_two_nodes_star() {
        let adj = generate(&Topology::Star, 2);
        assert_eq!(edge_count(&adj), 1);
        assert_eq!(adj[0][1], 1.0);
    }

    #[test]
    fn test_validate_rejects_non_square() {
        let bad = vec![vec![0.0, 1.0], vec![1.0]];
        assert!(validate(&bad).is_err());
    }

    #[test]
    fn test_validate_rejects_non_symmetric() {
        let bad = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
        assert!(validate(&bad).is_err());
    }

    #[test]
    fn test_validate_rejects_nonzero_diagonal() {
        let bad = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        assert!(validate(&bad).is_err());
    }

    #[test]
    fn test_validate_rejects_negative() {
        let bad = vec![vec![0.0, -1.0], vec![-1.0, 0.0]];
        assert!(validate(&bad).is_err());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Topology::Star), "Star");
        assert_eq!(format!("{}", Topology::SmallWorld(0.3)), "SmallWorld(0.30)");
    }
}
