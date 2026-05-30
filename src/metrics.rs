//! Benchmark metrics for graph topologies.

use crate::topology;
use nalgebra::DMatrix;
use std::collections::VecDeque;

/// Report containing all computed metrics for a topology.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TopologyReport {
    pub topology_name: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub diameter: Option<usize>,
    pub avg_path_length: Option<f64>,
    pub global_clustering_coefficient: f64,
    pub spectral_gap: f64,
    pub conservation_speed: f64,
}

/// Compute all metrics for a given topology at `n` nodes.
pub fn benchmark(topo: &topology::Topology, n: usize) -> TopologyReport {
    let adj = topology::generate(topo, n);
    let edge_count = count_edges(&adj);
    let (diameter, avg_pl) = path_metrics(&adj);
    let clustering = global_clustering_coefficient(&adj);
    let spectral = spectral_gap(&adj);
    let conservation = conservation_speed(&adj);

    TopologyReport {
        topology_name: topo.to_string(),
        node_count: n,
        edge_count,
        diameter,
        avg_path_length: avg_pl,
        global_clustering_coefficient: clustering,
        spectral_gap: spectral,
        conservation_speed: conservation,
    }
}

fn count_edges(adj: &[Vec<f64>]) -> usize {
    let mut c = 0;
    for (i, row) in adj.iter().enumerate() {
        for &val in &row[i + 1..] {
            if val > 0.0 {
                c += 1;
            }
        }
    }
    c
}

/// BFS shortest paths from `source`. Returns distances (None = unreachable).
fn bfs_distances(adj: &[Vec<f64>], source: usize) -> Vec<Option<usize>> {
    let n = adj.len();
    let mut dist = vec![None; n];
    dist[source] = Some(0);
    let mut queue = VecDeque::new();
    queue.push_back(source);
    while let Some(u) = queue.pop_front() {
        let d = dist[u].unwrap();
        for v in 0..n {
            if adj[u][v] > 0.0 && dist[v].is_none() {
                dist[v] = Some(d + 1);
                queue.push_back(v);
            }
        }
    }
    dist
}

/// Compute diameter and average path length.
///
/// Returns `None` for both if the graph is disconnected.
fn path_metrics(adj: &[Vec<f64>]) -> (Option<usize>, Option<f64>) {
    let n = adj.len();
    if n <= 1 {
        return (Some(0), Some(0.0));
    }

    let mut max_dist = 0usize;
    let mut total_dist = 0usize;
    let mut pair_count = 0usize;

    for s in 0..n {
        let dists = bfs_distances(adj, s);
        for (t, d) in dists.iter().enumerate() {
            if t != s {
                pair_count += 1;
                match d {
                    Some(d) => {
                        max_dist = max_dist.max(*d);
                        total_dist += d;
                    }
                    None => {
                        // Disconnected
                        return (None, None);
                    }
                }
            }
        }
    }

    let avg = if pair_count > 0 {
        total_dist as f64 / pair_count as f64
    } else {
        0.0
    };
    (Some(max_dist), Some(avg))
}

/// Compute the local clustering coefficient for node `v`.
pub fn local_clustering_coefficient(adj: &[Vec<f64>], v: usize) -> f64 {
    let n = adj.len();
    let neighbors: Vec<usize> = (0..n).filter(|&u| adj[v][u] > 0.0).collect();
    let k = neighbors.len();
    if k < 2 {
        return 0.0;
    }

    let mut links = 0usize;
    for i in 0..k {
        for j in i + 1..k {
            if adj[neighbors[i]][neighbors[j]] > 0.0 {
                links += 1;
            }
        }
    }

    2.0 * links as f64 / (k * (k - 1)) as f64
}

/// Compute the global clustering coefficient (average of local coefficients).
pub fn global_clustering_coefficient(adj: &[Vec<f64>]) -> f64 {
    let n = adj.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = (0..n)
        .map(|v| local_clustering_coefficient(adj, v))
        .sum();
    sum / n as f64
}

/// Compute the spectral gap: difference between the two largest eigenvalues
/// of the adjacency matrix.
pub fn spectral_gap(adj: &[Vec<f64>]) -> f64 {
    let n = adj.len();
    if n < 2 {
        return 0.0;
    }

    let data: Vec<f64> = adj.iter().flat_map(|row| row.iter().copied()).collect();
    let matrix = DMatrix::from_row_slice(n, n, &data);

    // Use symmetric eigenvalue decomposition
    let sym = (matrix.clone() + &matrix.transpose()) * 0.5;
    let eig = sym.symmetric_eigenvalues();

    let mut eigenvalues: Vec<f64> = eig.iter().copied().collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    if eigenvalues.len() >= 2 {
        eigenvalues[0] - eigenvalues[1]
    } else {
        0.0
    }
}

/// Estimate conservation speed: how fast a value distributes to equilibrium.
///
/// We simulate a diffusion process where each node's value is the average of
/// its neighbors' values (plus itself). We measure how many steps until the
/// standard deviation across nodes drops below `threshold` (default 0.01)
/// relative to the initial value.
///
/// Returns the number of steps (lower = faster convergence).
pub fn conservation_speed(adj: &[Vec<f64>]) -> f64 {
    conservation_speed_with_threshold(adj, 0.01, 1000)
}

/// Conservation speed with configurable threshold and max iterations.
pub fn conservation_speed_with_threshold(
    adj: &[Vec<f64>],
    threshold: f64,
    max_steps: usize,
) -> f64 {
    let n = adj.len();
    if n <= 1 {
        return 0.0;
    }

    // Initial state: all value concentrated at node 0
    let mut values = vec![0.0; n];
    values[0] = 1.0;

    // Precompute neighbor lists
    let neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| adj[i][j] > 0.0)
                .collect()
        })
        .collect();

    // Degree of each node
    let degree: Vec<usize> = neighbors.iter().map(|nb| nb.len()).collect();

    // Target: uniform distribution
    let _target = 1.0 / n as f64;
    let initial_std = (1.0 / n as f64).sqrt(); // std of [1.0, 0.0, ..., 0.0]

    for step in 1..=max_steps {
        let mut new_values = vec![0.0; n];
        for i in 0..n {
            // Value at i becomes average of self + neighbors
            let deg = degree[i];
            let sum: f64 = values[i]
                + neighbors[i].iter().map(|&j| values[j]).sum::<f64>();
            new_values[i] = sum / (deg + 1) as f64;
        }
        values = new_values;

        // Compute std deviation
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        if std / initial_std < threshold {
            return step as f64;
        }
    }

    // Didn't converge within max_steps
    max_steps as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::Topology;

    #[test]
    fn test_star_diameter() {
        let adj = topology::generate(&Topology::Star, 5);
        let (diam, _) = path_metrics(&adj);
        assert_eq!(diam, Some(2));
    }

    #[test]
    fn test_chain_diameter() {
        let adj = topology::generate(&Topology::Chain, 5);
        let (diam, _) = path_metrics(&adj);
        assert_eq!(diam, Some(4));
    }

    #[test]
    fn test_ring_diameter() {
        let adj = topology::generate(&Topology::Ring, 6);
        let (diam, _) = path_metrics(&adj);
        assert_eq!(diam, Some(3));
    }

    #[test]
    fn test_mesh_diameter() {
        let adj = topology::generate(&Topology::Mesh, 5);
        let (diam, _) = path_metrics(&adj);
        assert_eq!(diam, Some(1));
    }

    #[test]
    fn test_avg_path_length_mesh() {
        let adj = topology::generate(&Topology::Mesh, 4);
        let (_, avg) = path_metrics(&adj);
        assert_eq!(avg, Some(1.0));
    }

    #[test]
    fn test_clustering_mesh() {
        let adj = topology::generate(&Topology::Mesh, 5);
        let cc = global_clustering_coefficient(&adj);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clustering_star() {
        let adj = topology::generate(&Topology::Star, 5);
        let cc = global_clustering_coefficient(&adj);
        assert!(cc < 0.01);
    }

    #[test]
    fn test_clustering_ring() {
        // Ring: each node has 2 neighbors, which are NOT connected to each other
        // So clustering coefficient = 0
        let adj = topology::generate(&Topology::Ring, 6);
        let cc = global_clustering_coefficient(&adj);
        assert!(cc < 0.01);
    }

    #[test]
    fn test_spectral_gap_positive() {
        let adj = topology::generate(&Topology::Mesh, 5);
        let gap = spectral_gap(&adj);
        assert!(gap > 0.0);
    }

    #[test]
    fn test_spectral_gap_star() {
        let adj = topology::generate(&Topology::Star, 5);
        let gap = spectral_gap(&adj);
        assert!(gap > 0.0);
    }

    #[test]
    fn test_conservation_speed_star() {
        let adj = topology::generate(&Topology::Star, 10);
        let speed = conservation_speed(&adj);
        // Star should converge reasonably fast
        assert!(speed > 0.0 && speed < 500.0);
    }

    #[test]
    fn test_conservation_speed_mesh_fast() {
        let adj = topology::generate(&Topology::Mesh, 5);
        let speed = conservation_speed(&adj);
        // Complete graph converges very fast
        assert!(speed < 20.0);
    }

    #[test]
    fn test_benchmark_star() {
        let report = benchmark(&Topology::Star, 10);
        assert_eq!(report.topology_name, "Star");
        assert_eq!(report.node_count, 10);
        assert_eq!(report.edge_count, 9);
        assert_eq!(report.diameter, Some(2));
    }

    #[test]
    fn test_benchmark_chain() {
        let report = benchmark(&Topology::Chain, 5);
        assert_eq!(report.diameter, Some(4));
    }

    #[test]
    fn test_local_clustering_triangle() {
        // Triangle: 3 nodes all connected
        let adj = topology::generate(&Topology::Mesh, 3);
        let cc = local_clustering_coefficient(&adj, 0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_disconnected_graph() {
        // Two separate nodes with no edges — disconnected
        let adj = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let (diam, avg) = path_metrics(&adj);
        assert_eq!(diam, None);
        assert_eq!(avg, None);
    }

    #[test]
    fn test_single_node_metrics() {
        let adj = topology::generate(&Topology::Star, 1);
        let (diam, avg) = path_metrics(&adj);
        assert_eq!(diam, Some(0));
        assert_eq!(avg, Some(0.0));
    }
}
