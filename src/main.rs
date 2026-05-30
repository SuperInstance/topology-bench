//! CLI entry point for `topology-bench`.

use clap::Parser;
use topology_bench::{metrics, Topology};

#[derive(Parser, Debug)]
#[command(name = "topology-bench", version, about = "Benchmark graph topologies")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Compare multiple topologies side by side
    Compare {
        /// Number of nodes
        #[arg(long, default_value_t = 50)]
        nodes: usize,

        /// Comma-separated list of topologies: star,chain,ring,mesh,smallworld,random
        #[arg(long, default_value = "star,ring,mesh")]
        topologies: String,

        /// Rewiring probability for small-world (default: 0.3)
        #[arg(long, default_value_t = 0.3)]
        small_world_p: f64,

        /// Edge probability for random graph (default: 0.1)
        #[arg(long, default_value_t = 0.1)]
        random_p: f64,
    },
    /// Benchmark a single topology
    Single {
        /// Topology type: star, chain, ring, mesh, smallworld, random
        topology: String,

        /// Number of nodes
        #[arg(long, default_value_t = 50)]
        nodes: usize,

        /// Rewiring probability for small-world
        #[arg(long, default_value_t = 0.3)]
        small_world_p: f64,

        /// Edge probability for random graph
        #[arg(long, default_value_t = 0.1)]
        random_p: f64,
    },
}

fn parse_topology(name: &str, sw_p: f64, rand_p: f64) -> Option<Topology> {
    match name.to_lowercase().as_str() {
        "star" => Some(Topology::Star),
        "chain" => Some(Topology::Chain),
        "ring" => Some(Topology::Ring),
        "mesh" => Some(Topology::Mesh),
        "smallworld" | "small_world" => Some(Topology::SmallWorld(sw_p)),
        "random" => Some(Topology::Random(rand_p)),
        _ => None,
    }
}

fn print_report(report: &metrics::TopologyReport) {
    println!(
        "  {:<15} {:>6} {:>6} {:>10} {:>12} {:>12} {:>12} {:>12}",
        report.topology_name,
        report.node_count,
        report.edge_count,
        format!("{:?}", report.diameter),
        format!("{:.4}", report.avg_path_length.unwrap_or(f64::NAN)),
        format!("{:.4}", report.global_clustering_coefficient),
        format!("{:.4}", report.spectral_gap),
        format!("{:.1}", report.conservation_speed),
    );
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compare {
            nodes,
            topologies,
            small_world_p,
            random_p,
        } => {
            let names: Vec<&str> = topologies.split(',').map(|s| s.trim()).collect();
            let topos: Vec<Topology> = names
                .iter()
                .filter_map(|&name| parse_topology(name, small_world_p, random_p))
                .collect();

            if topos.is_empty() {
                eprintln!("No valid topologies specified. Use: star,chain,ring,mesh,smallworld,random");
                std::process::exit(1);
            }

            println!("Topology Benchmark Comparison (nodes={nodes})");
            println!("{}", "─".repeat(95));
            println!(
                "  {:<15} {:>6} {:>6} {:>10} {:>12} {:>12} {:>12} {:>12}",
                "Topology", "Nodes", "Edges", "Diameter", "AvgPathLen", "Clustering", "SpectralGap", "ConvSpeed"
            );
            println!("{}", "─".repeat(95));

            for topo in &topos {
                let report = metrics::benchmark(topo, nodes);
                print_report(&report);
            }
            println!("{}", "─".repeat(95));
        }
        Commands::Single {
            topology,
            nodes,
            small_world_p,
            random_p,
        } => {
            let topo = match parse_topology(&topology, small_world_p, random_p) {
                Some(t) => t,
                None => {
                    eprintln!("Unknown topology: {topology}");
                    std::process::exit(1);
                }
            };
            let report = metrics::benchmark(&topo, nodes);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
    }
}
