//! `topology-bench` — Benchmark graph topologies for the Grand Pattern ecosystem.
//!
//! This crate provides topology generation, benchmark metrics, and comparison tools
//! for analyzing graph structures.

pub mod metrics;
pub mod topology;

pub use metrics::TopologyReport;
pub use topology::Topology;
