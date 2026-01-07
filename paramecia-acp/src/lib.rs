//! Paramecia ACP (Agent Communication Protocol) implementation
//!
//! This crate provides the ACP server implementation for Paramecia,
//! allowing communication between the Paramecia agent and external clients
//! using the Agent Communication Protocol.

pub mod agent;
pub mod error;
pub mod tools;
pub mod types;

pub use agent::*;
pub use error::*;
pub use types::*;
