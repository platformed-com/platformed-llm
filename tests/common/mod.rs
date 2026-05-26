//! Shared helpers for integration tests.
//!
//! Each `tests/*.rs` file is a separate test binary, so this module is
//! reached via `mod common;` inside individual test files.

#![allow(dead_code)] // Each test binary uses a different subset of helpers.

pub mod test_models;
