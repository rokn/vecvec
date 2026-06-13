//! `vecvec` — the admin/client CLI.
//!
//! Scaffold only at M0; clap subcommands (create-collection, upsert, query,
//! commit/branch/tag/diff/restore, versions, export) arrive with the API in M13.

fn main() {
    println!(
        "vecvec {} — scaffold (commands land in M13)",
        vecvec_core::VERSION
    );
}
