//! `vecvec` — the admin/client CLI.
//!
//! Drives a running server over gRPC (create / upsert / query / commit / versions)
//! and operates on a collection's on-disk directory for backup (export / import).

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use vecvec_proto::pb;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Parser)]
#[command(name = "vecvec", version, about = "vecvec vector database CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a collection.
    Create {
        #[arg(long, default_value = "http://127.0.0.1:6334")]
        addr: String,
        #[arg(long)]
        name: String,
        #[arg(long)]
        dim: u32,
        #[arg(long, default_value = "cosine")]
        metric: String,
    },
    /// Upsert vectors from a JSONL file (one `{"vector":[...],"payload":{...}}` per line).
    Upsert {
        #[arg(long, default_value = "http://127.0.0.1:6334")]
        addr: String,
        #[arg(long)]
        collection: String,
        #[arg(long)]
        file: PathBuf,
    },
    /// Atomic mixed write: upsert from a JSONL file and/or delete ids, applied as
    /// one indivisible batch, optionally committing a version after (transaction-like).
    Batch {
        #[arg(long, default_value = "http://127.0.0.1:6334")]
        addr: String,
        #[arg(long)]
        collection: String,
        /// JSONL file of vectors to upsert (same format as `upsert`).
        #[arg(long)]
        file: Option<PathBuf>,
        /// Comma-separated point ids to delete (e.g. `--delete 1,2,3`).
        #[arg(long, value_delimiter = ',')]
        delete: Vec<u64>,
        /// Commit a new version after the batch is applied.
        #[arg(long)]
        commit: bool,
        #[arg(long)]
        message: Option<String>,
        #[arg(long)]
        tag: Option<String>,
    },
    /// Nearest-neighbor query with a comma-separated vector.
    Query {
        #[arg(long, default_value = "http://127.0.0.1:6334")]
        addr: String,
        #[arg(long)]
        collection: String,
        #[arg(long)]
        vector: String,
        #[arg(long, default_value_t = 10)]
        k: u32,
    },
    /// Commit the working state as a new version.
    Commit {
        #[arg(long, default_value = "http://127.0.0.1:6334")]
        addr: String,
        #[arg(long)]
        collection: String,
        #[arg(long)]
        message: Option<String>,
        #[arg(long)]
        tag: Option<String>,
    },
    /// List a collection's versions.
    Versions {
        #[arg(long, default_value = "http://127.0.0.1:6334")]
        addr: String,
        #[arg(long)]
        collection: String,
    },
    /// Export a collection directory to a tar backup.
    Export {
        /// The collection directory (e.g. <data>/collections/<name>).
        #[arg(long)]
        dir: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Import a tar backup into a directory.
    Import {
        #[arg(long = "in")]
        input: PathBuf,
        #[arg(long)]
        dir: PathBuf,
    },
}

/// Parses a JSONL file of `{"vector":[...],"payload":{...}}` lines into wire vectors.
fn parse_vectors_jsonl(path: &std::path::Path) -> Result<Vec<pb::Vector>, BoxError> {
    let text = std::fs::read_to_string(path)?;
    let mut vectors = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let values: Vec<f32> = v["vector"]
            .as_array()
            .ok_or("missing 'vector'")?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();
        let payload = v
            .get("payload")
            .filter(|p| !p.is_null())
            .map(|p| p.to_string());
        vectors.push(pb::Vector { values, payload });
    }
    Ok(vectors)
}

fn metric_code(m: &str) -> Result<i32, BoxError> {
    Ok(match m {
        "cosine" => pb::Metric::Cosine as i32,
        "dot" => pb::Metric::Dot as i32,
        "euclidean" => pb::Metric::Euclidean as i32,
        other => return Err(format!("unknown metric {other:?}").into()),
    })
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    match Cli::parse().command {
        Command::Create {
            addr,
            name,
            dim,
            metric,
        } => {
            let metric = metric_code(&metric)?;
            let mut client = pb::collections_client::CollectionsClient::connect(addr).await?;
            client
                .create(pb::CreateCollectionRequest { name, dim, metric })
                .await?;
            println!("created");
        }
        Command::Upsert {
            addr,
            collection,
            file,
        } => {
            let vectors = parse_vectors_jsonl(&file)?;
            let n = vectors.len();
            let mut client = pb::points_client::PointsClient::connect(addr).await?;
            let stream = tokio_stream::iter(std::iter::once(pb::UpsertRequest {
                collection,
                vectors,
            }));
            let resp = client.upsert(stream).await?.into_inner();
            println!(
                "upserted {} (ids {}..)",
                n,
                resp.ids.first().copied().unwrap_or(0)
            );
        }
        Command::Batch {
            addr,
            collection,
            file,
            delete,
            commit,
            message,
            tag,
        } => {
            let upserts = match &file {
                Some(path) => parse_vectors_jsonl(path)?,
                None => Vec::new(),
            };
            let mut client = pb::points_client::PointsClient::connect(addr).await?;
            let resp = client
                .write_batch(pb::WriteBatchRequest {
                    collection,
                    upserts,
                    deletes: delete,
                    commit,
                    message,
                    tag,
                })
                .await?
                .into_inner();
            let committed = resp
                .version
                .map(|v| format!(", committed v{v}"))
                .unwrap_or_default();
            println!(
                "upserted {} (ids {}..), deleted {}{}",
                resp.ids.len(),
                resp.ids.first().copied().unwrap_or(0),
                resp.deleted,
                committed,
            );
        }
        Command::Query {
            addr,
            collection,
            vector,
            k,
        } => {
            let vector: Vec<f32> = vector
                .split(',')
                .map(|s| s.trim().parse())
                .collect::<Result<_, _>>()?;
            let mut client = pb::query_client::QueryClient::connect(addr).await?;
            let resp = client
                .query(pb::QueryRequest {
                    collection,
                    vector,
                    k,
                    at: None,
                    filter: None,
                })
                .await?
                .into_inner();
            for r in resp.results {
                println!("{}\t{}", r.id, r.score);
            }
        }
        Command::Commit {
            addr,
            collection,
            message,
            tag,
        } => {
            let mut client = pb::versioning_client::VersioningClient::connect(addr).await?;
            let resp = client
                .commit(pb::CommitRequest {
                    collection,
                    message,
                    tag,
                })
                .await?
                .into_inner();
            println!("committed version {}", resp.version);
        }
        Command::Versions { addr, collection } => {
            let mut client = pb::versioning_client::VersioningClient::connect(addr).await?;
            let resp = client
                .list_versions(pb::ListVersionsRequest { collection })
                .await?
                .into_inner();
            for v in resp.versions {
                println!(
                    "v{}\t{}\t{}",
                    v.version,
                    v.trigger,
                    v.message.unwrap_or_default()
                );
            }
        }
        Command::Export { dir, out } => {
            let config = vecvec_core::durable::read_config(&dir)?;
            let dc =
                vecvec_core::DurableCollection::open(&dir, config, vecvec_core::FsyncMode::Sync)?;
            dc.export(&out)?;
            println!("exported {} -> {}", dir.display(), out.display());
        }
        Command::Import { input, dir } => {
            vecvec_core::import(&input, &dir)?;
            println!("imported {} -> {}", input.display(), dir.display());
        }
    }
    Ok(())
}
