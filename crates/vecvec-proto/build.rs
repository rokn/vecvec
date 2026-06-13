use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    // Emit a file descriptor set so the server can expose gRPC reflection.
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .file_descriptor_set_path(out_dir.join("vecvec_descriptor.bin"))
        .compile_protos(&["proto/vecvec.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/vecvec.proto");
    Ok(())
}
