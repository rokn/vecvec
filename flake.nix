{
  description = "vecvec — super-fast in-memory Rust vector DB with automatic git-like versioning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        inherit (pkgs) lib;

        # Toolchain components shared by the stable and nightly shells.
        rustExtensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];

        # Default dev toolchain: latest stable Rust.
        rustStable = pkgs.rust-bin.stable.latest.default.override {
          extensions = rustExtensions;
        };

        # Nightly toolchain that ships `miri` — used for the unsafe-module UB
        # tests (mmap cast-and-go, flat link arenas, self-referential Segment).
        # Enter with `nix develop .#nightly` and run `cargo miri test`.
        rustNightly = pkgs.rust-bin.selectLatestNightlyWith (toolchain:
          toolchain.default.override {
            extensions = rustExtensions ++ [ "miri" ];
          });

        # Build tools that must be on PATH at build time.
        nativeDeps = with pkgs; [
          pkg-config
          protobuf   # protoc — required by tonic 0.14 / prost 0.14 codegen
          cmake      # some -sys crates build with cmake
        ];

        # Libraries linked against (found via pkg-config / linker).
        libDeps = with pkgs;
          [ openssl ]
          ++ lib.optionals stdenv.isDarwin [ libiconv ];

        # Developer CLIs used by the test/bench/CI workflow from BuildPlan.md.
        toolDeps = with pkgs; [
          cargo-nextest   # test runner
          cargo-deny      # license/advisory/ban checks
          cargo-watch     # incremental dev loop
          sccache         # compile cache (CI + local)
          grpcurl         # exercise gRPC reflection/health
          curl            # fetch SIFT/GloVe bench datasets
          jq
        ];

        mkShell = rustToolchain:
          pkgs.mkShell {
            nativeBuildInputs = nativeDeps;
            buildInputs = libDeps;
            packages = [ rustToolchain ] ++ toolDeps;

            # tonic-build / prost-build look these up.
            PROTOC = "${pkgs.protobuf}/bin/protoc";
            PROTOC_INCLUDE = "${pkgs.protobuf}/include";
            # rust-analyzer std sources.
            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
            RUST_BACKTRACE = "1";

            shellHook = ''
              echo "vecvec dev shell"
              echo "  $(rustc --version)"
              echo "  $(cargo --version)"
              echo "  protoc $(protoc --version | awk '{print $2}')"
              echo "  tip: 'export RUSTC_WRAPPER=sccache' to enable the compile cache"
            '';
          };
      in
      {
        devShells.default = mkShell rustStable;
        devShells.nightly = mkShell rustNightly;

        formatter = pkgs.nixpkgs-fmt;
      });
}
