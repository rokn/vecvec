# ---- server builder ----
# Pin to the same toolchain as rust-toolchain.toml (1.96.0).
FROM rust:1.96-bookworm AS builder

# vecvec-proto's build.rs runs protoc (tonic/prost codegen).
RUN apt-get update \
    && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy the whole workspace and build the server binary in release mode.
COPY . .
RUN cargo build --release --locked -p vecvec-server

# ---- UI builder ----
# Builds the vecvec // SCOPE static bundle (Vite/React) under vvui/.
FROM node:20-bookworm-slim AS ui-builder
WORKDIR /ui
COPY vvui/package.json vvui/package-lock.json ./
RUN npm ci
COPY vvui/ ./
RUN npm run build

# ---- runtime ----
FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home --home-dir /data vecvec

# Server binary, the Caddy static-server/proxy used to serve the SCOPE UI, the
# built UI assets, and the launcher.
COPY --from=builder /build/target/release/vecvec-server /usr/local/bin/vecvec-server
COPY --from=caddy:2 /usr/bin/caddy /usr/local/bin/caddy
COPY --from=ui-builder /ui/dist /srv/ui
COPY deploy/Caddyfile /etc/caddy/Caddyfile
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Bind to all interfaces inside the container (the binary defaults to 127.0.0.1)
# and persist data under /data. VECVEC_SCOPE=1 serves the UI on :8080.
ENV VECVEC_GRPC_ADDR=0.0.0.0:6334 \
    VECVEC_REST_ADDR=0.0.0.0:6333 \
    VECVEC_DATA_DIR=/data/vecvec-data \
    VECVEC_SCOPE=1

USER vecvec
WORKDIR /data
VOLUME ["/data"]

# REST (6333), gRPC (6334), SCOPE UI (8080).
EXPOSE 6333 6334 8080

ENTRYPOINT ["entrypoint.sh"]
