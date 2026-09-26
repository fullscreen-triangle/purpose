# Serves the chigutiro API. Consolidation (training) is deliberately not in
# this image: run it on a machine with the compute for it (`chigutiro
# consolidate`, or `serve` with CHIGUTIRO_PURPOSE_BIN set).
FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates crates
RUN cargo build --release --bin chigutiro

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /data chigutiro
COPY --from=build /src/target/release/chigutiro /usr/local/bin/chigutiro
USER chigutiro
# Mount a persistent volume here: it holds the record log and model state.
VOLUME /data
ENV CHIGUTIRO_DATA=/data \
    CHIGUTIRO_HOST=0.0.0.0 \
    CHIGUTIRO_PORT=8740
EXPOSE 8740
# CHIGUTIRO_KEY and CHIGUTIRO_TOKEN must be provided as secrets at run time.
CMD ["chigutiro", "serve"]
