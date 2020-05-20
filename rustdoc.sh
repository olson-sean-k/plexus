#! /usr/bin/env bash

# TODO: This requires the `--no-deps` flag. Without it, when `rustdoc` attempts
#       to embed the KaTeX header it will fail to resolve the path.
# TODO: This script sets the `RUSTDOCFLAGS` environment variable to configure
#       the KaTeX header for documentation. This cannot be accomplished with
#       Cargo configuration yet.
#
#       See https://github.com/rust-lang/cargo/issues/8097

set -e

RUSTDOCFLAGS=--html-in-header=./doc/katex-header.html \
cargo +nightly doc --no-deps $@
