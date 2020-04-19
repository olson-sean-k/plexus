#! /usr/bin/env bash

# TODO: This script sets the `RUSTDOCFLAGS` environment variable to configure
#       the KaTeX header for documentation. This is also done via Cargo
#       configuration for the `plexus` package, but Cargo executes `rustdoc`
#       from different directories depending on the command (`cargo doc` vs.
#       `cargo test`), causing relative paths to resolve differently and fail
#       for either documentation or tests.
#
#       Cargo is configured using a path that works for `cargo test`, and this
#       script provides a workaround to enable `cargo doc`.
#
#       See https://github.com/rust-lang/cargo/issues/8097

set -e

RUSTDOCFLAGS=--html-in-header=./doc/katex-header.html \
cargo +nightly doc $@
