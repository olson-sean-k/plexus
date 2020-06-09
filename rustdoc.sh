#! /usr/bin/env bash

# TODO: This script sets the `RUSTDOCFLAGS` environment variable to configure
#       the KaTeX header for documentation. This cannot be accomplished with
#       Cargo configuration yet.
#
#       See https://github.com/rust-lang/cargo/issues/8097

set -e

RUSTDOCFLAGS=--html-in-header=$(realpath ./doc/katex-header.html) \
cargo +nightly doc $@
