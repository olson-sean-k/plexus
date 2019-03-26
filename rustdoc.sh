#! /usr/bin/env bash

set -e

RUSTDOCFLAGS="--html-in-header doc/katex-header.html" \
cargo +nightly doc --no-deps "$@"
