#! /usr/bin/env bash

set -e

cargo clippy --all-features --all-targets "$@"
