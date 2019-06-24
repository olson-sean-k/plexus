#! /usr/bin/env bash

set -e

cargo +nightly fmt "$@"
