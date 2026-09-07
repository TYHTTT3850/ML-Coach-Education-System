#!/bin/sh
set -eu

# Bind mounts are created by the host and can mask image-time ownership.
mkdir -p /app/data/chroma /app/data/uploads
chown -R coursecoach:coursecoach /app/data

exec gosu coursecoach "$@"
