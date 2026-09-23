#!/usr/bin/env bash
# Stop only the driver replicas this evaluation started.
#
# Filtering is by container name, never by image: other people on this box run
# containers from the same AlpaSim images and an ancestor filter would take
# theirs down too.
set -euo pipefail
CONTAINER_PREFIX="${CONTAINER_PREFIX:-axe-curatedval-drv}"
mapfile -t names < <(docker ps -a --filter "name=^${CONTAINER_PREFIX}-" --format '{{.Names}}')
if [[ "${#names[@]}" -eq 0 ]]; then
    echo "no containers matching ${CONTAINER_PREFIX}-*"
    exit 0
fi
printf 'stopping %d container(s):\n' "${#names[@]}"
printf '  %s\n' "${names[@]}"
docker rm -f "${names[@]}" >/dev/null
