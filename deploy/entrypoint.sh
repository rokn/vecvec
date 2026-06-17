#!/usr/bin/env bash
# Launch the vecvec server, and — unless disabled — the SCOPE UI alongside it.
# VECVEC_SCOPE=1 (the default) serves the UI on :8080 via Caddy; set it to 0 to
# run the database only.
set -euo pipefail

vecvec-server &
server_pid=$!

if [ "${VECVEC_SCOPE:-1}" = "1" ]; then
	caddy run --config /etc/caddy/Caddyfile --adapter caddyfile &
	scope_pid=$!
	echo "vecvec // SCOPE UI on http://0.0.0.0:8080"
fi

# Exit (and let the container restart) as soon as either process dies.
wait -n
exit $?
