#!/bin/bash
# Creates the agent's own database + login in the SAME Postgres instance as app_db.
# Runs automatically on a fresh volume (docker-entrypoint-initdb.d); safe to re-run by hand.
set -euo pipefail

: "${AGENT_DB_PASSWORD:?AGENT_DB_PASSWORD is not set}"
AGENT_DB_NAME="${AGENT_DB_NAME:-agent_app}"
AGENT_DB_USER="${AGENT_DB_USER:-agent_app}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
     -v db="$AGENT_DB_NAME" -v usr="$AGENT_DB_USER" -v pw="$AGENT_DB_PASSWORD" <<'EOSQL'
-- 1. Login role (create once, keep the password in sync)
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'usr', :'pw')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'usr') \gexec
SELECT format('ALTER ROLE %I PASSWORD %L', :'usr', :'pw') \gexec

-- 2. Database owned by that role (create once)
SELECT format('CREATE DATABASE %I OWNER %I', :'db', :'usr')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'db') \gexec

-- 3. Only the owner may connect (not web_anon, not PUBLIC)
SELECT format('REVOKE ALL ON DATABASE %I FROM PUBLIC', :'db') \gexec
EOSQL

# 4. Keep other roles out of its public schema too
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$AGENT_DB_NAME" \
     -v usr="$AGENT_DB_USER" <<'EOSQL'
REVOKE ALL ON SCHEMA public FROM PUBLIC;
SELECT format('GRANT ALL ON SCHEMA public TO %I', :'usr') \gexec
EOSQL
echo "agent database ready: $AGENT_DB_NAME (owner $AGENT_DB_USER)"