#!/usr/bin/env bash
# Deploy Virtual Paddock Engineer to Azure Container Apps.
#
#   Browser ──► paddock-web (nginx, PUBLIC) ──/api──► paddock-api (FastAPI, PRIVATE) ──► Supabase / Gemini
#
# Prereqs (once):  brew install azure-cli && az login ; Docker Desktop running
# Usage (from the repo root):   bash scripts/deploy-azure.sh
#
# Safe to re-run: the first run creates everything, later runs rebuild the images and
# update the two apps in place (same URL).
#
# Lessons baked in from the first (manual) deploy on an Azure for Students subscription:
#   - Only these regions are allowed: japaneast, japanwest, eastasia, malaysiawest, indonesiacentral.
#   - eastasia (Hong Kong) can't call the Gemini API ("User location is not supported") -> use japaneast.
#   - Cloud builds (az acr build / "ACR Tasks") are blocked -> build locally for amd64 and push.
#   - Only ONE Container Apps environment is allowed per subscription.
#   - "Express" environments break private app-to-app calls -> create a ConsumptionOnly environment.
#   - Registry login must be passed explicitly (--registry-username/--registry-password).

set -euo pipefail
cd "$(dirname "$0")/.."            # always run from the repo root

# ---------------------------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------------------------
LOCATION="${LOCATION:-japaneast}"   # where the apps run (must be allowed AND supported by Gemini)
ACR_LOCATION="${ACR_LOCATION:-eastasia}"  # where the registry lives (any allowed region works)
RG="paddock-rg"                     # resource group = one folder holding every resource below
ACR="${ACR:-paddockreg15457}"       # registry name (globally unique; this is the one already created)
ENV_NAME="paddock-jp"               # Container Apps environment = shared private network for both apps
API="paddock-api"                   # backend app name (also its private address: http://paddock-api)
WEB="paddock-web"                   # frontend app name (gets the public URL)
TAG="$(date +%Y%m%d%H%M%S)"         # new image tag each deploy, so Azure always picks up the new build
REGISTRY="$ACR.azurecr.io"

# Secrets come from .env (same file docker-compose uses). Loaded into this shell only, never printed.
if [[ ! -f .env ]]; then echo "Missing .env in repo root" >&2; exit 1; fi
set -a; source .env; set +a
: "${SUPABASE_URL:?missing in .env}" "${SUPABASE_SERVICE_KEY:?missing in .env}" "${GEMINI_API_KEY:?missing in .env}"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. One-time setup: CLI extension + resource providers ("turn on" these Azure services)
# ---------------------------------------------------------------------------
step "Preparing Azure CLI and providers"
az extension add --name containerapp --upgrade --only-show-errors
for p in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry; do
  az provider register -n "$p" --wait --only-show-errors
done

# ---------------------------------------------------------------------------
# 2. Resource group (delete it later to remove EVERYTHING:  az group delete -n paddock-rg)
# ---------------------------------------------------------------------------
step "Resource group: $RG"
if ! az group show -n "$RG" -o none 2>/dev/null; then
  az group create -n "$RG" -l "$ACR_LOCATION" -o none
fi

# ---------------------------------------------------------------------------
# 3. Container registry (stores the images). Basic tier ≈ $5/month, paid from student credit.
# ---------------------------------------------------------------------------
step "Container registry: $ACR"
if ! az acr show -n "$ACR" -o none 2>/dev/null; then
  az acr create -g "$RG" -n "$ACR" -l "$ACR_LOCATION" --sku Basic --admin-enabled true -o none
fi
ACR_PASSWORD="$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)"

# ---------------------------------------------------------------------------
# 4. Build images ON THIS MAC for amd64 (Azure's chip type; the Mac is arm64) and push them.
#    (Cloud builds with `az acr build` are blocked on student subscriptions.)
# ---------------------------------------------------------------------------
step "Logging Docker in to $REGISTRY"
az acr login -n "$ACR" --only-show-errors

step "Building + pushing backend image"
docker buildx build --platform linux/amd64 -t "$REGISTRY/paddock-backend:$TAG" -f backend/Dockerfile --push .

step "Building + pushing frontend image"
docker buildx build --platform linux/amd64 -t "$REGISTRY/paddock-frontend:$TAG" --push ./frontend

# ---------------------------------------------------------------------------
# 5. Container Apps environment — ConsumptionOnly (NOT Express), so apps can call each other by name.
# ---------------------------------------------------------------------------
step "Container Apps environment: $ENV_NAME"
if ! az containerapp env show -g "$RG" -n "$ENV_NAME" -o none 2>/dev/null; then
  az containerapp env create -g "$RG" -n "$ENV_NAME" -l "$LOCATION" \
    --environment-mode ConsumptionOnly -o none
fi

# ---------------------------------------------------------------------------
# 6. Backend app — PRIVATE (internal ingress): only apps inside the environment can call it.
#    Secrets are stored encrypted in Azure and exposed to the container as env vars.
#    min-replicas 0 = scale to zero when idle (stays inside the free monthly grant).
# ---------------------------------------------------------------------------
step "Backend app: $API"
SECRETS=(
  "supabase-url=$SUPABASE_URL"
  "supabase-key=$SUPABASE_SERVICE_KEY"
  "gemini-key=$GEMINI_API_KEY"
)
API_ENV=(
  "SUPABASE_URL=secretref:supabase-url"
  "SUPABASE_SERVICE_KEY=secretref:supabase-key"
  "GEMINI_API_KEY=secretref:gemini-key"
  "LLM_PROVIDER=gemini"
  "EMBEDDING_PROVIDER=gemini"
)
if ! az containerapp show -g "$RG" -n "$API" -o none 2>/dev/null; then
  az containerapp create -g "$RG" -n "$API" --environment "$ENV_NAME" \
    --image "$REGISTRY/paddock-backend:$TAG" \
    --registry-server "$REGISTRY" --registry-username "$ACR" --registry-password "$ACR_PASSWORD" \
    --ingress internal --target-port 8000 \
    --cpu 0.5 --memory 1.0Gi --min-replicas 0 --max-replicas 1 \
    --secrets "${SECRETS[@]}" --env-vars "${API_ENV[@]}" -o none
  # Let nginx call it over plain http INSIDE the environment (still private to the internet).
  az containerapp ingress update -g "$RG" -n "$API" --allow-insecure -o none
else
  az containerapp secret set -g "$RG" -n "$API" --secrets "${SECRETS[@]}" -o none
  az containerapp update -g "$RG" -n "$API" \
    --image "$REGISTRY/paddock-backend:$TAG" --set-env-vars "${API_ENV[@]}" -o none
fi

# ---------------------------------------------------------------------------
# 7. Frontend app — PUBLIC (external ingress, HTTPS handled by Azure).
#    BACKEND_URL tells nginx where to forward /api (see frontend/nginx.conf).
# ---------------------------------------------------------------------------
step "Frontend app: $WEB"
if ! az containerapp show -g "$RG" -n "$WEB" -o none 2>/dev/null; then
  az containerapp create -g "$RG" -n "$WEB" --environment "$ENV_NAME" \
    --image "$REGISTRY/paddock-frontend:$TAG" \
    --registry-server "$REGISTRY" --registry-username "$ACR" --registry-password "$ACR_PASSWORD" \
    --ingress external --target-port 80 \
    --cpu 0.25 --memory 0.5Gi --min-replicas 0 --max-replicas 1 \
    --env-vars "BACKEND_URL=http://$API" -o none
else
  az containerapp update -g "$RG" -n "$WEB" \
    --image "$REGISTRY/paddock-frontend:$TAG" --set-env-vars "BACKEND_URL=http://$API" -o none
fi

# ---------------------------------------------------------------------------
# 8. Done
# ---------------------------------------------------------------------------
URL="https://$(az containerapp show -g "$RG" -n "$WEB" --query properties.configuration.ingress.fqdn -o tsv)"
step "Deployed!"
echo "App:     $URL"
echo "Health:  curl $URL/api/health"
echo "Logs:    az containerapp logs show -g $RG -n $API --follow"
echo "(First request after idle can take ~10-30s while the containers wake up.)"
