#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Error on line $LINENO. Exiting."; exit 1' ERR

# --- Load config from .env if present ---
if [ -f .env ]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        key="$(echo "$key" | xargs)"
        export "$key=$value"
    done < .env
fi

# --- Required config (from .env or environment) ---
: "${PROJECT_ID:?Error: PROJECT_ID is not set. Add it to .env or export it.}"
: "${REGION:=us-central1}"
: "${BUCKET_URI:?Error: BUCKET_URI is not set. Add it to .env or export it.}"

# --- Derived config (override via .env if needed) ---
: "${REPO_NAME:=segment-any-change-repo}"
: "${IMAGE_NAME:=segment-any-change-model-server}"
: "${MODEL_DISPLAY_NAME:=segment-any-change-model}"
: "${ENDPOINT_DISPLAY_NAME:=segment-any-change-endpoint}"
: "${DEPLOYED_MODEL_NAME:=segment-any-change-model-deployed}"
: "${VERTEX_SERVICE_ACCOUNT_NAME:=seg-anychange-endpoint-sa}"
: "${MACHINE_TYPE:=n1-standard-8}"
: "${ACCELERATOR:=type=nvidia-tesla-t4,count=1}"
: "${MIN_REPLICAS:=1}"
: "${MAX_REPLICAS:=1}"
VERTEX_SERVICE_ACCOUNT_EMAIL="${VERTEX_SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Image tag (timestamped for unique versioning)
IMAGE_TAG="v$(date +%Y%m%d-%H%M%S)"
IMAGE_URI_DEFAULT="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
IMAGE_NAME_FULL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

# --- helpers ---
to_full_model_name() {
  local raw="$1"
  if [[ "$raw" == projects/*/locations/*/models/* ]]; then
    echo "$raw"
  else
    echo "projects/${PROJECT_ID}/locations/${REGION}/models/${raw}"
  fi
}

wait_for_alias() {
  local parent_model_name="$1"
  local alias="$2"
  local tries="${3:-24}"  # ~2 minutes max (24 * 5s)
  local version_id=""

  echo "   Waiting for alias '${alias}'..." >&2
  for i in $(seq 1 "$tries"); do
    version_id=$(gcloud ai models list-version \
      "${parent_model_name}" \
      --region="${REGION}" \
      --filter="versionAliases:${alias}" \
      --format="value(versionId)" || true)

    if [[ -n "${version_id}" ]]; then
      echo "   Alias '${alias}' found on version '${version_id}'." >&2
      echo -n "${version_id}"
      return 0
    fi
    sleep 5
  done
  echo "" >&2
  return 1
}

resolve_image_uri() {
  if [[ -n "${EXISTING_IMAGE_URI:-}" ]]; then
    IMAGE_URI="${EXISTING_IMAGE_URI}"
    if [[ "${IMAGE_URI}" == *@sha256:* ]]; then
      IMAGE_ALIAS="sha$(echo "${IMAGE_URI}" | sed -E 's~.*@sha256:([0-9a-f]{8}).*~\1~')"
    else
      IMAGE_ALIAS="$(basename "${IMAGE_URI}")"
      IMAGE_ALIAS="${IMAGE_ALIAS##*:}"
    fi
  else
    IMAGE_URI="${IMAGE_URI_DEFAULT}"
    IMAGE_ALIAS="${IMAGE_TAG}"
  fi
}

_deploy_model_to_endpoint() {
  local endpoint_id="$1"
  local model_version="$2"

  echo "[7/7] Deploying model to endpoint (this may take 10-15 minutes)..."
  echo "   Machine: ${MACHINE_TYPE}, Accelerator: ${ACCELERATOR}"
  echo "   Replicas: ${MIN_REPLICAS}-${MAX_REPLICAS}"
  gcloud ai endpoints deploy-model "${endpoint_id}" \
    --region="${REGION}" \
    --model="${model_version}" \
    --display-name="${DEPLOYED_MODEL_NAME}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="${ACCELERATOR}" \
    --service-account="${VERTEX_SERVICE_ACCOUNT_EMAIL}" \
    --min-replica-count="${MIN_REPLICAS}" \
    --max-replica-count="${MAX_REPLICAS}" \
    --traffic-split="0=100" >/dev/null

  echo "   Deployed."
}

# ===========================================================================
# Commands
# ===========================================================================

usage() {
    echo "Usage: ./deploy.sh [command]"
    echo ""
    echo "Commands:"
    echo "  deploy     Build image, upload model, create endpoint, deploy (full pipeline)"
    echo "  redeploy   Deploy latest model version to existing endpoint (skip build/upload)"
    echo "  stop       Undeploy model from endpoint and delete endpoint"
    echo "  clean      Delete endpoint, model versions, and container images"
    echo "  status     Show endpoint and model status"
    echo "  logs       Show recent prediction logs"
    echo ""
    echo "Config is loaded from .env (see .env.example)."
    echo "To skip image build: export EXISTING_IMAGE_URI=<uri> before running deploy."
    echo ""
}

do_deploy() {
    resolve_image_uri

    echo "################################################################"
    echo "Starting Vertex AI Deployment"
    echo "  Project:       ${PROJECT_ID}"
    echo "  Region:        ${REGION}"
    echo "  Image:         ${IMAGE_URI}"
    echo "  Version alias: ${IMAGE_ALIAS}"
    echo "  Bucket:        ${BUCKET_URI}"
    echo "  Endpoint:      ${ENDPOINT_DISPLAY_NAME}"
    echo "################################################################"

    # Step 1: Enable APIs
    echo "[1/7] Enabling required APIs..."
    gcloud config set project "${PROJECT_ID}" >/dev/null
    gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com \
                           storage.googleapis.com cloudbuild.googleapis.com >/dev/null
    echo "   APIs ready."

    # Step 2: Ensure Artifact Registry
    echo "[2/7] Ensuring Artifact Registry repo '${REPO_NAME}'..."
    if ! gcloud artifacts repositories describe "${REPO_NAME}" \
         --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
      gcloud artifacts repositories create "${REPO_NAME}" \
        --repository-format=docker --location="${REGION}" \
        --description="Repo for segment any change model" >/dev/null
      echo "   Created."
    else
      echo "   Already exists."
    fi

    # Step 3: Build & push image
    echo "[3/7] Preparing Docker image..."
    if [[ -n "${EXISTING_IMAGE_URI:-}" ]]; then
      echo "   Using existing image: ${IMAGE_URI}"
    else
      echo "   Building & pushing (${IMAGE_URI})..."
      gcloud builds submit --tag "${IMAGE_URI}" . --project="${PROJECT_ID}" --timeout=3600s >/dev/null
      echo "   Image pushed."
    fi

    # Step 4: Upload model version
    echo "[4/7] Resolving parent model..."
    PARENT_MODEL_RAW=$(gcloud ai models list \
      --region="${REGION}" \
      --filter="displayName=${MODEL_DISPLAY_NAME}" \
      --format="value(name)")

    if [[ -z "${PARENT_MODEL_RAW}" ]]; then
      echo "   Parent model not found; creating '${MODEL_DISPLAY_NAME}'..."
      gcloud ai models upload \
        --region="${REGION}" \
        --display-name="${MODEL_DISPLAY_NAME}" \
        --container-image-uri="${IMAGE_URI}" \
        --artifact-uri="${BUCKET_URI}" \
        --container-health-route="/health" \
        --container-predict-route="/predict" \
        --container-ports="8080" >/dev/null
      PARENT_MODEL_RAW=$(gcloud ai models list \
        --region="${REGION}" \
        --filter="displayName=${MODEL_DISPLAY_NAME}" \
        --format="value(name)")
      echo "   Created: ${PARENT_MODEL_RAW}"
    else
      echo "   Found: ${PARENT_MODEL_RAW}"
    fi

    PARENT_MODEL_NAME="$(to_full_model_name "${PARENT_MODEL_RAW}")"

    echo "   Uploading new version..."
    gcloud ai models upload \
      --region="${REGION}" \
      --display-name="${MODEL_DISPLAY_NAME}" \
      --parent-model="${PARENT_MODEL_NAME}" \
      --version-aliases="candidate,${IMAGE_ALIAS}" \
      --version-description="Segment Any Change built from ${IMAGE_URI}" \
      --container-image-uri="${IMAGE_URI}" \
      --artifact-uri="${BUCKET_URI}" \
      --container-health-route="/health" \
      --container-predict-route="/predict" \
      --container-ports="8080" >/dev/null

    VERSION_ID="$(wait_for_alias "${PARENT_MODEL_NAME}" "${IMAGE_ALIAS}")" || {
      echo "Could not resolve model version by alias ${IMAGE_ALIAS} after waiting."
      gcloud ai models list-version --model="${PARENT_MODEL_NAME}" --region="${REGION}" \
        --format="table(versionId,versionAliases.list(),createTime)"
      exit 1
    }

    NEW_MODEL_VERSION_NAME="${PARENT_MODEL_NAME}@${VERSION_ID}"
    echo "   Model version: ${NEW_MODEL_VERSION_NAME}"

    # Step 5: Create endpoint (reuse if exists)
    echo "[5/7] Ensuring endpoint '${ENDPOINT_DISPLAY_NAME}'..."
    ENDPOINT_ID=$(gcloud ai endpoints list \
      --region="${REGION}" \
      --filter="displayName=${ENDPOINT_DISPLAY_NAME}" \
      --format="value(name)" 2>/dev/null) || true

    if [[ -z "${ENDPOINT_ID}" ]]; then
      ENDPOINT_ID=$(gcloud ai endpoints create \
        --region="${REGION}" \
        --display-name="${ENDPOINT_DISPLAY_NAME}" \
        --format="value(name)")
      echo "   Created: ${ENDPOINT_ID}"
    else
      echo "   Already exists: ${ENDPOINT_ID}"
    fi

    # Step 6: Service account + GCS access
    echo "[6/7] Ensuring service account & storage access..."
    if ! gcloud iam service-accounts describe "${VERTEX_SERVICE_ACCOUNT_EMAIL}" \
         --project="${PROJECT_ID}" >/dev/null 2>&1; then
      gcloud iam service-accounts create "${VERTEX_SERVICE_ACCOUNT_NAME}" \
        --display-name="Service Account for Segment Any Change Endpoint" >/dev/null
      echo "   Created SA: ${VERTEX_SERVICE_ACCOUNT_EMAIL}"
    else
      echo "   SA exists: ${VERTEX_SERVICE_ACCOUNT_EMAIL}"
    fi
    gsutil iam ch "serviceAccount:${VERTEX_SERVICE_ACCOUNT_EMAIL}:roles/storage.objectViewer" "${BUCKET_URI}" >/dev/null || true
    echo "   Granted storage.objectViewer on ${BUCKET_URI}"

    # Step 7: Deploy to endpoint
    _deploy_model_to_endpoint "${ENDPOINT_ID}" "${NEW_MODEL_VERSION_NAME}"

    echo "################################################################"
    echo "Deployment complete."
    echo "  Endpoint ID:    ${ENDPOINT_ID}"
    echo "  Model Version:  ${NEW_MODEL_VERSION_NAME}"
    echo "  Image:          ${IMAGE_URI}"
    echo "  Version alias:  ${IMAGE_ALIAS}"
    echo "################################################################"
}

do_redeploy() {
    echo "Redeploying to existing endpoint..."

    # Find endpoint
    ENDPOINT_ID=$(gcloud ai endpoints list \
      --region="${REGION}" \
      --filter="displayName=${ENDPOINT_DISPLAY_NAME}" \
      --format="value(name)" 2>/dev/null) || true

    if [[ -z "${ENDPOINT_ID}" ]]; then
      echo "Endpoint '${ENDPOINT_DISPLAY_NAME}' not found. Run './deploy.sh deploy' first."
      exit 1
    fi
    echo "   Endpoint: ${ENDPOINT_ID}"

    # Undeploy existing model if any
    DEPLOYED_MODEL_ID=$(gcloud ai endpoints describe "${ENDPOINT_ID}" \
      --region="${REGION}" \
      --format="value(deployedModels[0].id)" 2>/dev/null) || true

    if [[ -n "${DEPLOYED_MODEL_ID}" ]]; then
      echo "   Undeploying current model (${DEPLOYED_MODEL_ID})..."
      gcloud ai endpoints undeploy-model "${ENDPOINT_ID}" \
        --region="${REGION}" \
        --deployed-model-id="${DEPLOYED_MODEL_ID}" \
        --quiet >/dev/null
      echo "   Undeployed."
    fi

    # Find latest model version
    MODEL_ID=$(gcloud ai models list \
      --region="${REGION}" \
      --filter="displayName=${MODEL_DISPLAY_NAME}" \
      --format="value(name)" 2>/dev/null) || true

    if [[ -z "${MODEL_ID}" ]]; then
      echo "Model '${MODEL_DISPLAY_NAME}' not found. Run './deploy.sh deploy' first."
      exit 1
    fi

    FULL_MODEL_NAME="$(to_full_model_name "${MODEL_ID}")"
    LATEST_VERSION=$(gcloud ai models list-version "${FULL_MODEL_NAME}" \
      --region="${REGION}" \
      --sort-by="~createTime" \
      --limit=1 \
      --format="value(versionId)")

    if [[ -z "${LATEST_VERSION}" ]]; then
      echo "No model versions found. Run './deploy.sh deploy' first."
      exit 1
    fi

    MODEL_VERSION="${FULL_MODEL_NAME}@${LATEST_VERSION}"
    echo "   Model version: ${MODEL_VERSION}"

    _deploy_model_to_endpoint "${ENDPOINT_ID}" "${MODEL_VERSION}"

    echo "################################################################"
    echo "Redeploy complete."
    echo "  Endpoint: ${ENDPOINT_ID}"
    echo "  Model:    ${MODEL_VERSION}"
    echo "################################################################"
}

do_status() {
    echo "Checking Vertex AI resources..."
    echo ""

    # Endpoint
    ENDPOINT_ID=$(gcloud ai endpoints list \
      --region="${REGION}" \
      --filter="displayName=${ENDPOINT_DISPLAY_NAME}" \
      --format="value(name)" 2>/dev/null) || true

    if [[ -z "${ENDPOINT_ID}" ]]; then
      echo "Endpoint '${ENDPOINT_DISPLAY_NAME}' not found."
    else
      echo "Endpoint: ${ENDPOINT_DISPLAY_NAME}"
      echo "  ID: ${ENDPOINT_ID}"
      gcloud ai endpoints describe "${ENDPOINT_ID}" \
        --region="${REGION}" \
        --format="table(deployedModels[].displayName,deployedModels[].id)" 2>/dev/null || true
    fi

    echo ""

    # Model
    MODEL_ID=$(gcloud ai models list \
      --region="${REGION}" \
      --filter="displayName=${MODEL_DISPLAY_NAME}" \
      --format="value(name)" 2>/dev/null) || true

    if [[ -z "${MODEL_ID}" ]]; then
      echo "Model '${MODEL_DISPLAY_NAME}' not found."
    else
      echo "Model: ${MODEL_DISPLAY_NAME}"
      echo "  ID: ${MODEL_ID}"
      echo "  Versions:"
      gcloud ai models list-version "$(to_full_model_name "${MODEL_ID}")" \
        --region="${REGION}" \
        --format="table(versionId,versionAliases.list(),createTime)" 2>/dev/null || true
    fi
}

do_stop() {
    echo "Undeploying model from endpoint..."

    ENDPOINT_ID=$(gcloud ai endpoints list \
      --region="${REGION}" \
      --filter="displayName=${ENDPOINT_DISPLAY_NAME}" \
      --format="value(name)" 2>/dev/null) || true

    if [[ -z "${ENDPOINT_ID}" ]]; then
      echo "Endpoint '${ENDPOINT_DISPLAY_NAME}' not found. Nothing to stop."
      return
    fi

    # Get deployed model ID
    DEPLOYED_MODEL_ID=$(gcloud ai endpoints describe "${ENDPOINT_ID}" \
      --region="${REGION}" \
      --format="value(deployedModels[0].id)" 2>/dev/null) || true

    if [[ -n "${DEPLOYED_MODEL_ID}" ]]; then
      echo "  Undeploying model ${DEPLOYED_MODEL_ID} from endpoint..."
      gcloud ai endpoints undeploy-model "${ENDPOINT_ID}" \
        --region="${REGION}" \
        --deployed-model-id="${DEPLOYED_MODEL_ID}" \
        --quiet >/dev/null
      echo "  Model undeployed."
    else
      echo "  No model deployed on endpoint."
    fi

    echo "  Deleting endpoint..."
    gcloud ai endpoints delete "${ENDPOINT_ID}" \
      --region="${REGION}" \
      --quiet >/dev/null
    echo "  Endpoint deleted."

    echo "Stop complete. Model and images are retained."
    echo "Use './deploy.sh clean' to remove everything."
}

do_clean() {
    echo "Cleaning up all Vertex AI resources..."
    echo ""

    # Stop first (undeploy + delete endpoint)
    do_stop 2>/dev/null || true

    # Delete model versions
    MODEL_ID=$(gcloud ai models list \
      --region="${REGION}" \
      --filter="displayName=${MODEL_DISPLAY_NAME}" \
      --format="value(name)" 2>/dev/null) || true

    if [[ -n "${MODEL_ID}" ]]; then
      FULL_MODEL_NAME="$(to_full_model_name "${MODEL_ID}")"
      echo "Deleting model '${MODEL_DISPLAY_NAME}'..."
      gcloud ai models delete "${FULL_MODEL_NAME}" \
        --region="${REGION}" \
        --quiet >/dev/null 2>&1 && echo "  Model deleted." || echo "  Could not delete model (may have active deployments)."
    else
      echo "Model '${MODEL_DISPLAY_NAME}' not found, skipping."
    fi

    # Delete container images
    echo "Deleting container images from Artifact Registry..."
    gcloud artifacts docker images delete "${IMAGE_NAME_FULL}" \
      --delete-tags --quiet 2>/dev/null && echo "  Images deleted." || echo "  No images found, skipping."

    echo ""
    echo "Clean up complete."
}

do_logs() {
    ENDPOINT_ID=$(gcloud ai endpoints list \
      --region="${REGION}" \
      --filter="displayName=${ENDPOINT_DISPLAY_NAME}" \
      --format="value(name)" 2>/dev/null) || true

    if [[ -z "${ENDPOINT_ID}" ]]; then
      echo "Endpoint '${ENDPOINT_DISPLAY_NAME}' not found."
      return
    fi

    echo "Showing recent logs for endpoint ${ENDPOINT_ID}..."
    gcloud logging read "resource.type=aiplatform.googleapis.com/Endpoint AND resource.labels.endpoint_id=$(basename "${ENDPOINT_ID}")" \
      --project="${PROJECT_ID}" \
      --limit=50 \
      --format="table(timestamp,textPayload)" \
      --freshness=1h
}

# ===========================================================================
# Route command
# ===========================================================================
case "${1:-}" in
    deploy)   do_deploy ;;
    redeploy) do_redeploy ;;
    stop)     do_stop ;;
    clean)    do_clean ;;
    status)   do_status ;;
    logs)     do_logs ;;
    *)        usage ;;
esac
