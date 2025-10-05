from dotenv import load_dotenv

load_dotenv()

from api.middleware.subscriptionMiddleware import SubscriptionMiddleware
from api.middleware.autumnAccessMiddleware import AutumnAccessMiddleware

from fastapi.responses import (
    RedirectResponse,
)  # Add RedirectResponse import
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
import os

from api.routes import (
    run,
    volumes,
    internal,
    workflow,
    log,
    workflows,
    machines,
    comfy_node,
    comfy_proxy,
    deployments,
    runs,
    session,
    files,
    models,
    platform,
    search,
    form,
    admin,
    image_optimization,
    share,
    auth_response,
    clerk_webhook,
    autumn_webhook,
)
from api.autumn_mount import autumn_app
from api.modal import builder


import logfire
import logging

# import all you need from fastapi-pagination
# from fastapi_pagination import Page, add_pagination, paginate
from api.middleware.authMiddleware import AuthMiddleware


# from logtail import LogtailHandler
import logging
from api.router import app, public_api_router, api_router

# Import codegen integration
from api.codegen_integration import setup_codegen_integration, feature_flags

logger = logfire
logfire.configure(
    service_name="comfydeploy-api",
    distributed_tracing=False,
    send_to_logfire=False,
)

logging.basicConfig(level=logging.INFO)

# logger = logging.getLogger(__name__)

# Replace the existing oauth2_scheme declaration with this:
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        # "read:runs": "Read access to runs",
        # "write:runs": "Write access to runs",
        # Add more scopes as needed
    },
)

app.webhooks.include_router(run.webhook_router)


# Add proxy redirect route - this needs to be added before the api_router include
@app.api_route(
    "/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy_redirect(path: str):
    """Redirect /proxy requests to /api/comfy-org/proxy/ for ComfyUI compatibility"""
    return RedirectResponse(url=f"/api/comfy-org/proxy/{path}", status_code=307)


@app.api_route(
    "/proxy", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
)
async def proxy_redirect_base():
    """Redirect base /proxy requests to /api/comfy-org/proxy/ for ComfyUI compatibility"""
    return RedirectResponse(url="/api/comfy-org/proxy/", status_code=307)


@app.api_route("/customers/storage", methods=["POST"])
async def proxy_redirect_comfy_org_storage():
    """Redirect /customers/storage requests to /api/comfy-org/customers/storage/ for ComfyUI compatibility"""
    return RedirectResponse(url="/api/comfy-org/customers/storage", status_code=307)


# Include routers
api_router.include_router(run.router)
api_router.include_router(internal.router)
api_router.include_router(workflows.router)
api_router.include_router(workflow.router)
api_router.include_router(builder.router)
api_router.include_router(machines.router)
api_router.include_router(machines.public_router)
api_router.include_router(log.router)
api_router.include_router(volumes.router)
api_router.include_router(comfy_node.router)
api_router.include_router(comfy_proxy.router)
api_router.include_router(share.router)
api_router.include_router(deployments.router)
api_router.include_router(session.router)
api_router.include_router(session.beta_router)
api_router.include_router(runs.router)
api_router.include_router(files.router)
api_router.include_router(models.router)
api_router.include_router(platform.router)
api_router.include_router(search.router)
api_router.include_router(form.router)
api_router.include_router(admin.router)  # Add the admin router to internal API
api_router.include_router(image_optimization.router)
api_router.include_router(auth_response.router)
api_router.include_router(clerk_webhook.router)
api_router.include_router(autumn_webhook.router)

# Mount Autumn ASGI app at /api/autumn
app.mount("/api/autumn", autumn_app)

# This is for the docs generation
public_api_router.include_router(run.router)
public_api_router.include_router(session.router)
public_api_router.include_router(machines.public_router)
# public_api_router.include_router(session.beta_router)
public_api_router.include_router(deployments.router)
public_api_router.include_router(files.router)
public_api_router.include_router(models.router)
public_api_router.include_router(search.router)
public_api_router.include_router(image_optimization.router)
# public_api_router.include_router(platform.router)
# public_api_router.include_router(run.webhook_router)

app.include_router(api_router, prefix="/api")  # Add the prefix here instead


# Set up logging

# # Hide sqlalchemy logs
# logging.getLogger('sqlalchemy').setLevel(logging.ERROR)

# Add CORS middleware
# app.add_middleware(SpendLimitMiddleware)
app.add_middleware(AutumnAccessMiddleware)
app.add_middleware(SubscriptionMiddleware)
app.add_middleware(AuthMiddleware)

# Get frontend URL from environment variable, default to localhost:3000 for development

allow_origins = []
allow_origin_regex = None

if os.getenv("ENV") == "development":
    allow_origins.append("http://localhost:3000")
    allow_origins.append("http://localhost:3001")
    # Allow wildcard subdomains for app.github.dev
    allow_origin_regex = r"https://.*\.app\.github\.dev"
elif os.getenv("ENV") == "staging":
    allow_origins.extend(
        [
            "http://localhost:3000",
            "http://localhost:3001",
            "https://staging.app.comfydeploy.com",
            "https://staging.studio.comfydeploy.com",
        ]
    )
    # Allow preview deployments hosted on Vercel and GitHub Codespaces
    allow_origin_regex = r"https://.*\.vercel\.app|https://.*\.app\.github\.dev"
else:
    allow_origins.extend(
        [
            # Production
            "https://app.comfydeploy.com",
            "https://studio.comfydeploy.com",
            # Staging
            "https://staging.app.comfydeploy.com",
            "https://staging.studio.comfydeploy.com",
        ]
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,  # Allow all subdomains of comfydeploy.com
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,  # Allow credentials (cookies)
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set up codegen integration if enabled
if feature_flags.is_enabled("organization_auth"):
    logger.info("Enabling codegen organization-based authentication")
    app = setup_codegen_integration(app)

logfire.instrument_fastapi(
    app,
    excluded_urls=r".*/api/update-run$",
)
# logfire.instrument_sqlalchemy(
#     engine=engine.sync_engine,
# )
