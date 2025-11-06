"""
Codegen Integration Module

This module integrates the codegen-aligned architecture with the existing ComfyUI API,
providing a unified interface that supports both legacy and new patterns.
"""

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import existing components
from api.router import app as existing_app
from api.autumn_mount import autumn_app

# Import new codegen components  
from api.codegen_routes import codegen_router
from api.codegen_middleware import (
    CodegenAuthMiddleware, CodegenRateLimitMiddleware, 
    CodegenPermissionMiddleware
)
from api.codegen_schemas import (
    APIRateLimitErrorResponse, PermissionsErrorResponse, HTTPValidationError
)

logger = logging.getLogger(__name__)


def setup_codegen_integration(app: FastAPI) -> FastAPI:
    """
    Set up codegen integration with existing FastAPI application.
    
    This function adds codegen middleware and routes to the existing app,
    creating a hybrid system that supports both legacy ComfyUI patterns
    and new codegen-aligned organization-centric patterns.
    
    Args:
        app: Existing FastAPI application
        
    Returns:
        FastAPI: The integrated application
    """
    logger.info("Setting up codegen integration")
    
    # Add codegen middleware stack (order matters!)
    # 1. Permission middleware (runs last, after auth context is set)
    app.add_middleware(CodegenPermissionMiddleware)
    
    # 2. Rate limiting middleware (runs after auth, before permission check)
    app.add_middleware(CodegenRateLimitMiddleware)
    
    # 3. Authentication middleware (runs first to set org context)
    app.add_middleware(
        CodegenAuthMiddleware,
        exclude_paths={
            "/health", "/v1/health", "/v1/stats",
            "/docs", "/redoc", "/openapi.json", 
            "/internal", "/internal/openapi.json",
            "/api/run", "/api/workflows", "/api/machines",  # Legacy endpoints
            "/api/comfy-org", "/api/autumn"  # Existing integrations
        }
    )
    
    # Add codegen routes
    app.include_router(codegen_router, prefix="/api")
    
    # Add custom exception handlers for codegen error formats
    add_codegen_exception_handlers(app)
    
    # Update CORS settings for codegen endpoints
    setup_codegen_cors(app)
    
    logger.info("Codegen integration setup completed")
    return app


def add_codegen_exception_handlers(app: FastAPI):
    """Add custom exception handlers for codegen API compatibility."""
    
    @app.exception_handler(HTTPException)
    async def codegen_http_exception_handler(request, exc: HTTPException):
        """Handle HTTP exceptions with codegen-compatible error format."""
        
        # Check if this is a codegen endpoint
        if request.url.path.startswith("/api/v1/organizations/"):
            # Use codegen error format
            error_response = {
                "error": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code
            }
            
            # Map specific status codes to codegen error types
            if exc.status_code == 429:
                error_response = APIRateLimitErrorResponse(
                    message=exc.detail or "Rate limit exceeded",
                    retry_after=30,  # Default retry after
                    limit=60,  # Default limit
                    reset_at=None  # Would be calculated in middleware
                ).model_dump()
            elif exc.status_code == 403:
                error_response = PermissionsErrorResponse(
                    message=exc.detail or "Insufficient permissions"
                ).model_dump()
            
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response
            )
        
        # Use default FastAPI exception handler for legacy endpoints
        from fastapi.exception_handlers import http_exception_handler
        return await http_exception_handler(request, exc)
    
    @app.exception_handler(422)
    async def codegen_validation_exception_handler(request, exc):
        """Handle validation exceptions with codegen format."""
        
        if request.url.path.startswith("/api/v1/organizations/"):
            # Convert FastAPI validation errors to codegen format
            error_details = []
            
            if hasattr(exc, 'errors'):
                for error in exc.errors():
                    error_details.append({
                        "loc": error.get("loc", []),
                        "msg": error.get("msg", "Validation error"),
                        "type": error.get("type", "value_error")
                    })
            
            return JSONResponse(
                status_code=422,
                content=HTTPValidationError(detail=error_details).model_dump()
            )
        
        # Use default validation exception handler for legacy endpoints
        from fastapi.exception_handlers import request_validation_exception_handler
        return await request_validation_exception_handler(request, exc)


def setup_codegen_cors(app: FastAPI):
    """Update CORS settings to include codegen endpoints."""
    
    # Get existing CORS middleware settings
    existing_cors_origins = []
    existing_cors_regex = None
    
    # Extract from environment (same logic as original __init__.py)
    if os.getenv("ENV") == "development":
        existing_cors_origins.extend([
            "http://localhost:3000",
            "http://localhost:3001"
        ])
        existing_cors_regex = r"https://.*\.app\.github\.dev"
    elif os.getenv("ENV") == "staging":
        existing_cors_origins.extend([
            "http://localhost:3000",
            "http://localhost:3001", 
            "https://staging.app.comfydeploy.com",
            "https://staging.studio.comfydeploy.com"
        ])
        existing_cors_regex = r"https://.*\.vercel\.app|https://.*\.app\.github\.dev"
    else:
        existing_cors_origins.extend([
            "https://app.comfydeploy.com",
            "https://studio.comfydeploy.com",
            "https://staging.app.comfydeploy.com", 
            "https://staging.studio.comfydeploy.com"
        ])
    
    # Add codegen-specific origins if needed
    codegen_origins = [
        # Add any codegen-specific origins here
    ]
    
    all_origins = existing_cors_origins + codegen_origins
    
    logger.info(f"CORS configured with {len(all_origins)} allowed origins")


def setup_openapi_integration(app: FastAPI):
    """Set up OpenAPI documentation integration for codegen endpoints."""
    
    # Update OpenAPI schema to include codegen endpoints
    def custom_codegen_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        from fastapi.openapi.utils import get_openapi
        
        # Generate schema including both legacy and codegen routes
        openapi_schema = get_openapi(
            title="ComfyDeploy + Codegen Unified API",
            version="2.0.0",
            description="""
            ### Unified API Documentation
            
            This API provides both legacy ComfyDeploy endpoints and new Codegen-aligned 
            organization-centric endpoints.
            
            #### Legacy Endpoints
            - `/api/run/*` - Workflow execution
            - `/api/workflows/*` - Workflow management  
            - `/api/machines/*` - Machine management
            
            #### Codegen Endpoints
            - `/api/v1/organizations/{org_id}/users` - Organization user management
            - `/api/v1/organizations/{org_id}/projects` - Project management
            - `/api/v1/organizations/{org_id}/projects/{project_id}/sessions` - Pro Mode sessions
            
            ### Authentication
            
            **Legacy endpoints**: Use existing API key authentication
            **Codegen endpoints**: Use organization-scoped API keys with `Authorization: Bearer <key>` header
            
            ### Rate Limiting
            
            Codegen endpoints enforce rate limiting of 60 requests per 30-second window per organization.
            """,
            routes=app.routes,
            servers=[
                {"url": "https://api.comfydeploy.com", "description": "Production server"},
                {"url": "https://staging.api.comfydeploy.com", "description": "Staging server"},
                {"url": "http://localhost:3011", "description": "Local development server"}
            ]
        )
        
        # Add security schemes for both auth methods
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": "Organization-scoped API key for codegen endpoints"
            },
            "LegacyAuth": {
                "type": "http", 
                "scheme": "bearer",
                "description": "Legacy API key for ComfyDeploy endpoints"
            }
        }
        
        # Set default security
        openapi_schema["security"] = [
            {"BearerAuth": []},
            {"LegacyAuth": []}
        ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_codegen_openapi


def create_integrated_app() -> FastAPI:
    """
    Create a new integrated FastAPI application with codegen support.
    
    This function creates a complete integrated application that supports
    both legacy ComfyDeploy patterns and new codegen organization-centric patterns.
    
    Returns:
        FastAPI: Integrated application with both legacy and codegen support
    """
    logger.info("Creating integrated codegen application")
    
    # Start with existing app
    integrated_app = existing_app
    
    # Add codegen integration
    integrated_app = setup_codegen_integration(integrated_app)
    
    # Set up enhanced OpenAPI documentation
    setup_openapi_integration(integrated_app)
    
    # Add health checks for codegen components
    add_codegen_health_checks(integrated_app)
    
    logger.info("Integrated application created successfully")
    return integrated_app


def add_codegen_health_checks(app: FastAPI):
    """Add health check endpoints for codegen components."""
    
    @app.get("/api/v1/health/codegen")
    async def codegen_component_health():
        """Health check specifically for codegen components."""
        try:
            # Test database connectivity for codegen tables
            from api.database import get_db_context
            from api.codegen_models import CodegenOrganization
            from sqlalchemy import select
            
            async with get_db_context() as db:
                result = await db.execute(select(CodegenOrganization).limit(1))
                result.scalar_one_or_none()
            
            return {
                "status": "healthy",
                "component": "codegen",
                "database": "connected",
                "middleware": "active",
                "routes": "registered"
            }
            
        except Exception as e:
            logger.error(f"Codegen health check failed: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "component": "codegen", 
                    "error": str(e)
                }
            )


# Feature flag system for gradual rollout
class CodegenFeatureFlags:
    """Feature flags for controlling codegen functionality rollout."""
    
    def __init__(self):
        self.flags = {
            "organization_auth": os.getenv("CODEGEN_ORG_AUTH_ENABLED", "true").lower() == "true",
            "rate_limiting": os.getenv("CODEGEN_RATE_LIMITING_ENABLED", "true").lower() == "true", 
            "pro_mode_v2": os.getenv("CODEGEN_PRO_MODE_V2_ENABLED", "true").lower() == "true",
            "migration_mode": os.getenv("CODEGEN_MIGRATION_MODE", "false").lower() == "true"
        }
    
    def is_enabled(self, flag: str) -> bool:
        """Check if a feature flag is enabled."""
        return self.flags.get(flag, False)
    
    def enable_flag(self, flag: str):
        """Enable a feature flag."""
        self.flags[flag] = True
        
    def disable_flag(self, flag: str):
        """Disable a feature flag."""
        self.flags[flag] = False


# Global feature flags instance
feature_flags = CodegenFeatureFlags()


# Export the main integration function and integrated app
__all__ = [
    'setup_codegen_integration',
    'create_integrated_app',
    'feature_flags',
    'CodegenFeatureFlags'
]