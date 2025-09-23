"""
Codegen-compatible middleware for organization-based authentication and rate limiting.

This module provides middleware components that implement codegen's authentication
patterns, including organization-based access control and rate limiting.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set, Tuple, Callable
import logging
import hashlib
import hmac
import json

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update
from sqlalchemy.orm import selectinload

from api.database import get_db_context
from api.codegen_models import (
    CodegenOrganization, CodegenOrganizationMembership, CodegenAPIKey, 
    CodegenRateLimit, OrganizationRole
)
from api.codegen_schemas import (
    APIRateLimitErrorResponse, PermissionsErrorResponse,
    validate_organization_id
)

logger = logging.getLogger(__name__)

# Rate limiting configuration matching codegen patterns
DEFAULT_RATE_LIMIT = 60  # requests per window
DEFAULT_WINDOW_SECONDS = 30  # 30-second window
RATE_LIMIT_HEADER_PREFIX = "X-RateLimit"

# Organization context key for request state
ORG_CONTEXT_KEY = "organization_context"


class OrganizationContext:
    """Organization context for requests."""
    
    def __init__(
        self,
        organization_id: int,
        organization: CodegenOrganization,
        user_id: str,
        membership: CodegenOrganizationMembership,
        api_key: Optional[CodegenAPIKey] = None
    ):
        self.organization_id = organization_id
        self.organization = organization
        self.user_id = user_id
        self.membership = membership
        self.api_key = api_key
        
    @property
    def user_role(self) -> OrganizationRole:
        """Get the user's role in the organization."""
        return self.membership.role
    
    @property
    def permissions(self) -> Set[str]:
        """Get the user's permissions in the organization."""
        base_permissions = set()
        
        # Role-based permissions
        if self.user_role == OrganizationRole.OWNER:
            base_permissions.update([
                "read", "write", "admin", "manage_users", "manage_projects",
                "manage_api_keys", "manage_settings"
            ])
        elif self.user_role == OrganizationRole.ADMIN:
            base_permissions.update([
                "read", "write", "manage_users", "manage_projects", "manage_api_keys"
            ])
        elif self.user_role == OrganizationRole.MEMBER:
            base_permissions.update(["read", "write"])
        elif self.user_role == OrganizationRole.VIEWER:
            base_permissions.update(["read"])
            
        # Additional permissions from membership
        base_permissions.update(self.membership.permissions or [])
        
        # API key specific permissions
        if self.api_key and self.api_key.scopes:
            base_permissions.intersection_update(self.api_key.scopes)
            
        return base_permissions
    
    def has_permission(self, permission: str) -> bool:
        """Check if the user has a specific permission."""
        return permission in self.permissions


class CodegenAuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for codegen-compatible organization-based auth.
    
    Handles API key validation, organization context injection, and permission
    checking for all organization-scoped endpoints.
    """
    
    def __init__(self, app, exclude_paths: Optional[Set[str]] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or {
            "/health", "/docs", "/redoc", "/openapi.json", 
            "/internal", "/internal/openapi.json"
        }
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through authentication middleware."""
        
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
            
        # Check if this is an organization-scoped endpoint
        if not self._is_organization_endpoint(request.url.path):
            # Non-organization endpoints use legacy auth
            return await call_next(request)
            
        try:
            # Extract and validate organization ID from path
            org_id = self._extract_organization_id(request.url.path)
            if not org_id:
                return self._create_error_response(
                    "invalid_organization_id",
                    "Invalid organization ID in URL path",
                    status.HTTP_400_BAD_REQUEST
                )
                
            # Validate API key and get user context
            api_key_header = request.headers.get("Authorization")
            if not api_key_header or not api_key_header.startswith("Bearer "):
                return self._create_error_response(
                    "missing_api_key",
                    "Missing or invalid Authorization header",
                    status.HTTP_401_UNAUTHORIZED
                )
                
            api_key = api_key_header.replace("Bearer ", "").strip()
            
            # Validate authentication and get organization context
            org_context = await self._validate_authentication(api_key, org_id)
            if not org_context:
                return self._create_error_response(
                    "invalid_authentication", 
                    "Invalid API key or insufficient permissions",
                    status.HTTP_401_UNAUTHORIZED
                )
                
            # Inject organization context into request state
            request.state.__setattr__(ORG_CONTEXT_KEY, org_context)
            
            # Continue to next middleware/endpoint
            response = await call_next(request)
            
            # Update API key usage statistics
            await self._update_api_key_usage(org_context.api_key)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Authentication middleware error: %s", str(e))
            return self._create_error_response(
                "authentication_error",
                "Internal authentication error",
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _is_organization_endpoint(self, path: str) -> bool:
        """Check if the endpoint is organization-scoped."""
        return path.startswith("/v1/organizations/") or path.startswith("/api/v1/organizations/")
    
    def _extract_organization_id(self, path: str) -> Optional[int]:
        """Extract organization ID from URL path."""
        try:
            # Handle both /v1/organizations/{org_id} and /api/v1/organizations/{org_id}
            parts = path.split("/")
            if "organizations" in parts:
                org_index = parts.index("organizations")
                if len(parts) > org_index + 1:
                    org_id = int(parts[org_index + 1])
                    return org_id if validate_organization_id(org_id) else None
        except (ValueError, IndexError):
            pass
        return None
    
    async def _validate_authentication(
        self, 
        api_key: str, 
        org_id: int
    ) -> Optional[OrganizationContext]:
        """Validate API key and return organization context."""
        
        # Hash the API key for database lookup
        api_key_hash = self._hash_api_key(api_key)
        
        async with get_db_context() as db:
            # Query for API key with organization and membership info
            result = await db.execute(
                select(CodegenAPIKey)
                .options(
                    selectinload(CodegenAPIKey.organization),
                    selectinload(CodegenAPIKey.user)
                )
                .where(
                    and_(
                        CodegenAPIKey.key_hash == api_key_hash,
                        CodegenAPIKey.is_active == True,
                        CodegenAPIKey.organization_id == org_id,
                        or_(
                            CodegenAPIKey.expires_at.is_(None),
                            CodegenAPIKey.expires_at > func.now()
                        )
                    )
                )
            )
            
            api_key_obj = result.scalar_one_or_none()
            if not api_key_obj:
                return None
                
            # Check if organization is active
            if api_key_obj.organization.status != "active":
                return None
                
            # Get user's membership in the organization
            membership_result = await db.execute(
                select(CodegenOrganizationMembership)
                .where(
                    and_(
                        CodegenOrganizationMembership.organization_id == org_id,
                        CodegenOrganizationMembership.user_id == api_key_obj.user_id
                    )
                )
            )
            
            membership = membership_result.scalar_one_or_none()
            if not membership:
                return None
                
            return OrganizationContext(
                organization_id=org_id,
                organization=api_key_obj.organization,
                user_id=api_key_obj.user_id,
                membership=membership,
                api_key=api_key_obj
            )
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage lookup."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def _update_api_key_usage(self, api_key: Optional[CodegenAPIKey]):
        """Update API key usage statistics."""
        if not api_key:
            return
            
        try:
            async with get_db_context() as db:
                await db.execute(
                    update(CodegenAPIKey)
                    .where(CodegenAPIKey.id == api_key.id)
                    .values(
                        last_used_at=func.now(),
                        usage_count=CodegenAPIKey.usage_count + 1
                    )
                )
                await db.commit()
        except Exception as e:
            logger.warning("Failed to update API key usage: %s", str(e))
    
    def _create_error_response(
        self, 
        error_code: str, 
        message: str, 
        status_code: int
    ) -> JSONResponse:
        """Create standardized error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": error_code,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


class CodegenRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware implementing codegen's rate limiting patterns.
    
    Enforces 60 requests per 30-second window per organization, with support
    for per-user and per-API key granular limits.
    """
    
    def __init__(self, app, default_limit: int = DEFAULT_RATE_LIMIT):
        super().__init__(app)
        self.default_limit = default_limit
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background task for cleaning up expired rate limit entries."""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_entries())
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiting middleware."""
        
        # Skip rate limiting for non-organization endpoints
        if not self._is_organization_endpoint(request.url.path):
            return await call_next(request)
        
        # Get organization context from previous middleware
        org_context = getattr(request.state, ORG_CONTEXT_KEY, None)
        if not org_context:
            # No organization context means auth failed - let it through
            return await call_next(request)
        
        try:
            # Check rate limit
            rate_limit_result = await self._check_rate_limit(request, org_context)
            
            if not rate_limit_result["allowed"]:
                # Rate limit exceeded
                error_response = APIRateLimitErrorResponse(
                    message=f"Rate limit exceeded. Max {rate_limit_result['limit']} requests per {DEFAULT_WINDOW_SECONDS} seconds.",
                    retry_after=rate_limit_result["retry_after"],
                    limit=rate_limit_result["limit"],
                    reset_at=datetime.now(timezone.utc) + timedelta(seconds=rate_limit_result["retry_after"])
                )
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content=error_response.model_dump(),
                    headers={
                        f"{RATE_LIMIT_HEADER_PREFIX}-Limit": str(rate_limit_result["limit"]),
                        f"{RATE_LIMIT_HEADER_PREFIX}-Remaining": "0",
                        f"{RATE_LIMIT_HEADER_PREFIX}-Reset": str(int(time.time() + rate_limit_result["retry_after"])),
                        "Retry-After": str(rate_limit_result["retry_after"])
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response.headers[f"{RATE_LIMIT_HEADER_PREFIX}-Limit"] = str(rate_limit_result["limit"])
            response.headers[f"{RATE_LIMIT_HEADER_PREFIX}-Remaining"] = str(rate_limit_result["remaining"])
            response.headers[f"{RATE_LIMIT_HEADER_PREFIX}-Reset"] = str(rate_limit_result["reset_at"])
            
            return response
            
        except Exception as e:
            logger.exception("Rate limiting middleware error: %s", str(e))
            # Don't block requests on rate limiting errors
            return await call_next(request)
    
    def _is_organization_endpoint(self, path: str) -> bool:
        """Check if the endpoint is organization-scoped."""
        return path.startswith("/v1/organizations/") or path.startswith("/api/v1/organizations/")
    
    async def _check_rate_limit(
        self, 
        request: Request, 
        org_context: OrganizationContext
    ) -> Dict[str, any]:
        """Check rate limit for the request."""
        
        endpoint_pattern = self._get_endpoint_pattern(request.url.path)
        current_time = datetime.now(timezone.utc)
        window_start = current_time - timedelta(seconds=DEFAULT_WINDOW_SECONDS)
        
        # Determine effective rate limit
        effective_limit = self.default_limit
        if org_context.organization:
            effective_limit = org_context.organization.rate_limit_per_minute
        if org_context.api_key and org_context.api_key.rate_limit_override:
            effective_limit = org_context.api_key.rate_limit_override
            
        async with get_db_context() as db:
            # Get or create rate limit entry
            result = await db.execute(
                select(CodegenRateLimit)
                .where(
                    and_(
                        CodegenRateLimit.organization_id == org_context.organization_id,
                        CodegenRateLimit.user_id == org_context.user_id,
                        CodegenRateLimit.endpoint_pattern == endpoint_pattern
                    )
                )
            )
            
            rate_limit_entry = result.scalar_one_or_none()
            
            if not rate_limit_entry:
                # Create new rate limit entry
                rate_limit_entry = CodegenRateLimit(
                    organization_id=org_context.organization_id,
                    user_id=org_context.user_id,
                    api_key_id=org_context.api_key.id if org_context.api_key else None,
                    endpoint_pattern=endpoint_pattern,
                    window_seconds=DEFAULT_WINDOW_SECONDS,
                    max_requests=effective_limit,
                    window_start=current_time,
                    current_count=1,
                    total_requests=1,
                    last_request_at=current_time
                )
                
                db.add(rate_limit_entry)
                await db.commit()
                
                return {
                    "allowed": True,
                    "limit": effective_limit,
                    "remaining": effective_limit - 1,
                    "reset_at": int((current_time + timedelta(seconds=DEFAULT_WINDOW_SECONDS)).timestamp())
                }
            
            # Check if we need to reset the window
            if rate_limit_entry.window_start < window_start:
                # Reset the window
                rate_limit_entry.window_start = current_time
                rate_limit_entry.current_count = 1
            else:
                # Check current count
                if rate_limit_entry.current_count >= effective_limit:
                    # Rate limit exceeded
                    rate_limit_entry.total_blocked += 1
                    await db.commit()
                    
                    retry_after = int((rate_limit_entry.window_start + timedelta(seconds=DEFAULT_WINDOW_SECONDS) - current_time).total_seconds())
                    
                    return {
                        "allowed": False,
                        "limit": effective_limit,
                        "remaining": 0,
                        "retry_after": max(1, retry_after),
                        "reset_at": int((rate_limit_entry.window_start + timedelta(seconds=DEFAULT_WINDOW_SECONDS)).timestamp())
                    }
                
                # Increment count
                rate_limit_entry.current_count += 1
            
            # Update statistics
            rate_limit_entry.total_requests += 1
            rate_limit_entry.last_request_at = current_time
            await db.commit()
            
            return {
                "allowed": True,
                "limit": effective_limit,
                "remaining": effective_limit - rate_limit_entry.current_count,
                "reset_at": int((rate_limit_entry.window_start + timedelta(seconds=DEFAULT_WINDOW_SECONDS)).timestamp())
            }
    
    def _get_endpoint_pattern(self, path: str) -> str:
        """Get endpoint pattern for rate limiting."""
        # Normalize organization-specific paths to patterns
        # e.g., /v1/organizations/123/users -> /v1/organizations/*/users
        parts = path.split("/")
        
        if "organizations" in parts:
            org_index = parts.index("organizations")
            if len(parts) > org_index + 1:
                # Replace org ID with wildcard
                parts[org_index + 1] = "*"
        
        return "/".join(parts)
    
    async def _cleanup_expired_entries(self):
        """Background task to clean up expired rate limit entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
                
                async with get_db_context() as db:
                    result = await db.execute(
                        select(func.count(CodegenRateLimit.id))
                        .where(CodegenRateLimit.last_request_at < cutoff_time)
                    )
                    
                    count_to_delete = result.scalar() or 0
                    
                    if count_to_delete > 0:
                        await db.execute(
                            CodegenRateLimit.__table__.delete()
                            .where(CodegenRateLimit.last_request_at < cutoff_time)
                        )
                        await db.commit()
                        logger.info(f"Cleaned up {count_to_delete} expired rate limit entries")
                        
            except Exception as e:
                logger.exception("Error in rate limit cleanup task: %s", str(e))


class CodegenPermissionMiddleware(BaseHTTPMiddleware):
    """
    Permission checking middleware for fine-grained access control.
    
    Validates user permissions for specific operations based on organization
    roles and endpoint-specific requirements.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.endpoint_permissions = self._build_endpoint_permission_map()
    
    def _build_endpoint_permission_map(self) -> Dict[Tuple[str, str], str]:
        """Build mapping of (method, path_pattern) -> required_permission."""
        return {
            # User endpoints
            ("GET", "/v1/organizations/*/users"): "read",
            ("GET", "/v1/organizations/*/users/*"): "read",
            ("POST", "/v1/organizations/*/users"): "manage_users",
            ("PUT", "/v1/organizations/*/users/*"): "manage_users",
            ("DELETE", "/v1/organizations/*/users/*"): "manage_users",
            
            # Project endpoints
            ("GET", "/v1/organizations/*/projects"): "read",
            ("GET", "/v1/organizations/*/projects/*"): "read",
            ("POST", "/v1/organizations/*/projects"): "write",
            ("PUT", "/v1/organizations/*/projects/*"): "write",
            ("DELETE", "/v1/organizations/*/projects/*"): "manage_projects",
            
            # Organization management
            ("GET", "/v1/organizations/*"): "read",
            ("PUT", "/v1/organizations/*"): "admin",
            ("DELETE", "/v1/organizations/*"): "admin",
            ("POST", "/v1/organizations/*/members"): "manage_users",
            ("PUT", "/v1/organizations/*/members/*"): "manage_users",
            ("DELETE", "/v1/organizations/*/members/*"): "manage_users",
            
            # API key management
            ("GET", "/v1/organizations/*/api-keys"): "manage_api_keys",
            ("POST", "/v1/organizations/*/api-keys"): "manage_api_keys",
            ("PUT", "/v1/organizations/*/api-keys/*"): "manage_api_keys",
            ("DELETE", "/v1/organizations/*/api-keys/*"): "manage_api_keys",
            
            # Session and agent management
            ("GET", "/v1/organizations/*/projects/*/sessions"): "read",
            ("POST", "/v1/organizations/*/projects/*/sessions"): "write",
            ("GET", "/v1/organizations/*/projects/*/agents"): "read",
            ("POST", "/v1/organizations/*/projects/*/agents"): "write",
            
            # Pro Mode endpoints
            ("POST", "/v1/organizations/*/projects/*/pro-mode"): "write",
            ("GET", "/v1/organizations/*/projects/*/pro-mode/*"): "read",
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through permission middleware."""
        
        # Skip permission checking for non-organization endpoints
        if not self._is_organization_endpoint(request.url.path):
            return await call_next(request)
        
        # Get organization context from auth middleware
        org_context = getattr(request.state, ORG_CONTEXT_KEY, None)
        if not org_context:
            return await call_next(request)
        
        # Check permissions for this endpoint
        required_permission = self._get_required_permission(
            request.method, request.url.path
        )
        
        if required_permission and not org_context.has_permission(required_permission):
            error_response = PermissionsErrorResponse(
                message=f"Insufficient permissions. Required: {required_permission}",
                required_permission=required_permission,
                user_role=org_context.user_role.value
            )
            
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content=error_response.model_dump()
            )
        
        return await call_next(request)
    
    def _is_organization_endpoint(self, path: str) -> bool:
        """Check if the endpoint is organization-scoped."""
        return path.startswith("/v1/organizations/") or path.startswith("/api/v1/organizations/")
    
    def _get_required_permission(self, method: str, path: str) -> Optional[str]:
        """Get required permission for method and path."""
        # Normalize path to pattern
        path_pattern = self._normalize_path_to_pattern(path)
        return self.endpoint_permissions.get((method, path_pattern))
    
    def _normalize_path_to_pattern(self, path: str) -> str:
        """Normalize path to pattern for permission matching."""
        parts = path.split("/")
        
        # Replace numeric IDs and UUIDs with wildcards
        for i, part in enumerate(parts):
            if part.isdigit() or self._is_uuid(part):
                parts[i] = "*"
        
        return "/".join(parts)
    
    def _is_uuid(self, value: str) -> bool:
        """Check if a string looks like a UUID."""
        try:
            import uuid
            uuid.UUID(value)
            return True
        except ValueError:
            return False


# Utility functions for middleware integration
def get_organization_context(request: Request) -> Optional[OrganizationContext]:
    """Get organization context from request state."""
    return getattr(request.state, ORG_CONTEXT_KEY, None)


def require_permission(permission: str):
    """Dependency to require specific permission."""
    def check_permission(request: Request):
        org_context = get_organization_context(request)
        if not org_context:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No organization context available"
            )
        
        if not org_context.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        
        return org_context
    
    return check_permission


def require_role(required_role: OrganizationRole):
    """Dependency to require specific organization role."""
    def check_role(request: Request):
        org_context = get_organization_context(request)
        if not org_context:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No organization context available"
            )
        
        role_hierarchy = {
            OrganizationRole.VIEWER: 1,
            OrganizationRole.MEMBER: 2,
            OrganizationRole.ADMIN: 3,
            OrganizationRole.OWNER: 4
        }
        
        user_level = role_hierarchy.get(org_context.user_role, 0)
        required_level = role_hierarchy.get(required_role, 999)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: {required_role.value} or higher"
            )
        
        return org_context
    
    return check_role


# Export all middleware components
__all__ = [
    'OrganizationContext',
    'CodegenAuthMiddleware',
    'CodegenRateLimitMiddleware', 
    'CodegenPermissionMiddleware',
    'get_organization_context',
    'require_permission',
    'require_role',
    'ORG_CONTEXT_KEY'
]