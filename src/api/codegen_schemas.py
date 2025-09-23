"""
Pydantic schemas for codegen API compatibility.

This module provides Pydantic models that match the codegen project's API
response patterns, including pagination, error handling, and validation schemas.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Generic, TypeVar, Union
from decimal import Decimal
from pydantic import BaseModel, Field, validator, ConfigDict
from uuid import UUID
import enum

# Generic type for paginated responses
T = TypeVar('T')

class CodegenBaseModel(BaseModel):
    """Base model with common configuration for codegen API compatibility."""
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat()[:-3] + "Z" if v else None,
            UUID: lambda v: str(v) if v else None,
            Decimal: lambda v: str(v) if v else None,
        }
    )


# Enum classes for API responses
class OrganizationRoleEnum(str, enum.Enum):
    """Organization role enumeration for API responses"""
    OWNER = "owner"
    ADMIN = "admin" 
    MEMBER = "member"
    VIEWER = "viewer"


class OrganizationStatusEnum(str, enum.Enum):
    """Organization status enumeration for API responses"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"


class ProjectStatusEnum(str, enum.Enum):
    """Project status enumeration for API responses"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DRAFT = "draft"


class SessionStatusEnum(str, enum.Enum):
    """Session status enumeration for API responses"""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AgentInstanceStatusEnum(str, enum.Enum):
    """Agent instance status enumeration for API responses"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# User-related schemas matching codegen patterns
class UserResponse(CodegenBaseModel):
    """User response model matching codegen's user API patterns."""
    id: int = Field(..., description="Unique user identifier")
    name: Optional[str] = Field(None, description="User's full name")
    username: Optional[str] = Field(None, description="User's username")
    email: Optional[str] = Field(None, description="User's email address")
    avatar_url: Optional[str] = Field(None, description="URL to user's avatar image")
    github_username: Optional[str] = Field(None, description="User's GitHub username")
    created_at: datetime = Field(..., description="User creation timestamp")
    updated_at: datetime = Field(..., description="User last update timestamp")


class OrganizationResponse(CodegenBaseModel):
    """Organization response model for codegen API compatibility."""
    id: int = Field(..., description="Unique organization identifier")
    name: str = Field(..., description="Organization name")
    display_name: Optional[str] = Field(None, description="Organization display name")
    description: Optional[str] = Field(None, description="Organization description")
    status: OrganizationStatusEnum = Field(..., description="Organization status")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Organization settings")
    rate_limit_per_minute: int = Field(60, description="API rate limit per minute")
    max_users: int = Field(100, description="Maximum number of users")
    max_projects: int = Field(50, description="Maximum number of projects")
    created_at: datetime = Field(..., description="Organization creation timestamp")
    updated_at: datetime = Field(..., description="Organization last update timestamp")


class OrganizationMembershipResponse(CodegenBaseModel):
    """Organization membership response model."""
    id: UUID = Field(..., description="Membership identifier")
    organization_id: int = Field(..., description="Organization identifier")
    user_id: str = Field(..., description="User identifier")
    role: OrganizationRoleEnum = Field(..., description="User role in organization")
    permissions: List[str] = Field(default_factory=list, description="Additional permissions")
    joined_at: datetime = Field(..., description="Membership start date")
    last_active_at: Optional[datetime] = Field(None, description="Last activity timestamp")
    user: Optional[UserResponse] = Field(None, description="User details")


class ProjectResponse(CodegenBaseModel):
    """Project response model for organization-scoped projects."""
    id: UUID = Field(..., description="Unique project identifier")
    organization_id: int = Field(..., description="Organization identifier")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    status: ProjectStatusEnum = Field(..., description="Project status")
    config: Dict[str, Any] = Field(default_factory=dict, description="Project configuration")
    repository_url: Optional[str] = Field(None, description="Repository URL")
    default_branch: str = Field("main", description="Default git branch")
    owner_id: Optional[str] = Field(None, description="Project owner user ID")
    collaborators: List[str] = Field(default_factory=list, description="Collaborator user IDs")
    created_at: datetime = Field(..., description="Project creation timestamp")
    updated_at: datetime = Field(..., description="Project last update timestamp")
    archived_at: Optional[datetime] = Field(None, description="Project archive timestamp")
    owner: Optional[UserResponse] = Field(None, description="Project owner details")


class AgentInstanceResponse(CodegenBaseModel):
    """Agent instance response model for Pro Mode agent tracking."""
    id: UUID = Field(..., description="Agent instance identifier")
    organization_id: int = Field(..., description="Organization identifier")
    project_id: UUID = Field(..., description="Project identifier")
    session_id: Optional[UUID] = Field(None, description="Session identifier")
    agent_name: str = Field(..., description="Agent name")
    agent_version: Optional[str] = Field(None, description="Agent version")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    status: AgentInstanceStatusEnum = Field(..., description="Agent execution status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Execution progress (0.0 to 1.0)")
    started_at: Optional[datetime] = Field(None, description="Execution start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Execution completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Agent inputs")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Agent outputs")
    execution_log: List[Dict[str, Any]] = Field(default_factory=list, description="Execution log entries")
    cpu_time_seconds: Optional[float] = Field(None, description="CPU time used in seconds")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class SessionResponse(CodegenBaseModel):
    """Session response model for Pro Mode session management."""
    id: UUID = Field(..., description="Session identifier")
    organization_id: int = Field(..., description="Organization identifier")
    project_id: UUID = Field(..., description="Project identifier")
    user_id: str = Field(..., description="User identifier")
    name: str = Field(..., description="Session name")
    description: Optional[str] = Field(None, description="Session description")
    config: Dict[str, Any] = Field(default_factory=dict, description="Session configuration")
    status: SessionStatusEnum = Field(..., description="Session status")
    total_agents: int = Field(0, description="Total number of agents")
    completed_agents: int = Field(0, description="Number of completed agents")
    failed_agents: int = Field(0, description="Number of failed agents")
    synthesis_strategy: str = Field("tournament", description="Tournament synthesis strategy")
    max_parallel_agents: int = Field(3, description="Maximum parallel agents")
    timeout_minutes: int = Field(60, description="Session timeout in minutes")
    started_at: Optional[datetime] = Field(None, description="Session start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Session completion timestamp")
    final_output: Dict[str, Any] = Field(default_factory=dict, description="Final session output")
    synthesis_results: List[Dict[str, Any]] = Field(default_factory=list, description="Synthesis results")
    session_log: List[Dict[str, Any]] = Field(default_factory=list, description="Session log entries")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    user: Optional[UserResponse] = Field(None, description="User details")
    project: Optional[ProjectResponse] = Field(None, description="Project details")
    agent_instances: Optional[List[AgentInstanceResponse]] = Field(None, description="Agent instances")


# Pagination models matching codegen patterns
class PaginationMeta(CodegenBaseModel):
    """Pagination metadata matching codegen patterns."""
    total: int = Field(..., description="Total number of items")
    skip: int = Field(0, description="Number of items skipped")
    limit: int = Field(100, description="Maximum items per page")
    has_more: bool = Field(..., description="Whether more items are available")


class Page(CodegenBaseModel, Generic[T]):
    """Generic paginated response matching codegen's Page pattern."""
    items: List[T] = Field(..., description="List of items")
    meta: PaginationMeta = Field(..., description="Pagination metadata")


# Specific paginated response types
class Page_UserResponse_(Page[UserResponse]):
    """Paginated user response matching codegen's exact schema name."""
    pass


class Page_OrganizationResponse_(Page[OrganizationResponse]):
    """Paginated organization response."""
    pass


class Page_ProjectResponse_(Page[ProjectResponse]):
    """Paginated project response."""
    pass


class Page_AgentInstanceResponse_(Page[AgentInstanceResponse]):
    """Paginated agent instance response."""
    pass


class Page_SessionResponse_(Page[SessionResponse]):
    """Paginated session response."""
    pass


# Error response models matching codegen patterns
class APIErrorDetail(CodegenBaseModel):
    """API error detail model."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")


class APIRateLimitErrorResponse(CodegenBaseModel):
    """Rate limit error response matching codegen patterns."""
    error: str = Field("rate_limit_exceeded", description="Error type")
    message: str = Field(..., description="Error message")
    retry_after: int = Field(..., description="Seconds to wait before retrying")
    limit: int = Field(..., description="Rate limit threshold")
    reset_at: datetime = Field(..., description="When the rate limit resets")


class PermissionsErrorResponse(CodegenBaseModel):
    """Permissions error response matching codegen patterns."""
    error: str = Field("insufficient_permissions", description="Error type")
    message: str = Field(..., description="Error message")
    required_permission: Optional[str] = Field(None, description="Required permission")
    user_role: Optional[str] = Field(None, description="User's current role")


class ValidationErrorDetail(CodegenBaseModel):
    """Validation error detail."""
    loc: List[Union[str, int]] = Field(..., description="Error location")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class HTTPValidationError(CodegenBaseModel):
    """HTTP validation error response matching FastAPI/codegen patterns."""
    detail: List[ValidationErrorDetail] = Field(..., description="Validation error details")


# Request models for API endpoints
class CreateOrganizationRequest(CodegenBaseModel):
    """Request model for creating organizations."""
    name: str = Field(..., min_length=1, max_length=255, description="Organization name")
    display_name: Optional[str] = Field(None, max_length=255, description="Display name")
    description: Optional[str] = Field(None, description="Organization description")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Organization settings")


class UpdateOrganizationRequest(CodegenBaseModel):
    """Request model for updating organizations."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Organization name")
    display_name: Optional[str] = Field(None, max_length=255, description="Display name")
    description: Optional[str] = Field(None, description="Organization description")
    settings: Optional[Dict[str, Any]] = Field(None, description="Organization settings")
    status: Optional[OrganizationStatusEnum] = Field(None, description="Organization status")


class CreateProjectRequest(CodegenBaseModel):
    """Request model for creating projects."""
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    config: Dict[str, Any] = Field(default_factory=dict, description="Project configuration")
    repository_url: Optional[str] = Field(None, description="Repository URL")
    default_branch: str = Field("main", description="Default git branch")
    collaborators: List[str] = Field(default_factory=list, description="Collaborator user IDs")


class UpdateProjectRequest(CodegenBaseModel):
    """Request model for updating projects."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    status: Optional[ProjectStatusEnum] = Field(None, description="Project status")
    config: Optional[Dict[str, Any]] = Field(None, description="Project configuration")
    repository_url: Optional[str] = Field(None, description="Repository URL")
    default_branch: Optional[str] = Field(None, description="Default git branch")
    collaborators: Optional[List[str]] = Field(None, description="Collaborator user IDs")


class CreateAgentInstanceRequest(CodegenBaseModel):
    """Request model for creating agent instances."""
    agent_name: str = Field(..., min_length=1, max_length=255, description="Agent name")
    agent_version: Optional[str] = Field(None, description="Agent version")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Agent inputs")
    session_id: Optional[UUID] = Field(None, description="Session identifier")


class CreateSessionRequest(CodegenBaseModel):
    """Request model for creating Pro Mode sessions."""
    name: str = Field(..., min_length=1, max_length=255, description="Session name")
    description: Optional[str] = Field(None, description="Session description")
    config: Dict[str, Any] = Field(default_factory=dict, description="Session configuration")
    synthesis_strategy: str = Field("tournament", description="Tournament synthesis strategy")
    max_parallel_agents: int = Field(3, ge=1, le=10, description="Maximum parallel agents")
    timeout_minutes: int = Field(60, ge=5, le=1440, description="Session timeout in minutes")


class AddOrganizationMemberRequest(CodegenBaseModel):
    """Request model for adding organization members."""
    user_id: str = Field(..., description="User identifier")
    role: OrganizationRoleEnum = Field(OrganizationRoleEnum.MEMBER, description="User role")
    permissions: List[str] = Field(default_factory=list, description="Additional permissions")


class UpdateOrganizationMemberRequest(CodegenBaseModel):
    """Request model for updating organization members."""
    role: Optional[OrganizationRoleEnum] = Field(None, description="User role")
    permissions: Optional[List[str]] = Field(None, description="Additional permissions")


# Pro Mode specific schemas
class ProModeRequest(CodegenBaseModel):
    """Pro Mode tournament synthesis request matching existing patterns."""
    prompt: str = Field(..., description="The task prompt for agents")
    agents: List[str] = Field(default_factory=lambda: ["general-purpose"], description="Agent types to use")
    max_iterations: int = Field(3, ge=1, le=10, description="Maximum tournament iterations")
    synthesis_strategy: str = Field("tournament", description="Synthesis strategy")
    timeout_minutes: int = Field(60, ge=5, le=1440, description="Overall timeout")
    project_id: Optional[UUID] = Field(None, description="Project identifier")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")


class ProModeResponse(CodegenBaseModel):
    """Pro Mode response model."""
    session_id: UUID = Field(..., description="Session identifier")
    status: str = Field(..., description="Session status")
    message: str = Field(..., description="Status message")
    agents_created: int = Field(..., description="Number of agents created")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


# Health and status schemas
class HealthResponse(CodegenBaseModel):
    """Health check response model."""
    status: str = Field("healthy", description="Service status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Check timestamp")
    version: str = Field("1.0.0", description="API version")
    database_status: str = Field("connected", description="Database connection status")
    dependencies: Dict[str, str] = Field(default_factory=dict, description="Dependency statuses")


class StatsResponse(CodegenBaseModel):
    """Statistics response model."""
    total_organizations: int = Field(..., description="Total organizations")
    total_users: int = Field(..., description="Total users")
    total_projects: int = Field(..., description="Total projects")
    active_sessions: int = Field(..., description="Active sessions")
    total_agent_instances: int = Field(..., description="Total agent instances")
    api_requests_last_hour: int = Field(..., description="API requests in last hour")


# Validation helpers
def validate_organization_id(v: int) -> int:
    """Validate organization ID format."""
    if not isinstance(v, int) or v <= 0:
        raise ValueError("Organization ID must be a positive integer")
    return v


def validate_pagination_params(skip: int = 0, limit: int = 100) -> tuple[int, int]:
    """Validate and normalize pagination parameters."""
    skip = max(0, skip)
    limit = max(1, min(100, limit))  # Codegen uses max 100 items per page
    return skip, limit


# Export all schemas for easy importing
__all__ = [
    'CodegenBaseModel',
    'OrganizationRoleEnum',
    'OrganizationStatusEnum', 
    'ProjectStatusEnum',
    'SessionStatusEnum',
    'AgentInstanceStatusEnum',
    'UserResponse',
    'OrganizationResponse',
    'OrganizationMembershipResponse',
    'ProjectResponse',
    'AgentInstanceResponse',
    'SessionResponse',
    'PaginationMeta',
    'Page',
    'Page_UserResponse_',
    'Page_OrganizationResponse_',
    'Page_ProjectResponse_',
    'Page_AgentInstanceResponse_',
    'Page_SessionResponse_',
    'APIRateLimitErrorResponse',
    'PermissionsErrorResponse',
    'HTTPValidationError',
    'ValidationErrorDetail',
    'CreateOrganizationRequest',
    'UpdateOrganizationRequest',
    'CreateProjectRequest',
    'UpdateProjectRequest',
    'CreateAgentInstanceRequest',
    'CreateSessionRequest',
    'AddOrganizationMemberRequest',
    'UpdateOrganizationMemberRequest',
    'ProModeRequest',
    'ProModeResponse',
    'HealthResponse',
    'StatsResponse',
    'validate_organization_id',
    'validate_pagination_params'
]