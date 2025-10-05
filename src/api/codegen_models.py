"""
Codegen-aligned database models for organization-centric API architecture.

This module provides SQLAlchemy models that match the codegen project's
organizational patterns, implementing the /v1/organizations/{org_id}/* API structure.
"""

from datetime import datetime, timezone
import enum
import uuid
from typing import List, Optional, Dict, Any
from decimal import Decimal

from sqlalchemy import (
    BigInteger, Column, String, Enum, DateTime, Boolean, MetaData, Float, 
    JSON, ForeignKey, Integer, func, Text, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.inspection import inspect as sqlalchemy_inspect
import json

# Use the same metadata as existing models for schema consistency
metadata = MetaData(schema="comfyui_deploy")

# Import the existing SerializableMixin for consistent serialization
from api.models import SerializableMixin

Base = declarative_base(metadata=metadata)

class OrganizationRole(str, enum.Enum):
    """User roles within an organization"""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

class OrganizationStatus(str, enum.Enum):
    """Organization status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"

class ProjectStatus(str, enum.Enum):
    """Project status within organizations"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DRAFT = "draft"

class AgentInstanceStatus(str, enum.Enum):
    """Status of codegen agent instances"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SessionStatus(str, enum.Enum):
    """Status of Pro Mode sessions"""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class CodegenOrganization(SerializableMixin, Base):
    """
    Organization model aligned with codegen's organizational patterns.
    
    Supports the /v1/organizations/{org_id}/* API structure and provides
    the foundation for all organization-scoped resources.
    """
    __tablename__ = "codegen_organizations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(Enum(OrganizationStatus), nullable=False, default=OrganizationStatus.ACTIVE)
    
    # Organization settings
    settings = Column(JSON, default=dict)
    rate_limit_per_minute = Column(Integer, default=60)  # Codegen uses 60 requests per 30 seconds
    max_users = Column(Integer, default=100)
    max_projects = Column(Integer, default=50)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    memberships = relationship("CodegenOrganizationMembership", back_populates="organization", cascade="all, delete-orphan")
    projects = relationship("CodegenProject", back_populates="organization", cascade="all, delete-orphan")
    api_keys = relationship("CodegenAPIKey", back_populates="organization", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_codegen_organizations_status', 'status'),
        Index('ix_codegen_organizations_created_at', 'created_at'),
    )


class CodegenOrganizationMembership(SerializableMixin, Base):
    """
    Organization membership model tracking user roles and permissions.
    
    Links users to organizations with specific roles and manages access control
    for organization-scoped resources.
    """
    __tablename__ = "codegen_organization_memberships"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(Integer, ForeignKey("codegen_organizations.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    role = Column(Enum(OrganizationRole), nullable=False, default=OrganizationRole.MEMBER)
    permissions = Column(JSON, default=list)  # Additional fine-grained permissions
    
    # Membership metadata
    joined_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    invited_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    last_active_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    organization = relationship("CodegenOrganization", back_populates="memberships")
    user = relationship("User", foreign_keys=[user_id])
    inviter = relationship("User", foreign_keys=[invited_by])
    
    # Unique constraint to prevent duplicate memberships
    __table_args__ = (
        Index('ix_codegen_org_memberships_org_user', 'organization_id', 'user_id', unique=True),
        Index('ix_codegen_org_memberships_user_id', 'user_id'),
        Index('ix_codegen_org_memberships_role', 'role'),
    )


class CodegenProject(SerializableMixin, Base):
    """
    Project model for organization-scoped project management.
    
    Represents projects within organizations, supporting the codegen project
    management patterns and providing a foundation for project-scoped resources.
    """
    __tablename__ = "codegen_projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(Integer, ForeignKey("codegen_organizations.id", ondelete="CASCADE"), nullable=False)
    
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Enum(ProjectStatus), nullable=False, default=ProjectStatus.ACTIVE)
    
    # Project configuration
    config = Column(JSON, default=dict)
    repository_url = Column(String(512), nullable=True)
    default_branch = Column(String(100), default="main")
    
    # Project ownership and collaboration
    owner_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    collaborators = Column(JSON, default=list)  # List of user IDs with access
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    archived_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    organization = relationship("CodegenOrganization", back_populates="projects")
    owner = relationship("User", foreign_keys=[owner_id])
    agent_instances = relationship("CodegenAgentInstance", back_populates="project", cascade="all, delete-orphan")
    sessions = relationship("CodegenSession", back_populates="project", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('ix_codegen_projects_org_id', 'organization_id'),
        Index('ix_codegen_projects_status', 'status'),
        Index('ix_codegen_projects_owner_id', 'owner_id'),
        Index('ix_codegen_projects_created_at', 'created_at'),
    )


class CodegenAPIKey(SerializableMixin, Base):
    """
    API key model for organization-scoped authentication.
    
    Manages API keys with organization context and fine-grained permissions,
    supporting the codegen authentication patterns.
    """
    __tablename__ = "codegen_api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(Integer, ForeignKey("codegen_organizations.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    key_hash = Column(String(255), nullable=False, unique=True)  # Hashed API key
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # API key configuration
    scopes = Column(JSON, default=list)  # List of allowed scopes/permissions
    rate_limit_override = Column(Integer, nullable=True)  # Override org rate limit
    
    # Key status and lifecycle
    is_active = Column(Boolean, nullable=False, default=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    organization = relationship("CodegenOrganization", back_populates="api_keys")
    user = relationship("User", foreign_keys=[user_id])
    
    # Indexes
    __table_args__ = (
        Index('ix_codegen_api_keys_key_hash', 'key_hash'),
        Index('ix_codegen_api_keys_org_id', 'organization_id'),
        Index('ix_codegen_api_keys_user_id', 'user_id'),
        Index('ix_codegen_api_keys_active', 'is_active'),
    )


class CodegenAgentInstance(SerializableMixin, Base):
    """
    Agent instance model for codegen agent execution tracking.
    
    Tracks individual agent instances within projects, supporting the
    Pro Mode tournament synthesis and parallel agent execution patterns.
    """
    __tablename__ = "codegen_agent_instances"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(Integer, ForeignKey("codegen_organizations.id", ondelete="CASCADE"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("codegen_projects.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(UUID(as_uuid=True), ForeignKey("codegen_sessions.id", ondelete="CASCADE"), nullable=True)
    
    # Agent configuration
    agent_name = Column(String(255), nullable=False)
    agent_version = Column(String(100), nullable=True)
    config = Column(JSON, default=dict)
    
    # Execution status
    status = Column(Enum(AgentInstanceStatus), nullable=False, default=AgentInstanceStatus.IDLE)
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    
    # Execution metadata
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Results and outputs
    inputs = Column(JSON, default=dict)
    outputs = Column(JSON, default=dict)
    execution_log = Column(JSON, default=list)  # List of log entries
    
    # Resource usage tracking
    cpu_time_seconds = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    organization = relationship("CodegenOrganization")
    project = relationship("CodegenProject", back_populates="agent_instances")
    session = relationship("CodegenSession", back_populates="agent_instances")
    
    # Indexes
    __table_args__ = (
        Index('ix_codegen_agent_instances_org_id', 'organization_id'),
        Index('ix_codegen_agent_instances_project_id', 'project_id'),
        Index('ix_codegen_agent_instances_session_id', 'session_id'),
        Index('ix_codegen_agent_instances_status', 'status'),
        Index('ix_codegen_agent_instances_created_at', 'created_at'),
    )


class CodegenSession(SerializableMixin, Base):
    """
    Session model for Pro Mode session management.
    
    Tracks Pro Mode sessions within projects, managing tournament synthesis
    and parallel agent execution workflows.
    """
    __tablename__ = "codegen_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(Integer, ForeignKey("codegen_organizations.id", ondelete="CASCADE"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("codegen_projects.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Session configuration
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    config = Column(JSON, default=dict)
    
    # Session execution
    status = Column(Enum(SessionStatus), nullable=False, default=SessionStatus.ACTIVE)
    total_agents = Column(Integer, default=0)
    completed_agents = Column(Integer, default=0)
    failed_agents = Column(Integer, default=0)
    
    # Tournament synthesis configuration
    synthesis_strategy = Column(String(100), default="tournament")
    max_parallel_agents = Column(Integer, default=3)
    timeout_minutes = Column(Integer, default=60)
    
    # Session timeline
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Results
    final_output = Column(JSON, default=dict)
    synthesis_results = Column(JSON, default=list)
    session_log = Column(JSON, default=list)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    organization = relationship("CodegenOrganization")
    project = relationship("CodegenProject", back_populates="sessions")
    user = relationship("User", foreign_keys=[user_id])
    agent_instances = relationship("CodegenAgentInstance", back_populates="session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('ix_codegen_sessions_org_id', 'organization_id'),
        Index('ix_codegen_sessions_project_id', 'project_id'),
        Index('ix_codegen_sessions_user_id', 'user_id'),
        Index('ix_codegen_sessions_status', 'status'),
        Index('ix_codegen_sessions_created_at', 'created_at'),
    )


class CodegenRateLimit(SerializableMixin, Base):
    """
    Rate limiting model for organization-scoped API usage tracking.
    
    Implements the codegen rate limiting patterns (60 requests per 30 seconds)
    with per-organization and per-user granularity.
    """
    __tablename__ = "codegen_rate_limits"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(Integer, ForeignKey("codegen_organizations.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("codegen_api_keys.id", ondelete="CASCADE"), nullable=True)
    
    # Rate limiting configuration
    endpoint_pattern = Column(String(255), nullable=False)  # e.g., "/v1/organizations/*/users"
    window_seconds = Column(Integer, nullable=False, default=30)  # 30-second window
    max_requests = Column(Integer, nullable=False, default=60)    # 60 requests max
    
    # Current window tracking
    window_start = Column(DateTime(timezone=True), nullable=False)
    current_count = Column(Integer, default=0)
    
    # Usage statistics
    total_requests = Column(Integer, default=0)
    total_blocked = Column(Integer, default=0)
    last_request_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    organization = relationship("CodegenOrganization")
    user = relationship("User", foreign_keys=[user_id])
    api_key = relationship("CodegenAPIKey", foreign_keys=[api_key_id])
    
    # Indexes for efficient rate limit checking
    __table_args__ = (
        Index('ix_codegen_rate_limits_org_endpoint', 'organization_id', 'endpoint_pattern'),
        Index('ix_codegen_rate_limits_user_endpoint', 'user_id', 'endpoint_pattern'),
        Index('ix_codegen_rate_limits_window_start', 'window_start'),
        Index('ix_codegen_rate_limits_api_key', 'api_key_id'),
    )


# Migration utility functions
def create_default_organization(user_id: str, name: str = "Default Organization") -> CodegenOrganization:
    """
    Create a default organization for existing users during migration.
    
    Args:
        user_id: The user ID to create the organization for
        name: The organization name
        
    Returns:
        CodegenOrganization: The created organization
    """
    org = CodegenOrganization(
        name=name,
        display_name=name,
        description=f"Default organization for user migration",
        status=OrganizationStatus.ACTIVE
    )
    
    return org


def create_organization_membership(org_id: int, user_id: str, role: OrganizationRole = OrganizationRole.OWNER) -> CodegenOrganizationMembership:
    """
    Create an organization membership during migration.
    
    Args:
        org_id: The organization ID
        user_id: The user ID
        role: The user's role in the organization
        
    Returns:
        CodegenOrganizationMembership: The created membership
    """
    membership = CodegenOrganizationMembership(
        organization_id=org_id,
        user_id=user_id,
        role=role,
        joined_at=datetime.now(timezone.utc)
    )
    
    return membership


# Validation functions for codegen compatibility
def validate_organization_id(org_id: int) -> bool:
    """Validate organization ID format for codegen compatibility."""
    return isinstance(org_id, int) and org_id > 0


def validate_user_permissions(user_id: str, org_id: int, required_permission: str) -> bool:
    """
    Validate user permissions within an organization.
    
    This would typically query the database to check if the user has
    the required permission within the specified organization.
    """
    # This is a placeholder - actual implementation would query the database
    return True


# Export all models for easy importing
__all__ = [
    'CodegenOrganization',
    'CodegenOrganizationMembership', 
    'CodegenProject',
    'CodegenAPIKey',
    'CodegenAgentInstance',
    'CodegenSession',
    'CodegenRateLimit',
    'OrganizationRole',
    'OrganizationStatus',
    'ProjectStatus',
    'AgentInstanceStatus',
    'SessionStatus',
    'create_default_organization',
    'create_organization_membership',
    'validate_organization_id',
    'validate_user_permissions'
]