"""
Codegen-aligned API routes implementing organization-centric patterns.

This module provides FastAPI routers that match the codegen project's API
structure, including /v1/organizations/{org_id}/* endpoints with proper
authentication, rate limiting, and validation.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Request, Query, Path, Body
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, asc, update, delete
from sqlalchemy.orm import selectinload, joinedload

from api.database import get_db_context
from api.codegen_models import (
    CodegenOrganization, CodegenOrganizationMembership, CodegenProject,
    CodegenAgentInstance, CodegenSession, CodegenAPIKey, OrganizationRole,
    OrganizationStatus, ProjectStatus, SessionStatus, AgentInstanceStatus
)
from api.codegen_schemas import (
    UserResponse, OrganizationResponse, OrganizationMembershipResponse,
    ProjectResponse, AgentInstanceResponse, SessionResponse,
    Page_UserResponse_, Page_OrganizationResponse_, Page_ProjectResponse_,
    Page_AgentInstanceResponse_, Page_SessionResponse_,
    CreateOrganizationRequest, UpdateOrganizationRequest,
    CreateProjectRequest, UpdateProjectRequest,
    CreateAgentInstanceRequest, CreateSessionRequest,
    AddOrganizationMemberRequest, UpdateOrganizationMemberRequest,
    ProModeRequest, ProModeResponse, HealthResponse, StatsResponse,
    PaginationMeta, validate_pagination_params
)
from api.codegen_middleware import (
    get_organization_context, require_permission, require_role, OrganizationContext
)
from api.models import User

logger = logging.getLogger(__name__)

# Main API router for codegen endpoints
codegen_router = APIRouter(prefix="/v1", tags=["Codegen API"])

# Sub-routers for different resource types
organizations_router = APIRouter(prefix="/organizations", tags=["Organizations"])
users_router = APIRouter(tags=["Users"])
projects_router = APIRouter(tags=["Projects"])
sessions_router = APIRouter(tags=["Sessions"])
agents_router = APIRouter(tags=["Agents"])


# Health and system endpoints
@codegen_router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for service monitoring.
    
    Returns service status, database connectivity, and dependency health.
    """
    try:
        # Test database connectivity
        async with get_db_context() as db:
            await db.execute(select(1))
            database_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        database_status = "disconnected"
    
    return HealthResponse(
        status="healthy" if database_status == "connected" else "degraded",
        database_status=database_status,
        dependencies={
            "database": database_status,
            "codegen_adapter": "available"  # Assuming adapter is available
        }
    )


@codegen_router.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_system_stats():
    """
    Get system statistics for monitoring and analytics.
    
    Returns counts of organizations, users, projects, and active sessions.
    """
    async with get_db_context() as db:
        # Get organization count
        org_count_result = await db.execute(
            select(func.count(CodegenOrganization.id))
            .where(CodegenOrganization.status == OrganizationStatus.ACTIVE)
        )
        total_organizations = org_count_result.scalar() or 0
        
        # Get user count (from original users table)
        user_count_result = await db.execute(select(func.count(User.id)))
        total_users = user_count_result.scalar() or 0
        
        # Get project count
        project_count_result = await db.execute(
            select(func.count(CodegenProject.id))
            .where(CodegenProject.status == ProjectStatus.ACTIVE)
        )
        total_projects = project_count_result.scalar() or 0
        
        # Get active sessions count
        active_sessions_result = await db.execute(
            select(func.count(CodegenSession.id))
            .where(CodegenSession.status == SessionStatus.ACTIVE)
        )
        active_sessions = active_sessions_result.scalar() or 0
        
        # Get total agent instances
        agent_instances_result = await db.execute(
            select(func.count(CodegenAgentInstance.id))
        )
        total_agent_instances = agent_instances_result.scalar() or 0
        
        # API requests last hour would need separate tracking - placeholder for now
        api_requests_last_hour = 0
    
    return StatsResponse(
        total_organizations=total_organizations,
        total_users=total_users,
        total_projects=total_projects,
        active_sessions=active_sessions,
        total_agent_instances=total_agent_instances,
        api_requests_last_hour=api_requests_last_hour
    )


# Organization endpoints
@organizations_router.get("/{org_id}/users", response_model=Page_UserResponse_)
async def get_users(
    org_id: int = Path(..., description="Organization ID"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum items to return"),
    org_context: OrganizationContext = Depends(require_permission("read"))
):
    """
    Get users for the specified organization.
    
    Returns a paginated list of all users that belong to the specified organization.
    Results include user details such as name, email, GitHub username, and avatar.
    Use pagination parameters to control the number of results returned.
    
    Rate limit: 60 requests per 30 seconds.
    """
    skip, limit = validate_pagination_params(skip, limit)
    
    async with get_db_context() as db:
        # Get total count
        count_result = await db.execute(
            select(func.count(CodegenOrganizationMembership.id))
            .where(CodegenOrganizationMembership.organization_id == org_id)
        )
        total = count_result.scalar() or 0
        
        # Get paginated memberships with user details
        result = await db.execute(
            select(CodegenOrganizationMembership)
            .options(selectinload(CodegenOrganizationMembership.user))
            .where(CodegenOrganizationMembership.organization_id == org_id)
            .offset(skip)
            .limit(limit)
            .order_by(CodegenOrganizationMembership.joined_at.desc())
        )
        
        memberships = result.scalars().all()
        
        # Convert to UserResponse format
        users = []
        for membership in memberships:
            if membership.user:
                users.append(UserResponse(
                    id=hash(membership.user.id) % (2**31),  # Convert string ID to int for codegen compatibility
                    name=membership.user.name,
                    username=membership.user.username,
                    email=None,  # Email not stored in current schema
                    avatar_url=None,  # Avatar not stored in current schema
                    github_username=None,  # GitHub username not stored in current schema
                    created_at=membership.user.created_at,
                    updated_at=membership.user.updated_at
                ))
        
        return Page_UserResponse_(
            items=users,
            meta=PaginationMeta(
                total=total,
                skip=skip,
                limit=limit,
                has_more=skip + limit < total
            )
        )


@organizations_router.get("/{org_id}/users/{user_id}", response_model=UserResponse)
async def get_user(
    org_id: int = Path(..., description="Organization ID"),
    user_id: int = Path(..., description="User ID"),
    org_context: OrganizationContext = Depends(require_permission("read"))
):
    """
    Get details for a specific user in an organization.
    
    Returns detailed information about a user within the specified organization.
    The requesting user must be a member of the organization to access this endpoint.
    
    Rate limit: 60 requests per 30 seconds.
    """
    async with get_db_context() as db:
        # Convert int user_id back to string for lookup (codegen compatibility hack)
        # In a real implementation, you'd need proper user ID mapping
        result = await db.execute(
            select(CodegenOrganizationMembership)
            .options(selectinload(CodegenOrganizationMembership.user))
            .where(
                and_(
                    CodegenOrganizationMembership.organization_id == org_id,
                    # This is a simplified lookup - in practice you'd need user ID mapping
                    CodegenOrganizationMembership.user_id.like(f"%{str(user_id)[-6:]}")
                )
            )
        )
        
        membership = result.scalar_one_or_none()
        if not membership or not membership.user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found in organization"
            )
        
        return UserResponse(
            id=user_id,
            name=membership.user.name,
            username=membership.user.username,
            email=None,
            avatar_url=None,
            github_username=None,
            created_at=membership.user.created_at,
            updated_at=membership.user.updated_at
        )


@organizations_router.get("/{org_id}/projects", response_model=Page_ProjectResponse_)
async def get_projects(
    org_id: int = Path(..., description="Organization ID"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum items to return"),
    status: Optional[ProjectStatus] = Query(None, description="Filter by project status"),
    org_context: OrganizationContext = Depends(require_permission("read"))
):
    """
    Get projects for the specified organization.
    
    Returns a paginated list of projects within the organization.
    Supports filtering by project status and pagination.
    """
    skip, limit = validate_pagination_params(skip, limit)
    
    async with get_db_context() as db:
        # Build query conditions
        conditions = [CodegenProject.organization_id == org_id]
        if status:
            conditions.append(CodegenProject.status == status)
        
        # Get total count
        count_result = await db.execute(
            select(func.count(CodegenProject.id))
            .where(and_(*conditions))
        )
        total = count_result.scalar() or 0
        
        # Get paginated projects
        result = await db.execute(
            select(CodegenProject)
            .options(selectinload(CodegenProject.owner))
            .where(and_(*conditions))
            .offset(skip)
            .limit(limit)
            .order_by(CodegenProject.created_at.desc())
        )
        
        projects = result.scalars().all()
        
        # Convert to response format
        project_responses = []
        for project in projects:
            owner_response = None
            if project.owner:
                owner_response = UserResponse(
                    id=hash(project.owner.id) % (2**31),
                    name=project.owner.name,
                    username=project.owner.username,
                    email=None,
                    avatar_url=None,
                    github_username=None,
                    created_at=project.owner.created_at,
                    updated_at=project.owner.updated_at
                )
            
            project_responses.append(ProjectResponse(
                id=project.id,
                organization_id=project.organization_id,
                name=project.name,
                description=project.description,
                status=project.status,
                config=project.config,
                repository_url=project.repository_url,
                default_branch=project.default_branch,
                owner_id=project.owner_id,
                collaborators=project.collaborators,
                created_at=project.created_at,
                updated_at=project.updated_at,
                archived_at=project.archived_at,
                owner=owner_response
            ))
        
        return Page_ProjectResponse_(
            items=project_responses,
            meta=PaginationMeta(
                total=total,
                skip=skip,
                limit=limit,
                has_more=skip + limit < total
            )
        )


@organizations_router.post("/{org_id}/projects", response_model=ProjectResponse)
async def create_project(
    org_id: int = Path(..., description="Organization ID"),
    request: CreateProjectRequest = Body(...),
    org_context: OrganizationContext = Depends(require_permission("write"))
):
    """
    Create a new project in the organization.
    
    Creates a new project with the specified configuration.
    The requesting user becomes the project owner.
    """
    async with get_db_context() as db:
        # Create new project
        project = CodegenProject(
            organization_id=org_id,
            name=request.name,
            description=request.description,
            config=request.config,
            repository_url=request.repository_url,
            default_branch=request.default_branch,
            owner_id=org_context.user_id,
            collaborators=request.collaborators,
            status=ProjectStatus.ACTIVE
        )
        
        db.add(project)
        await db.commit()
        await db.refresh(project)
        
        # Load owner details
        await db.execute(
            select(CodegenProject)
            .options(selectinload(CodegenProject.owner))
            .where(CodegenProject.id == project.id)
        )
        
        owner_response = None
        if project.owner:
            owner_response = UserResponse(
                id=hash(project.owner.id) % (2**31),
                name=project.owner.name,
                username=project.owner.username,
                email=None,
                avatar_url=None,
                github_username=None,
                created_at=project.owner.created_at,
                updated_at=project.owner.updated_at
            )
        
        return ProjectResponse(
            id=project.id,
            organization_id=project.organization_id,
            name=project.name,
            description=project.description,
            status=project.status,
            config=project.config,
            repository_url=project.repository_url,
            default_branch=project.default_branch,
            owner_id=project.owner_id,
            collaborators=project.collaborators,
            created_at=project.created_at,
            updated_at=project.updated_at,
            archived_at=project.archived_at,
            owner=owner_response
        )


@organizations_router.get("/{org_id}/projects/{project_id}/sessions", response_model=Page_SessionResponse_)
async def get_project_sessions(
    org_id: int = Path(..., description="Organization ID"),
    project_id: uuid.UUID = Path(..., description="Project ID"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum items to return"),
    status: Optional[SessionStatus] = Query(None, description="Filter by session status"),
    org_context: OrganizationContext = Depends(require_permission("read"))
):
    """
    Get Pro Mode sessions for a project.
    
    Returns paginated list of sessions within the specified project.
    Supports filtering by session status.
    """
    skip, limit = validate_pagination_params(skip, limit)
    
    async with get_db_context() as db:
        # Verify project belongs to organization
        project_result = await db.execute(
            select(CodegenProject)
            .where(
                and_(
                    CodegenProject.id == project_id,
                    CodegenProject.organization_id == org_id
                )
            )
        )
        
        project = project_result.scalar_one_or_none()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Build query conditions
        conditions = [
            CodegenSession.organization_id == org_id,
            CodegenSession.project_id == project_id
        ]
        if status:
            conditions.append(CodegenSession.status == status)
        
        # Get total count
        count_result = await db.execute(
            select(func.count(CodegenSession.id))
            .where(and_(*conditions))
        )
        total = count_result.scalar() or 0
        
        # Get paginated sessions
        result = await db.execute(
            select(CodegenSession)
            .options(
                selectinload(CodegenSession.user),
                selectinload(CodegenSession.project),
                selectinload(CodegenSession.agent_instances)
            )
            .where(and_(*conditions))
            .offset(skip)
            .limit(limit)
            .order_by(CodegenSession.created_at.desc())
        )
        
        sessions = result.scalars().all()
        
        # Convert to response format
        session_responses = []
        for session in sessions:
            # Convert user to UserResponse
            user_response = None
            if session.user:
                user_response = UserResponse(
                    id=hash(session.user.id) % (2**31),
                    name=session.user.name,
                    username=session.user.username,
                    email=None,
                    avatar_url=None,
                    github_username=None,
                    created_at=session.user.created_at,
                    updated_at=session.user.updated_at
                )
            
            # Convert agent instances
            agent_responses = []
            for agent in session.agent_instances:
                agent_responses.append(AgentInstanceResponse(
                    id=agent.id,
                    organization_id=agent.organization_id,
                    project_id=agent.project_id,
                    session_id=agent.session_id,
                    agent_name=agent.agent_name,
                    agent_version=agent.agent_version,
                    config=agent.config,
                    status=agent.status,
                    progress=agent.progress,
                    started_at=agent.started_at,
                    completed_at=agent.completed_at,
                    error_message=agent.error_message,
                    inputs=agent.inputs,
                    outputs=agent.outputs,
                    execution_log=agent.execution_log,
                    cpu_time_seconds=agent.cpu_time_seconds,
                    memory_usage_mb=agent.memory_usage_mb,
                    created_at=agent.created_at,
                    updated_at=agent.updated_at
                ))
            
            session_responses.append(SessionResponse(
                id=session.id,
                organization_id=session.organization_id,
                project_id=session.project_id,
                user_id=session.user_id,
                name=session.name,
                description=session.description,
                config=session.config,
                status=session.status,
                total_agents=session.total_agents,
                completed_agents=session.completed_agents,
                failed_agents=session.failed_agents,
                synthesis_strategy=session.synthesis_strategy,
                max_parallel_agents=session.max_parallel_agents,
                timeout_minutes=session.timeout_minutes,
                started_at=session.started_at,
                completed_at=session.completed_at,
                final_output=session.final_output,
                synthesis_results=session.synthesis_results,
                session_log=session.session_log,
                created_at=session.created_at,
                updated_at=session.updated_at,
                user=user_response,
                agent_instances=agent_responses
            ))
        
        return Page_SessionResponse_(
            items=session_responses,
            meta=PaginationMeta(
                total=total,
                skip=skip,
                limit=limit,
                has_more=skip + limit < total
            )
        )


@organizations_router.post("/{org_id}/projects/{project_id}/pro-mode", response_model=ProModeResponse)
async def create_pro_mode_session(
    org_id: int = Path(..., description="Organization ID"),
    project_id: uuid.UUID = Path(..., description="Project ID"),
    request: ProModeRequest = Body(...),
    org_context: OrganizationContext = Depends(require_permission("write"))
):
    """
    Create a new Pro Mode tournament synthesis session.
    
    Initiates a Pro Mode session with the specified configuration,
    creating agent instances and beginning tournament synthesis.
    """
    async with get_db_context() as db:
        # Verify project belongs to organization
        project_result = await db.execute(
            select(CodegenProject)
            .where(
                and_(
                    CodegenProject.id == project_id,
                    CodegenProject.organization_id == org_id
                )
            )
        )
        
        project = project_result.scalar_one_or_none()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )
        
        # Create new session
        session = CodegenSession(
            organization_id=org_id,
            project_id=project_id,
            user_id=org_context.user_id,
            name=f"Pro Mode Session - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
            description=f"Tournament synthesis for: {request.prompt[:100]}...",
            config={
                "prompt": request.prompt,
                "agents": request.agents,
                "max_iterations": request.max_iterations,
                **request.config
            },
            status=SessionStatus.ACTIVE,
            synthesis_strategy=request.synthesis_strategy,
            max_parallel_agents=len(request.agents),
            timeout_minutes=request.timeout_minutes,
            started_at=datetime.now(timezone.utc)
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        # Create agent instances for each requested agent type
        agents_created = 0
        for agent_name in request.agents:
            agent = CodegenAgentInstance(
                organization_id=org_id,
                project_id=project_id,
                session_id=session.id,
                agent_name=agent_name,
                config=request.config,
                inputs={"prompt": request.prompt},
                status=AgentInstanceStatus.IDLE
            )
            
            db.add(agent)
            agents_created += 1
        
        # Update session with agent count
        session.total_agents = agents_created
        await db.commit()
        
        # Calculate estimated completion time
        estimated_completion = datetime.now(timezone.utc) + timedelta(minutes=request.timeout_minutes)
        
        return ProModeResponse(
            session_id=session.id,
            status="created",
            message=f"Pro Mode session created with {agents_created} agents",
            agents_created=agents_created,
            estimated_completion=estimated_completion
        )


# Include all sub-routers in the main organizations router
organizations_router.include_router(users_router)
organizations_router.include_router(projects_router)
organizations_router.include_router(sessions_router)
organizations_router.include_router(agents_router)

# Include organizations router in main codegen router
codegen_router.include_router(organizations_router)

# Export the main router
__all__ = ['codegen_router']