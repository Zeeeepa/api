import asyncio
from decimal import Decimal
import json
import os
from uuid import UUID, uuid4
import uuid
from api.routes.deployments import DeploymentCreate, create_deployment
from api.utils.inputs import get_inputs_from_workflow_api
from api.utils.outputs import get_outputs_from_workflow
from sqlalchemy.orm import defer
from pydantic import BaseModel

from .workflows import CustomJSONEncoder
from .utils import (
    UserIconData,
    ensure_run_timeout,
    fetch_user_icon,
    get_user_settings,
    post_process_output_data,
)
from .types import (
    DeploymentModel,
    WorkflowModel,
    WorkflowRunModel,
    WorkflowVersionModel,
)
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from .utils import post_process_outputs, select
from sqlalchemy import func
from datetime import datetime, timedelta, timezone
from fastapi.responses import JSONResponse
from pprint import pprint
from sqlalchemy import desc
from sqlalchemy import text
from api.models import (
    Deployment,
    Machine,
    MachineVersion,
    Workflow,
    WorkflowRun,
    WorkflowRunWithExtra,
    WorkflowVersion,
    WorkflowRunOutput,
    SerializableMixin,
)
from api.database import get_db
import logging
from typing import Any, Dict, List, Optional
from .share import get_dub_link
# from fastapi_pagination import Page, add_pagination, paginate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Workflow"])


class WorkflowUpdateModel(BaseModel):
    name: Optional[str] = None
    selected_machine_id: Optional[UUID] = None
    pinned: Optional[bool] = None
    deleted: Optional[bool] = None
    description: Optional[str] = None
    cover_image: Optional[str] = None

    class Config:
        from_attributes = True


@router.patch("/workflow/{workflow_id}", response_model=WorkflowModel)
async def update_workflow(
    request: Request,
    workflow_id: str,
    body: WorkflowUpdateModel,
    db: AsyncSession = Depends(get_db),
):
    # Get the workflow
    query = (
        select(Workflow)
        .where(Workflow.id == workflow_id)
        .where(Workflow.deleted == False)
        .apply_org_check(request)
    )

    result = await db.execute(query)
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Update workflow fields if provided in request body
    if body.pinned is not None:
        workflow.pinned = body.pinned
    else:
        # Only update timestamp if we're changing something other than pinned
        workflow.updated_at = datetime.now(timezone.utc)
        if body.name is not None:
            workflow.name = body.name
        if body.selected_machine_id is not None:
            workflow.selected_machine_id = body.selected_machine_id
        if body.deleted is not None:
            workflow.deleted = body.deleted
        if body.description is not None:
            workflow.description = body.description
        if body.cover_image is not None:
            workflow.cover_image = body.cover_image

    await db.commit()

    return workflow


@router.delete("/workflow/{workflow_id}")
async def delete_workflow(
    request: Request,
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
):
    # Get the workflow
    query = (
        select(Workflow)
        .where(Workflow.id == workflow_id)
        .where(Workflow.deleted == False)
        .apply_org_check(request)
    )

    result = await db.execute(query)
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Soft delete by setting deleted flag
    workflow.deleted = True
    workflow.updated_at = datetime.now(timezone.utc)

    await db.commit()

    return {"message": "Workflow deleted successfully"}


@router.post("/workflow/{workflow_id}/clone")
async def clone_workflow(
    request: Request,
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
):
    # Get the original workflow
    query = (
        select(Workflow)
        .where(Workflow.id == workflow_id)
        .where(Workflow.deleted == False)
        .apply_org_check(request)
    )

    result = await db.execute(query)
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Get the latest workflow version
    version_query = (
        select(WorkflowVersion)
        .where(WorkflowVersion.workflow_id == workflow_id)
        .order_by(WorkflowVersion.created_at.desc())
        .limit(1)
    )

    version_result = await db.execute(version_query)
    workflow_version = version_result.scalar_one_or_none()

    # Create new workflow as a clone
    new_workflow = Workflow(
        id=uuid4(),
        name=f"{workflow.name} (Cloned)",
        org_id=workflow.org_id,
        user_id=workflow.user_id,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    if workflow.selected_machine_id:
        new_workflow.selected_machine_id = workflow.selected_machine_id
    db.add(new_workflow)
    await db.flush()

    # Create new version for cloned workflow if original had a version
    if workflow_version:
        new_version = WorkflowVersion(
            id=uuid4(),
            workflow_id=new_workflow.id,
            workflow=workflow_version.workflow,
            workflow_api=workflow_version.workflow_api,
            snapshot=workflow_version.snapshot,
            dependencies=workflow_version.dependencies,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            version=1,
            comment="initial version",
            user_id=workflow.user_id,
        )
        db.add(new_version)

    await db.commit()

    return new_workflow


def serialize_row(row_data):
    """Serialize a row of data with proper type conversion"""
    if isinstance(row_data, uuid.UUID):
        return str(row_data)
    if isinstance(row_data, datetime):
        return row_data.isoformat()[:-3] + "Z"
    if isinstance(row_data, list):
        return [serialize_row(item) for item in row_data]
    if isinstance(row_data, dict):
        return {k: serialize_row(v) for k, v in row_data.items()}
    if isinstance(row_data, Decimal):
        return str(row_data)
    if hasattr(row_data, "to_dict") and callable(row_data.to_dict):
        return row_data.to_dict()
    return row_data


@router.get("/v2/workflow/{workflow_id}/runs", response_model=List[WorkflowRunModel])
async def get_all_runs(
    request: Request,
    workflow_id: str,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    query = text("""
    SELECT 
        r.id, r.status, r.created_at, r.updated_at, r.ended_at, r.queued_at, r.started_at, 
        r.workflow_id, r.user_id, r.org_id, r.origin, r.gpu, r.machine_version, r.machine_type, 
        r.modal_function_call_id, r.webhook_status, r.webhook_intermediate_status, r.batch_id, r.workflow_version_id,
        -- Computing timing metrics
        EXTRACT(EPOCH FROM (r.ended_at - r.created_at)) as duration,
        EXTRACT(EPOCH FROM (r.queued_at - r.created_at)) as comfy_deploy_cold_start,
        EXTRACT(EPOCH FROM (r.started_at - r.queued_at)) as cold_start_duration,
        EXTRACT(EPOCH FROM (r.started_at - r.created_at)) as cold_start_duration_total,
        EXTRACT(EPOCH FROM (r.ended_at - r.started_at)) as run_duration
    FROM comfyui_deploy.workflow_runs r
    INNER JOIN comfyui_deploy.workflows w ON w.id = r.workflow_id
    WHERE r.workflow_id = :workflow_id
    AND w.deleted = false
    AND (
        (CAST(:org_id AS TEXT) IS NOT NULL AND r.org_id = CAST(:org_id AS TEXT))
        OR (CAST(:org_id AS TEXT) IS NULL AND r.org_id IS NULL AND r.user_id = CAST(:user_id AS TEXT))
    )
    ORDER BY r.created_at DESC
    LIMIT :limit
    OFFSET :offset
    """)

    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if "org_id" in current_user else None

    result = await db.execute(
        query,
        {
            "user_id": user_id,
            "org_id": org_id,
            "limit": limit,
            "workflow_id": workflow_id,
            "offset": offset,
        },
    )

    # Convert raw SQL results using our standalone serialization function
    runs = [serialize_row(dict(row._mapping)) for row in result.fetchall()]

    if not runs:
        return []
    for run in runs:
        ensure_run_timeout(run)

    return JSONResponse(content=runs)


@router.get("/workflow/{workflow_id}/runs", response_model=List[WorkflowRunModel])
async def get_all_runs(
    request: Request,
    workflow_id: str,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    with_outputs: bool = True,
    with_inputs: bool = True,
):
    user_settings = await get_user_settings(request, db)

    # Create base query options
    query_options = [
        joinedload(WorkflowRun.workflow),
        joinedload(WorkflowRun.version),
    ]

    # Conditionally add outputs loading
    if with_outputs:
        query_options.append(joinedload(WorkflowRun.outputs))

    query = (
        select(WorkflowRunWithExtra)
        .options(*query_options)
        .outerjoin(
            WorkflowVersion, WorkflowRun.workflow_version_id == WorkflowVersion.id
        )
        .join(Workflow, WorkflowRun.workflow_id == Workflow.id)
        .where(WorkflowRun.workflow_id == workflow_id)
        .where(Workflow.deleted == False)
        .apply_org_check(request)
        .order_by(WorkflowRun.created_at.desc())
        .paginate(limit, offset)
    )

    # Instead of with_only_columns, use column_descriptions to deferred load specific columns
    if not with_inputs:
        query = query.options(defer(WorkflowRunWithExtra.workflow_inputs))

    # Always defer run_log
    query = query.options(defer(WorkflowRunWithExtra.run_log))

    result = await db.execute(query)
    runs = result.unique().scalars().all()

    if not runs:
        return []
    for run in runs:
        ensure_run_timeout(run)

    # Loop through each run and check its outputs
    if with_outputs:  # Only process outputs if they were loaded
        for run in runs:
            if run.outputs:
                post_process_outputs(run.outputs, user_settings)

    runs_data = []
    for run in runs:
        run_dict = run.to_dict()
        if run.version:
            run_dict["version"] = run.version.to_dict()
        else:
            run_dict["version"] = None  # Explicitly set to None if no version
        runs_data.append(run_dict)

    return JSONResponse(content=runs_data)


@router.get("/workflow/{workflow_id}/run/latest", response_model=List[WorkflowRunModel])
async def get_latest_run(
    request: Request,
    workflow_id: str,
    limit: int = 1,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    user_settings = await get_user_settings(request, db)

    # First get the latest run ID
    latest_run_query = (
        select(WorkflowRun)
        .join(Workflow, WorkflowRun.workflow_id == Workflow.id)
        .where(WorkflowRun.workflow_id == workflow_id)
        .where(Workflow.deleted == False)
        .apply_org_check(request)
        .order_by(WorkflowRun.created_at.desc())
        .limit(1)
    )

    result = await db.execute(latest_run_query)
    latest_run = result.scalar_one_or_none()

    if not latest_run:
        return []

    # Then fetch only the outputs for this run
    outputs_query = (
        select(WorkflowRunOutput)
        .where(WorkflowRunOutput.run_id == latest_run.id)
        .where(text("data ?| array['images', 'gifs', 'mesh']"))
        .order_by(desc("created_at"))
        .limit(1)
    )

    outputs_result = await db.execute(outputs_query)
    outputs = outputs_result.scalars().all()

    # Process the outputs
    if outputs:
        post_process_outputs(outputs, user_settings)

    result = latest_run.to_dict()
    result["outputs"] = [output.to_dict() for output in outputs]

    return JSONResponse(content=[result])


@router.get(
    "/workflow/{workflow_id}/versions", response_model=List[WorkflowVersionModel]
)
async def get_versions(
    request: Request,
    workflow_id: str,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    # Check if the user can access this workflow
    workflow = await db.execute(
        select(Workflow).where(Workflow.id == workflow_id).apply_org_check(request)
    )
    workflow = workflow.scalar_one_or_none()

    if not workflow:
        raise HTTPException(
            status_code=404, detail="Workflow not found or you don't have access to it"
        )

    query = (
        select(WorkflowVersion)
        .join(Workflow, WorkflowVersion.workflow_id == Workflow.id)
        .where(WorkflowVersion.workflow_id == workflow_id)
        .where(Workflow.deleted == False)
        .order_by(WorkflowVersion.created_at.desc())
        .paginate(limit, offset)
    )

    if search:
        query = query.where(
            func.lower(WorkflowVersion.comment).contains(search.lower())
        )

    result = await db.execute(query)
    runs = result.unique().scalars().all()

    unique_user_ids = list(set(run.user_id for run in runs if run.user_id))

    # Fetch user icons using the cached function
    results = await asyncio.gather(
        *[fetch_user_icon(user_id) for user_id in unique_user_ids]
    )

    # Process results - create dictionary by pairing user IDs with their icon data
    user_icons = {
        str(user_id): icon_data for user_id, icon_data in zip(unique_user_ids, results)
    }

    if not runs:
        return []

    runs_data = []
    for run in runs:
        run_dict = run.to_dict()
        if run.user_id:
            user_icon = user_icons.get(str(run.user_id))
            run_dict["user_icon"] = user_icon.image_url if user_icon else None
        else:
            run_dict["user_icon"] = None
        runs_data.append(run_dict)

    return JSONResponse(content=runs_data)


@router.get("/workflow/{workflow_id}", response_model=WorkflowModel)
async def get_workflow(
    request: Request,
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    limit: int = 1,  # Default to 5 most recent versions
):
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user.get("org_id")

    query = text("""
    WITH recent_versions AS (
        SELECT *
        FROM comfyui_deploy.workflow_versions
        WHERE workflow_id = :workflow_id
        ORDER BY created_at DESC
        LIMIT :limit
    )
    SELECT w.*, 
           json_agg(rv.* ORDER BY rv.created_at DESC) AS versions
    FROM comfyui_deploy.workflows w
    LEFT JOIN recent_versions rv ON w.id = rv.workflow_id
    WHERE w.id = :workflow_id
    AND w.deleted = false
    AND (
        (CAST(:org_id AS TEXT) IS NOT NULL AND w.org_id = CAST(:org_id AS TEXT))
        OR (CAST(:org_id AS TEXT) IS NULL AND w.org_id IS NULL AND w.user_id = CAST(:user_id AS TEXT))
    )
    GROUP BY w.id
    """)

    result = await db.execute(
        query,
        {
            "workflow_id": workflow_id,
            "limit": limit,
            "org_id": org_id,
            "user_id": user_id,
        },
    )

    workflow = result.fetchone()

    if not workflow:
        raise HTTPException(
            status_code=404, detail="Workflow not found or you don't have access to it"
        )

    # Convert the result to a dict
    workflow_dict = dict(workflow._mapping)
    # Parse the JSON string of versions back into a list of dicts
    # workflow_dict['versions'] = json.loads(workflow_dict['versions'])

    return JSONResponse(
        status_code=200,
        content=json.loads(json.dumps(workflow_dict, cls=CustomJSONEncoder)),
    )


@router.get("/workflow/{workflow_id}/version/{version}", response_model=WorkflowModel)
async def get_workflow_version(
    request: Request,
    workflow_id: str,
    version: int,
    db: AsyncSession = Depends(get_db),
):
    workflow_version = await db.execute(
        select(WorkflowVersion)
        .join(Workflow, Workflow.id == WorkflowVersion.workflow_id)
        .where(WorkflowVersion.workflow_id == workflow_id)
        .where(WorkflowVersion.version == version)
        .where(Workflow.deleted == False)
        .apply_org_check_by_type(Workflow, request)
    )
    workflow_version = workflow_version.scalar_one_or_none()

    if not workflow_version:
        raise HTTPException(
            status_code=404,
            detail="Workflow version not found or you don't have access to it",
        )

    return JSONResponse(content=workflow_version.to_dict())


@router.get("/workflow-version/{version}", response_model=WorkflowModel)
async def get_workflow_version_by_id(
    request: Request,
    version: UUID,
    db: AsyncSession = Depends(get_db),
):
    # Build base query
    query = (
        select(WorkflowVersion)
        .join(Workflow, Workflow.id == WorkflowVersion.workflow_id)
        .where(Workflow.deleted == False)
        .where(WorkflowVersion.id == version)
        .apply_org_check_by_type(Workflow, request)
    )

    workflow_version = await db.execute(query)
    workflow_version = workflow_version.scalar_one_or_none()

    if not workflow_version:
        raise HTTPException(
            status_code=404,
            detail="Workflow version not found or you don't have access to it",
        )

    return JSONResponse(content=workflow_version.to_dict())


@router.get("/workflow/{workflow_id}/gallery")
async def get_workflows_gallery(
    request: Request,
    workflow_id: str,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user.get("org_id")

    raw_query = text("""
    WITH filtered_runs AS (
        SELECT 
            run.id, 
            run.started_at, 
            run.ended_at, 
            run.queued_at, 
            run.created_at,
            run.org_id
        FROM 
            comfyui_deploy.workflow_runs AS run
        JOIN 
            comfyui_deploy.workflows AS workflow ON run.workflow_id = workflow.id
        WHERE 
            run.workflow_id = :workflow_id
            AND run.status = 'success'
            AND workflow.deleted = false
            AND (
                (CAST(:org_id AS TEXT) IS NOT NULL AND run.org_id = CAST(:org_id AS TEXT))
                OR (CAST(:org_id AS TEXT) IS NULL AND run.org_id IS NULL AND run.user_id = CAST(:user_id AS TEXT))
            )
    )
    SELECT
        output.id as output_id,
        filtered_runs.id as run_id,
        output.data,
        output.node_meta,
        (EXTRACT(EPOCH FROM filtered_runs.ended_at) - EXTRACT(EPOCH FROM filtered_runs.started_at)) AS run_duration,
        (EXTRACT(EPOCH FROM filtered_runs.started_at) - EXTRACT(EPOCH FROM filtered_runs.queued_at)) AS cold_start,
        (EXTRACT(EPOCH FROM filtered_runs.queued_at) - EXTRACT(EPOCH FROM filtered_runs.created_at)) AS queue_time
    FROM 
        comfyui_deploy.workflow_run_outputs AS output
    JOIN 
        filtered_runs ON output.run_id = filtered_runs.id
    WHERE
        output.data ?| ARRAY['images', 'gifs', 'mesh']
    ORDER BY 
        output.created_at DESC
    LIMIT :limit
    OFFSET :offset
    """)

    result = await db.execute(
        raw_query,
        {
            "workflow_id": workflow_id,
            "limit": limit,
            "offset": offset,
            "org_id": org_id,
            "user_id": user_id,
        },
    )

    outputs = [dict(row._mapping) for row in result.fetchall()]

    user_settings = await get_user_settings(request, db)
    for output in outputs:
        if output["data"]:
            post_process_output_data(output["data"], user_settings)

    return JSONResponse(
        status_code=200, content=json.loads(json.dumps(outputs, cls=CustomJSONEncoder))
    )


@router.get("/workflow/{workflow_id}/deployments")
async def get_deployments(
    request: Request,
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[DeploymentModel]:
    deployments = await db.execute(
        select(Deployment)
        .options(
            joinedload(Deployment.machine).load_only(Machine.name, Machine.id),
            joinedload(Deployment.workflow).load_only(Workflow.name),
            joinedload(Deployment.version),
        )
        .join(Workflow, Workflow.id == Deployment.workflow_id)
        .where(Deployment.workflow_id == workflow_id)
        .where(Workflow.deleted == False)
        .apply_org_check(request)
        .order_by(Deployment.environment.desc())
    )
    deployments = deployments.scalars().all()

    if not deployments:
        return JSONResponse(
            status_code=404,
            content={"detail": "Deployments not found or you don't have access to it"},
        )

    deployments_data = []
    for deployment in deployments:
        deployment_dict = deployment.to_dict()
        workflow_api = deployment.version.workflow_api if deployment.version else None
        inputs = get_inputs_from_workflow_api(workflow_api)

        workflow = deployment.version.workflow if deployment.version else None
        outputs = get_outputs_from_workflow(workflow)

        if inputs:
            deployment_dict["input_types"] = inputs

        if outputs:
            deployment_dict["output_types"] = outputs
            
        if deployment.environment == "public-share" and deployment.share_slug:
            dub_link = await get_dub_link(deployment.share_slug)
            if dub_link:
                deployment_dict["dub_link"] = dub_link.short_link

        deployments_data.append(deployment_dict)

    return deployments_data


class WorkflowCreateRequest(BaseModel):
    name: str
    workflow_json: str
    workflow_api: Optional[str] = None
    machine_id: Optional[str] = None
    machine_version_id: Optional[str] = None
    comfyui_snapshot: Optional[Dict[str, Any]] = None


@router.post("/workflow", response_model=WorkflowModel)
async def create_workflow(
    request: Request,
    body: WorkflowCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    # Check for authorized user
    try:
        user_id = request.state.current_user["user_id"]
    except (AttributeError, KeyError):
        return JSONResponse(status_code=401, content={"error": "Unauthorized access. "})

    # Validate JSON first
    try:
        json.loads(body.workflow_json)
        if body.workflow_api:
            json.loads(body.workflow_api)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON format"})

    # Check if machine exists and get machine version info if provided
    machine_id = None
    machine_version_id = None

    if body.machine_version_id:
        machine_version = await db.execute(
            select(MachineVersion)
            .join(Machine, Machine.id == MachineVersion.machine_id)
            .where(MachineVersion.id == body.machine_version_id)
            .apply_org_check_by_type(Machine, request)
        )
        machine_version = machine_version.scalar_one_or_none()
        if not machine_version:
            return JSONResponse(status_code=404, content={"error": "Machine version not found"})
        machine_id = machine_version.machine_id
        machine_version_id = machine_version.id
    elif body.machine_id:
        machine = await db.execute(
            select(Machine)
            .where(Machine.id == body.machine_id)
            .apply_org_check(request)
        )
        machine = machine.scalar_one_or_none()
        if not machine:
            return JSONResponse(status_code=404, content={"error": "Machine not found"})
        machine_id = machine.id

    try:
        # Your existing workflow creation code...
        org_id = (
            request.state.current_user["org_id"]
            if "org_id" in request.state.current_user
            else None
        )

        new_workflow = Workflow(
            id=uuid4(),
            user_id=user_id,
            org_id=org_id,
            name=body.name,
            selected_machine_id=machine_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        db.add(new_workflow)
        workflow_id = new_workflow.id

        # Create workflow version
        new_version = WorkflowVersion(
            id=uuid4(),
            workflow_id=workflow_id,
            workflow=json.loads(body.workflow_json),  # Convert string to dict
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            user_id=user_id,
            version=1,
            comment="initial version",
            machine_id=machine_id,
            machine_version_id=machine_version_id,
            comfyui_snapshot=body.comfyui_snapshot,
        )

        if body.workflow_api:
            new_version.workflow_api = json.loads(body.workflow_api)

        db.add(new_version)
        
        if machine_id is not None:
            await create_deployment(
                request,
                DeploymentCreate(
                    workflow_version_id=str(new_version.id),
                    workflow_id=str(workflow_id),
                    machine_id=str(machine_id),
                    machine_version_id=str(machine_version_id) if machine_version_id else None,
                    environment="preview",
                ),
                db=db,
            )

        await db.commit()

        return JSONResponse(
            status_code=200,
            content={
                "workflow_id": str(workflow_id),
                "machine_id": str(machine_id) if machine_id else None,
            },
        )

    except Exception as e:
        await db.rollback()
        raise
        # return JSONResponse(
        #     status_code=500, content={"error": f"Failed to create workflow: {str(e)}"}
        # )


class WorkflowVersionCreate(BaseModel):
    workflow: Dict[str, Any]
    workflow_api: Dict[str, Any]
    comment: Optional[str] = None
    machine_id: Optional[str] = None
    machine_version_id: Optional[str] = None
    comfyui_snapshot: Optional[Dict[str, Any]] = None


@router.post("/workflow/{workflow_id}/version")
async def create_workflow_version(
    request: Request,
    workflow_id: str,
    version_data: WorkflowVersionCreate,
    db: AsyncSession = Depends(get_db),
):
    try:
        current_user = request.state.current_user
        user_id = current_user["user_id"]
        org_id = current_user["org_id"] if "org_id" in current_user else None

        workflow = await db.execute(
            select(Workflow)
            .where(Workflow.id == workflow_id)
            .where(~Workflow.deleted)
            .limit(1)
            .apply_org_check(request)
        )
        workflow = workflow.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # async with db.begin():
        if version_data.machine_id and workflow.selected_machine_id != UUID(
            version_data.machine_id
        ):
            workflow.selected_machine_id = UUID(version_data.machine_id)

        # Get the next version number
        version_query = select(
            func.coalesce(func.max(WorkflowVersion.version), 0) + 1
        ).where(WorkflowVersion.workflow_id == workflow_id)
        result = await db.execute(version_query)
        next_version = result.scalar_one()

        # Create new version
        new_version = WorkflowVersion(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            workflow_id=workflow_id,
            version=next_version,
            user_id=user_id,
            workflow=version_data.workflow,
            workflow_api=version_data.workflow_api,
            comment=version_data.comment,
            machine_version_id=version_data.machine_version_id,
            machine_id=version_data.machine_id,
            comfyui_snapshot=version_data.comfyui_snapshot,
        )
        db.add(new_version)

        # Update workflow timestamp
        workflow.updated_at = func.now()

        if version_data.machine_id is not None:
            print("Creating deployment for staging")
            # When creating a new version, we also create a deployment for staging automatically
            result = await create_deployment(
                request,
                DeploymentCreate(
                    workflow_version_id=str(new_version.id),
                    workflow_id=workflow_id,
                    machine_id=version_data.machine_id,
                    environment="preview",
                ),
                db=db,
            )
            print(result)
        else:
            await db.commit()
            await db.flush()
            await db.refresh(workflow)

        return new_version.to_dict()

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
