import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional
from .types import WorkflowModel
from api.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import UserIconData, clean_up_outputs, fetch_user_icon, has, post_process_output_data, require_permission, select, post_process_outputs
from api.models import Deployment, User, Workflow, WorkflowRun, WorkflowVersion, WorkflowRunOutput
from .utils import get_user_settings, is_exceed_spend_limit
from sqlalchemy import func, select as sa_select, distinct, and_, or_
from sqlalchemy.orm import joinedload, load_only, contains_eager
from fastapi.responses import JSONResponse
from pprint import pprint
from sqlalchemy import desc
from sqlalchemy import text
from datetime import datetime
from uuid import UUID
from decimal import Decimal
import asyncio

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, UUID)):
            return str(obj)
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, UUID)):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

router = APIRouter(tags=["Workflow"])

@router.get("/workflows", response_model=List[WorkflowModel])
async def get_workflows(
    request: Request,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    raw_query = text("""
    WITH RECURSIVE search_param(term) AS (
        SELECT lower(:search)
    )
    SELECT
        wf.id AS id,
        wf.name AS name, 
        wf.created_at AS created_at, 
        wf.updated_at AS updated_at, 
        users.name AS user_name,
        users.id AS user_id
    FROM 
        comfyui_deploy.workflows AS wf
    INNER JOIN 
        comfyui_deploy.users AS users ON users.id = wf.user_id
    WHERE 
        wf.deleted = false AND ((CAST(:org_id AS TEXT) IS NOT NULL AND wf.org_id = CAST(:org_id AS TEXT))
            OR (CAST(:org_id AS TEXT) IS NULL AND org_id IS NULL AND wf.user_id = CAST(:user_id AS TEXT)))
            AND (CAST(:search AS TEXT) IS NULL OR lower(wf.name) LIKE '%' || (SELECT term FROM search_param) || '%')
    ORDER BY 
        wf.pinned DESC,
        wf.updated_at DESC
    LIMIT :limit
    OFFSET :offset
    """)

    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"] if 'org_id' in current_user else None
    
    # Execute the query
    result = await db.execute(
        raw_query,
        {
            "search": search,
            "limit": limit,
            "offset": offset,
            "org_id": org_id,
            "user_id": user_id,
        },
    )

    workflows = [
        dict(row._mapping)
        for row in result.fetchall()
    ]
    
    user_settings = await get_user_settings(request, db)
    
    # Fetch user icons
    unique_user_ids = list(set(workflow["user_id"] for workflow in workflows if workflow["user_id"]))
    user_icon_results = await asyncio.gather(
        *[fetch_user_icon(user_id) for user_id in unique_user_ids]
    )
    user_icons = {str(user_id): icon_data for user_id, icon_data in zip(unique_user_ids, user_icon_results)}

    for workflow in workflows:
        if "latest_output" in workflow and workflow["latest_output"]:
            post_process_output_data(workflow["latest_output"], user_settings)
        user_icon = user_icons.get(str(workflow["user_id"]))
        workflow["user_icon"] = user_icon.image_url if user_icon else None

    # Use the custom encoder to serialize the data
    return JSONResponse(
        status_code=200, 
        content=json.loads(json.dumps(workflows, cls=CustomJSONEncoder))
    )

@router.get("/workflows/all", response_model=List[WorkflowModel])
async def get_all_workflows(
    request: Request,
    search: Optional[str] = None,
    limit: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    workflows_query = (
        select(Workflow)
        .order_by(Workflow.created_at.desc())
        .where(~Workflow.deleted)
        .apply_org_check(request)
    )

    if search:
        workflows_query = workflows_query.where(
            func.lower(Workflow.name).like(f"%{search.lower()}%")
        )

    if limit:
        workflows_query = workflows_query.limit(limit)

    result = await db.execute(workflows_query)
    workflows = result.scalars().all()

    return JSONResponse(content=[workflow.to_dict() for workflow in workflows])