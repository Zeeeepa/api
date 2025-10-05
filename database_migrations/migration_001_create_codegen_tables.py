"""
Migration 001: Create Codegen Tables

This migration creates the codegen-aligned database tables for organization-centric
API architecture while preserving existing ComfyUI-focused tables for backward compatibility.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional
import uuid
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, MetaData, Table, Column, Integer, String, Boolean, DateTime, Float, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.sql import func

from api.database import get_db_context
from api.codegen_models import (
    CodegenOrganization, CodegenOrganizationMembership, CodegenProject,
    CodegenAPIKey, CodegenAgentInstance, CodegenSession, CodegenRateLimit,
    Base as CodegenBase, metadata
)

logger = logging.getLogger(__name__)

# Migration metadata
MIGRATION_ID = "001_create_codegen_tables"
MIGRATION_DESCRIPTION = "Create codegen-aligned tables for organization-centric API"
MIGRATION_VERSION = "1.0.0"


async def check_migration_applied() -> bool:
    """Check if this migration has already been applied."""
    async with get_db_context() as db:
        try:
            # Check if migration tracking table exists
            result = await db.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'comfyui_deploy' 
                    AND table_name = 'migration_history'
                );
            """))
            
            table_exists = result.scalar()
            
            if not table_exists:
                # Create migration tracking table
                await db.execute(text("""
                    CREATE TABLE IF NOT EXISTS comfyui_deploy.migration_history (
                        id SERIAL PRIMARY KEY,
                        migration_id VARCHAR(255) UNIQUE NOT NULL,
                        description TEXT,
                        version VARCHAR(50),
                        applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        applied_by VARCHAR(255),
                        rollback_script TEXT
                    );
                """))
                await db.commit()
                return False
            
            # Check if this specific migration was applied
            result = await db.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM comfyui_deploy.migration_history 
                    WHERE migration_id = :migration_id
                );
            """), {"migration_id": MIGRATION_ID})
            
            return result.scalar()
            
        except Exception as e:
            logger.error(f"Error checking migration status: {str(e)}")
            return False


async def apply_migration():
    """Apply the migration to create codegen tables."""
    logger.info(f"Applying migration {MIGRATION_ID}: {MIGRATION_DESCRIPTION}")
    
    async with get_db_context() as db:
        try:
            # Create all codegen tables using SQLAlchemy metadata
            await db.run_sync(CodegenBase.metadata.create_all)
            
            # Record migration in history
            await db.execute(text("""
                INSERT INTO comfyui_deploy.migration_history 
                (migration_id, description, version, applied_by, rollback_script)
                VALUES (:migration_id, :description, :version, :applied_by, :rollback_script)
                ON CONFLICT (migration_id) DO NOTHING;
            """), {
                "migration_id": MIGRATION_ID,
                "description": MIGRATION_DESCRIPTION,
                "version": MIGRATION_VERSION,
                "applied_by": "codegen_migration_system",
                "rollback_script": ROLLBACK_SCRIPT
            })
            
            await db.commit()
            logger.info(f"Migration {MIGRATION_ID} applied successfully")
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to apply migration {MIGRATION_ID}: {str(e)}")
            raise


async def rollback_migration():
    """Rollback the migration by dropping codegen tables."""
    logger.info(f"Rolling back migration {MIGRATION_ID}")
    
    async with get_db_context() as db:
        try:
            # Execute rollback script
            rollback_statements = ROLLBACK_SCRIPT.split(";")
            
            for statement in rollback_statements:
                if statement.strip():
                    await db.execute(text(statement.strip()))
            
            # Remove from migration history
            await db.execute(text("""
                DELETE FROM comfyui_deploy.migration_history 
                WHERE migration_id = :migration_id;
            """), {"migration_id": MIGRATION_ID})
            
            await db.commit()
            logger.info(f"Migration {MIGRATION_ID} rolled back successfully")
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to rollback migration {MIGRATION_ID}: {str(e)}")
            raise


# Rollback script to drop all codegen tables
ROLLBACK_SCRIPT = """
-- Drop codegen tables in reverse dependency order
DROP TABLE IF EXISTS comfyui_deploy.codegen_rate_limits CASCADE;
DROP TABLE IF EXISTS comfyui_deploy.codegen_agent_instances CASCADE;
DROP TABLE IF EXISTS comfyui_deploy.codegen_sessions CASCADE;
DROP TABLE IF EXISTS comfyui_deploy.codegen_api_keys CASCADE;
DROP TABLE IF EXISTS comfyui_deploy.codegen_projects CASCADE;
DROP TABLE IF EXISTS comfyui_deploy.codegen_organization_memberships CASCADE;
DROP TABLE IF EXISTS comfyui_deploy.codegen_organizations CASCADE;

-- Drop custom enums
DROP TYPE IF EXISTS organizationrole CASCADE;
DROP TYPE IF EXISTS organizationstatus CASCADE;
DROP TYPE IF EXISTS projectstatus CASCADE;
DROP TYPE IF EXISTS agentinstancestatus CASCADE;
DROP TYPE IF EXISTS sessionstatus CASCADE;
"""


async def validate_migration():
    """Validate that the migration was applied correctly."""
    logger.info(f"Validating migration {MIGRATION_ID}")
    
    async with get_db_context() as db:
        try:
            # Check that all expected tables exist
            expected_tables = [
                'codegen_organizations',
                'codegen_organization_memberships', 
                'codegen_projects',
                'codegen_api_keys',
                'codegen_agent_instances',
                'codegen_sessions',
                'codegen_rate_limits'
            ]
            
            for table_name in expected_tables:
                result = await db.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'comfyui_deploy' 
                        AND table_name = :table_name
                    );
                """), {"table_name": table_name})
                
                if not result.scalar():
                    raise Exception(f"Table {table_name} was not created")
            
            # Check that indexes were created
            expected_indexes = [
                'ix_codegen_organizations_status',
                'ix_codegen_org_memberships_org_user',
                'ix_codegen_projects_org_id',
                'ix_codegen_api_keys_key_hash',
                'ix_codegen_agent_instances_org_id',
                'ix_codegen_sessions_org_id',
                'ix_codegen_rate_limits_org_endpoint'
            ]
            
            for index_name in expected_indexes:
                result = await db.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM pg_indexes 
                        WHERE schemaname = 'comfyui_deploy' 
                        AND indexname = :index_name
                    );
                """), {"index_name": index_name})
                
                if not result.scalar():
                    logger.warning(f"Index {index_name} was not created")
            
            logger.info(f"Migration {MIGRATION_ID} validation completed successfully")
            
        except Exception as e:
            logger.error(f"Migration validation failed: {str(e)}")
            raise


async def create_default_data():
    """Create default data after migration."""
    logger.info("Creating default data for codegen tables")
    
    async with get_db_context() as db:
        try:
            # Create a default organization for existing users
            result = await db.execute(text("""
                SELECT COUNT(*) FROM comfyui_deploy.users;
            """))
            user_count = result.scalar() or 0
            
            if user_count > 0:
                # Create default organization
                default_org = CodegenOrganization(
                    name="Default Organization",
                    display_name="Default Organization", 
                    description="Default organization for existing users",
                    status="active"
                )
                
                db.add(default_org)
                await db.commit()
                await db.refresh(default_org)
                
                # Add all existing users to the default organization
                result = await db.execute(text("""
                    SELECT id FROM comfyui_deploy.users LIMIT 100;
                """))
                users = result.fetchall()
                
                for user_row in users:
                    membership = CodegenOrganizationMembership(
                        organization_id=default_org.id,
                        user_id=user_row.id,
                        role="owner",  # Make existing users owners of default org
                        joined_at=datetime.now(timezone.utc)
                    )
                    db.add(membership)
                
                await db.commit()
                logger.info(f"Created default organization with {len(users)} members")
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to create default data: {str(e)}")
            # Don't raise here - migration can succeed without default data


async def run_migration():
    """Main migration runner function."""
    try:
        # Check if migration is already applied
        if await check_migration_applied():
            logger.info(f"Migration {MIGRATION_ID} is already applied, skipping")
            return
        
        # Apply the migration
        await apply_migration()
        
        # Validate the migration
        await validate_migration()
        
        # Create default data
        await create_default_data()
        
        logger.info(f"Migration {MIGRATION_ID} completed successfully")
        
    except Exception as e:
        logger.error(f"Migration {MIGRATION_ID} failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Allow running migration directly
    asyncio.run(run_migration())