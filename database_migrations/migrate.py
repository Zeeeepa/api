"""
Database Migration Runner for Codegen API Transformation

This script handles the execution of database migrations for transforming
the ComfyUI-focused API to codegen-aligned architecture.
"""

import asyncio
import sys
import os
import importlib
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone

# Add the api source to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from api.database import get_db_context

logger = logging.getLogger(__name__)


class MigrationRunner:
    """Handles database migration execution and management."""
    
    def __init__(self, migrations_dir: Path = None):
        self.migrations_dir = migrations_dir or Path(__file__).parent
        self.available_migrations = self._discover_migrations()
        
    def _discover_migrations(self) -> List[Dict[str, Any]]:
        """Discover available migration files."""
        migrations = []
        
        # Look for migration files matching pattern migration_XXX_*.py
        for migration_file in self.migrations_dir.glob("migration_*.py"):
            if migration_file.name == "migrate.py":
                continue
                
            try:
                # Extract migration number and name
                name_parts = migration_file.stem.split("_", 2)
                if len(name_parts) >= 2:
                    migration_num = int(name_parts[1])
                    migration_name = "_".join(name_parts[2:]) if len(name_parts) > 2 else ""
                    
                    migrations.append({
                        "number": migration_num,
                        "name": migration_name,
                        "file": migration_file,
                        "module_name": migration_file.stem
                    })
                    
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping invalid migration file {migration_file}: {e}")
                
        # Sort by migration number
        migrations.sort(key=lambda x: x["number"])
        return migrations
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of migrations that have been applied."""
        async with get_db_context() as db:
            try:
                # Ensure migration history table exists
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
                
                # Get applied migrations
                result = await db.execute(text("""
                    SELECT migration_id FROM comfyui_deploy.migration_history 
                    ORDER BY applied_at ASC;
                """))
                
                return [row[0] for row in result.fetchall()]
                
            except Exception as e:
                logger.error(f"Failed to get applied migrations: {str(e)}")
                return []
    
    async def run_migrations(self, target_migration: int = None, dry_run: bool = False):
        """Run migrations up to target migration."""
        logger.info("Starting database migration process")
        
        applied_migrations = await self.get_applied_migrations()
        logger.info(f"Found {len(applied_migrations)} previously applied migrations")
        
        migrations_to_run = []
        
        for migration in self.available_migrations:
            migration_id = f"{migration['number']:03d}_{migration['name']}"
            
            # Skip if already applied
            if migration_id in applied_migrations:
                logger.info(f"Migration {migration_id} already applied, skipping")
                continue
            
            # Check target migration limit
            if target_migration and migration["number"] > target_migration:
                logger.info(f"Stopping at target migration {target_migration}")
                break
                
            migrations_to_run.append(migration)
        
        if not migrations_to_run:
            logger.info("No migrations to run")
            return
        
        logger.info(f"Running {len(migrations_to_run)} migrations")
        
        for migration in migrations_to_run:
            migration_id = f"{migration['number']:03d}_{migration['name']}"
            
            if dry_run:
                logger.info(f"[DRY RUN] Would run migration {migration_id}")
                continue
            
            try:
                logger.info(f"Running migration {migration_id}")
                
                # Import and run the migration
                module = importlib.import_module(migration["module_name"])
                
                if hasattr(module, "run_migration"):
                    await module.run_migration()
                    logger.info(f"Migration {migration_id} completed successfully")
                else:
                    logger.error(f"Migration {migration_id} missing run_migration function")
                    
            except Exception as e:
                logger.error(f"Migration {migration_id} failed: {str(e)}")
                raise
    
    async def rollback_migration(self, migration_id: str):
        """Rollback a specific migration."""
        logger.info(f"Rolling back migration {migration_id}")
        
        applied_migrations = await self.get_applied_migrations()
        
        if migration_id not in applied_migrations:
            logger.error(f"Migration {migration_id} is not applied")
            return
        
        # Find the migration file
        migration_info = None
        for migration in self.available_migrations:
            if f"{migration['number']:03d}_{migration['name']}" == migration_id:
                migration_info = migration
                break
        
        if not migration_info:
            logger.error(f"Migration file for {migration_id} not found")
            return
        
        try:
            # Import and run rollback
            module = importlib.import_module(migration_info["module_name"])
            
            if hasattr(module, "rollback_migration"):
                await module.rollback_migration()
                logger.info(f"Migration {migration_id} rolled back successfully")
            else:
                logger.error(f"Migration {migration_id} missing rollback_migration function")
                
        except Exception as e:
            logger.error(f"Rollback of migration {migration_id} failed: {str(e)}")
            raise
    
    async def status(self):
        """Show migration status."""
        applied_migrations = await self.get_applied_migrations()
        
        print(f"\nMigration Status:")
        print(f"{'Migration ID':<30} {'Status':<15} {'Applied At'}")
        print("-" * 60)
        
        async with get_db_context() as db:
            for migration in self.available_migrations:
                migration_id = f"{migration['number']:03d}_{migration['name']}"
                
                if migration_id in applied_migrations:
                    # Get application date
                    result = await db.execute(text("""
                        SELECT applied_at FROM comfyui_deploy.migration_history 
                        WHERE migration_id = :migration_id;
                    """), {"migration_id": migration_id})
                    
                    applied_at = result.scalar()
                    status = "APPLIED"
                    date_str = applied_at.strftime("%Y-%m-%d %H:%M:%S") if applied_at else "Unknown"
                else:
                    status = "PENDING"
                    date_str = "-"
                
                print(f"{migration_id:<30} {status:<15} {date_str}")
        
        print(f"\nTotal migrations: {len(self.available_migrations)}")
        print(f"Applied: {len(applied_migrations)}")
        print(f"Pending: {len(self.available_migrations) - len(applied_migrations)}")
    
    async def validate_database(self):
        """Validate database state after migrations."""
        logger.info("Validating database state")
        
        async with get_db_context() as db:
            try:
                # Check that both legacy and codegen tables exist
                legacy_tables = [
                    'users', 'workflows', 'workflow_runs', 'workflow_versions',
                    'machines', 'deployments', 'api_keys'
                ]
                
                codegen_tables = [
                    'codegen_organizations', 'codegen_organization_memberships',
                    'codegen_projects', 'codegen_api_keys', 'codegen_sessions',
                    'codegen_agent_instances', 'codegen_rate_limits'
                ]
                
                print("\nDatabase Validation:")
                print(f"{'Table':<35} {'Status'}")
                print("-" * 50)
                
                for table in legacy_tables + codegen_tables:
                    result = await db.execute(text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'comfyui_deploy' 
                            AND table_name = :table_name
                        );
                    """), {"table_name": table})
                    
                    exists = result.scalar()
                    status = "EXISTS" if exists else "MISSING"
                    marker = "✓" if exists else "✗"
                    
                    print(f"{table:<35} {status} {marker}")
                
                print("\nValidation completed")
                
            except Exception as e:
                logger.error(f"Database validation failed: {str(e)}")
                raise


async def main():
    """Main CLI function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Migration Runner")
    parser.add_argument("command", choices=["run", "status", "rollback", "validate"], 
                       help="Migration command to execute")
    parser.add_argument("--target", type=int, help="Target migration number")
    parser.add_argument("--migration-id", help="Migration ID for rollback")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    runner = MigrationRunner()
    
    try:
        if args.command == "run":
            await runner.run_migrations(target_migration=args.target, dry_run=args.dry_run)
            
        elif args.command == "status":
            await runner.status()
            
        elif args.command == "rollback":
            if not args.migration_id:
                print("Error: --migration-id required for rollback")
                sys.exit(1)
            await runner.rollback_migration(args.migration_id)
            
        elif args.command == "validate":
            await runner.validate_database()
            
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())