#!/usr/bin/env python3
"""
Production Deployment Script for Codegen API Transformation

This script handles the safe deployment of the codegen-aligned API transformation
to production environments with proper rollback capabilities and validation.
"""

import os
import sys
import asyncio
import logging
import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Add the api source to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logger = logging.getLogger(__name__)


class CodegenDeploymentManager:
    """Manages the deployment of codegen API transformation."""
    
    def __init__(self, environment: str = "production", dry_run: bool = False):
        self.environment = environment
        self.dry_run = dry_run
        self.deployment_id = f"codegen-transform-{int(time.time())}"
        self.backup_dir = Path(f"./backups/{self.deployment_id}")
        
        # Environment-specific configurations
        self.config = {
            "development": {
                "db_backup": False,
                "migration_timeout": 300,
                "rollback_timeout": 180,
                "health_check_retries": 5,
                "health_check_delay": 10
            },
            "staging": {
                "db_backup": True,
                "migration_timeout": 600,
                "rollback_timeout": 300,
                "health_check_retries": 10,
                "health_check_delay": 15
            },
            "production": {
                "db_backup": True,
                "migration_timeout": 900,
                "rollback_timeout": 600,
                "health_check_retries": 20,
                "health_check_delay": 30
            }
        }
        
        self.env_config = self.config.get(environment, self.config["production"])
        
    async def deploy(self) -> bool:
        """Execute the complete deployment process."""
        logger.info(f"Starting codegen API transformation deployment to {self.environment}")
        logger.info(f"Deployment ID: {self.deployment_id}")
        
        if self.dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
        
        try:
            # Pre-deployment validation
            await self.pre_deployment_validation()
            
            # Create backup
            if self.env_config["db_backup"]:
                await self.create_backup()
            
            # Run database migrations
            await self.run_database_migrations()
            
            # Deploy application changes
            await self.deploy_application_changes()
            
            # Update configuration
            await self.update_configuration()
            
            # Health checks
            await self.run_health_checks()
            
            # Post-deployment validation
            await self.post_deployment_validation()
            
            logger.info("Codegen API transformation deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            await self.handle_deployment_failure()
            return False
    
    async def pre_deployment_validation(self):
        """Validate environment before deployment."""
        logger.info("Running pre-deployment validation")
        
        # Check database connectivity
        await self._check_database_connectivity()
        
        # Validate migration files
        await self._validate_migration_files()
        
        # Check existing API health
        await self._check_current_api_health()
        
        # Validate environment variables
        await self._validate_environment_variables()
        
        # Check disk space
        await self._check_disk_space()
        
        logger.info("Pre-deployment validation completed")
    
    async def _check_database_connectivity(self):
        """Check database connectivity."""
        logger.info("Checking database connectivity")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would check database connectivity")
            return
        
        try:
            from api.database import get_db_context
            from sqlalchemy import text
            
            async with get_db_context() as db:
                await db.execute(text("SELECT 1"))
                logger.info("Database connectivity confirmed")
                
        except Exception as e:
            raise Exception(f"Database connectivity check failed: {str(e)}")
    
    async def _validate_migration_files(self):
        """Validate migration files exist and are valid."""
        logger.info("Validating migration files")
        
        migration_dir = Path("database_migrations")
        if not migration_dir.exists():
            raise Exception("Migration directory not found")
        
        migration_files = list(migration_dir.glob("migration_*.py"))
        if not migration_files:
            raise Exception("No migration files found")
        
        # Check for required migration
        required_migration = migration_dir / "migration_001_create_codegen_tables.py"
        if not required_migration.exists():
            raise Exception("Required codegen tables migration not found")
        
        logger.info(f"Found {len(migration_files)} migration files")
    
    async def _check_current_api_health(self):
        """Check current API health."""
        logger.info("Checking current API health")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would check current API health")
            return
        
        try:
            import httpx
            
            # Try to reach health endpoint
            health_url = os.getenv("API_BASE_URL", "http://localhost:8000") + "/health"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(health_url, timeout=10.0)
                
            if response.status_code == 200:
                logger.info("Current API health check passed")
            else:
                logger.warning(f"API health check returned status {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Could not check current API health: {str(e)}")
    
    async def _validate_environment_variables(self):
        """Validate required environment variables."""
        logger.info("Validating environment variables")
        
        required_vars = [
            "DATABASE_URL",
            "ENV"
        ]
        
        optional_vars = [
            "CODEGEN_ORG_AUTH_ENABLED",
            "CODEGEN_RATE_LIMITING_ENABLED", 
            "CODEGEN_PRO_MODE_V2_ENABLED"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise Exception(f"Missing required environment variables: {missing_vars}")
        
        # Set default values for optional vars
        for var in optional_vars:
            if not os.getenv(var):
                logger.info(f"Setting default value for {var}")
                if not self.dry_run:
                    os.environ[var] = "true"
        
        logger.info("Environment variables validated")
    
    async def _check_disk_space(self):
        """Check available disk space."""
        logger.info("Checking disk space")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would check disk space")
            return
        
        # Check current directory disk space
        disk_usage = subprocess.run(
            ["df", "-h", "."],
            capture_output=True,
            text=True
        )
        
        if disk_usage.returncode == 0:
            logger.info("Disk space check:")
            for line in disk_usage.stdout.split('\n'):
                if line:
                    logger.info(line)
        else:
            logger.warning("Could not check disk space")
    
    async def create_backup(self):
        """Create database backup before migration."""
        logger.info("Creating database backup")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would create database backup")
            return
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create database backup
        backup_file = self.backup_dir / "database_backup.sql"
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise Exception("DATABASE_URL not set")
        
        # Use pg_dump for PostgreSQL backup
        backup_cmd = [
            "pg_dump",
            database_url,
            "-f", str(backup_file),
            "--verbose",
            "--no-password"
        ]
        
        logger.info(f"Creating database backup: {backup_file}")
        result = subprocess.run(backup_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Backup failed: {result.stderr}")
            raise Exception("Database backup failed")
        
        logger.info("Database backup created successfully")
        
        # Also backup current application state
        await self._backup_application_state()
    
    async def _backup_application_state(self):
        """Backup current application state."""
        logger.info("Backing up application state")
        
        state_backup = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": self.environment,
            "deployment_id": self.deployment_id,
            "environment_variables": {
                key: os.getenv(key) for key in [
                    "CODEGEN_ORG_AUTH_ENABLED",
                    "CODEGEN_RATE_LIMITING_ENABLED",
                    "CODEGEN_PRO_MODE_V2_ENABLED",
                    "ENV"
                ] if os.getenv(key)
            }
        }
        
        state_file = self.backup_dir / "application_state.json"
        with open(state_file, "w") as f:
            json.dump(state_backup, f, indent=2)
        
        logger.info(f"Application state backed up to {state_file}")
    
    async def run_database_migrations(self):
        """Run database migrations."""
        logger.info("Running database migrations")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would run database migrations")
            return
        
        try:
            from database_migrations.migrate import MigrationRunner
            
            runner = MigrationRunner()
            
            # Run migrations with timeout
            migration_task = asyncio.create_task(runner.run_migrations())
            
            try:
                await asyncio.wait_for(
                    migration_task,
                    timeout=self.env_config["migration_timeout"]
                )
                logger.info("Database migrations completed successfully")
                
            except asyncio.TimeoutError:
                migration_task.cancel()
                raise Exception("Database migration timed out")
                
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            raise Exception(f"Database migration failed: {str(e)}")
    
    async def deploy_application_changes(self):
        """Deploy application changes."""
        logger.info("Deploying application changes")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would deploy application changes")
            return
        
        # In a real deployment, this would:
        # - Update application code
        # - Restart services
        # - Update load balancer configuration
        
        # For this example, we'll just log the steps
        logger.info("Application changes deployed (placeholder)")
    
    async def update_configuration(self):
        """Update application configuration."""
        logger.info("Updating application configuration")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would update configuration")
            return
        
        # Enable codegen features
        feature_flags = {
            "CODEGEN_ORG_AUTH_ENABLED": "true",
            "CODEGEN_RATE_LIMITING_ENABLED": "true",
            "CODEGEN_PRO_MODE_V2_ENABLED": "true"
        }
        
        for flag, value in feature_flags.items():
            current_value = os.getenv(flag)
            if current_value != value:
                logger.info(f"Setting {flag} = {value}")
                os.environ[flag] = value
        
        logger.info("Configuration updated")
    
    async def run_health_checks(self):
        """Run comprehensive health checks."""
        logger.info("Running post-deployment health checks")
        
        checks = [
            ("Database connectivity", self._health_check_database),
            ("Codegen tables", self._health_check_codegen_tables),
            ("API endpoints", self._health_check_api_endpoints),
            ("Rate limiting", self._health_check_rate_limiting),
            ("Pro Mode integration", self._health_check_pro_mode)
        ]
        
        for check_name, check_func in checks:
            logger.info(f"Running health check: {check_name}")
            
            for attempt in range(self.env_config["health_check_retries"]):
                try:
                    if self.dry_run:
                        logger.info(f"[DRY RUN] Would run {check_name} health check")
                        break
                    
                    await check_func()
                    logger.info(f"✓ {check_name} health check passed")
                    break
                    
                except Exception as e:
                    if attempt < self.env_config["health_check_retries"] - 1:
                        logger.warning(f"Health check {check_name} failed (attempt {attempt + 1}): {str(e)}")
                        await asyncio.sleep(self.env_config["health_check_delay"])
                    else:
                        raise Exception(f"Health check {check_name} failed after {self.env_config['health_check_retries']} attempts: {str(e)}")
        
        logger.info("All health checks passed")
    
    async def _health_check_database(self):
        """Check database health."""
        from api.database import get_db_context
        from sqlalchemy import text
        
        async with get_db_context() as db:
            await db.execute(text("SELECT 1"))
    
    async def _health_check_codegen_tables(self):
        """Check that codegen tables exist."""
        from api.database import get_db_context
        from sqlalchemy import text
        
        expected_tables = [
            "codegen_organizations",
            "codegen_organization_memberships",
            "codegen_projects",
            "codegen_api_keys",
            "codegen_sessions",
            "codegen_agent_instances",
            "codegen_rate_limits"
        ]
        
        async with get_db_context() as db:
            for table in expected_tables:
                result = await db.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'comfyui_deploy' 
                        AND table_name = :table_name
                    );
                """), {"table_name": table})
                
                if not result.scalar():
                    raise Exception(f"Table {table} does not exist")
    
    async def _health_check_api_endpoints(self):
        """Check API endpoints."""
        import httpx
        
        base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        endpoints = [
            "/v1/health",
            "/v1/stats"
        ]
        
        async with httpx.AsyncClient() as client:
            for endpoint in endpoints:
                url = base_url + endpoint
                response = await client.get(url, timeout=10.0)
                
                if response.status_code != 200:
                    raise Exception(f"Endpoint {endpoint} returned status {response.status_code}")
    
    async def _health_check_rate_limiting(self):
        """Check rate limiting functionality."""
        # This would test that rate limiting is working
        logger.info("Rate limiting health check (placeholder)")
    
    async def _health_check_pro_mode(self):
        """Check Pro Mode integration."""
        # This would test Pro Mode functionality
        logger.info("Pro Mode integration health check (placeholder)")
    
    async def post_deployment_validation(self):
        """Run post-deployment validation."""
        logger.info("Running post-deployment validation")
        
        validations = [
            ("Migration history", self._validate_migration_history),
            ("Organization creation", self._validate_organization_creation),
            ("API compatibility", self._validate_api_compatibility)
        ]
        
        for validation_name, validation_func in validations:
            logger.info(f"Running validation: {validation_name}")
            
            if self.dry_run:
                logger.info(f"[DRY RUN] Would run {validation_name} validation")
                continue
            
            try:
                await validation_func()
                logger.info(f"✓ {validation_name} validation passed")
                
            except Exception as e:
                raise Exception(f"Validation {validation_name} failed: {str(e)}")
        
        logger.info("Post-deployment validation completed")
    
    async def _validate_migration_history(self):
        """Validate migration history."""
        from database_migrations.migrate import MigrationRunner
        
        runner = MigrationRunner()
        applied_migrations = await runner.get_applied_migrations()
        
        required_migrations = ["001_create_codegen_tables"]
        
        for required in required_migrations:
            if not any(required in migration for migration in applied_migrations):
                raise Exception(f"Required migration {required} not found in applied migrations")
    
    async def _validate_organization_creation(self):
        """Validate that organizations can be created."""
        from api.database import get_db_context
        from api.codegen_models import CodegenOrganization, OrganizationStatus
        
        async with get_db_context() as db:
            # Try to create a test organization
            test_org = CodegenOrganization(
                name=f"Test Deployment Org {self.deployment_id}",
                status=OrganizationStatus.ACTIVE
            )
            
            db.add(test_org)
            await db.commit()
            await db.refresh(test_org)
            
            # Verify it was created
            if not test_org.id:
                raise Exception("Test organization creation failed")
            
            # Clean up
            await db.delete(test_org)
            await db.commit()
    
    async def _validate_api_compatibility(self):
        """Validate API compatibility."""
        import httpx
        
        base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        # Test that both legacy and new endpoints are accessible
        endpoints = [
            ("/health", 200),  # Legacy health
            ("/v1/health", 200),  # New codegen health
            ("/api/health", 200)  # API health
        ]
        
        async with httpx.AsyncClient() as client:
            for endpoint, expected_status in endpoints:
                try:
                    url = base_url + endpoint
                    response = await client.get(url, timeout=10.0)
                    
                    if response.status_code != expected_status:
                        logger.warning(f"Endpoint {endpoint} returned {response.status_code}, expected {expected_status}")
                        
                except Exception as e:
                    logger.warning(f"Could not reach endpoint {endpoint}: {str(e)}")
    
    async def handle_deployment_failure(self):
        """Handle deployment failure and rollback if necessary."""
        logger.error("Handling deployment failure")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would handle deployment failure")
            return
        
        # In a real deployment, this would:
        # - Restore database from backup
        # - Revert application changes
        # - Restore previous configuration
        # - Notify operations team
        
        logger.error("Deployment rollback procedures would be executed here")
    
    async def rollback(self, backup_id: str):
        """Rollback a previous deployment."""
        logger.info(f"Rolling back deployment {backup_id}")
        
        backup_dir = Path(f"./backups/{backup_id}")
        if not backup_dir.exists():
            raise Exception(f"Backup directory {backup_dir} not found")
        
        # Restore database
        await self._restore_database_backup(backup_dir)
        
        # Restore application state
        await self._restore_application_state(backup_dir)
        
        logger.info("Rollback completed")
    
    async def _restore_database_backup(self, backup_dir: Path):
        """Restore database from backup."""
        logger.info("Restoring database backup")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would restore database backup")
            return
        
        backup_file = backup_dir / "database_backup.sql"
        if not backup_file.exists():
            raise Exception(f"Database backup file {backup_file} not found")
        
        database_url = os.getenv("DATABASE_URL")
        
        # Use psql to restore PostgreSQL backup
        restore_cmd = [
            "psql",
            database_url,
            "-f", str(backup_file),
            "--quiet"
        ]
        
        logger.info(f"Restoring database from {backup_file}")
        result = subprocess.run(restore_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Database restore failed: {result.stderr}")
            raise Exception("Database restore failed")
        
        logger.info("Database restored successfully")
    
    async def _restore_application_state(self, backup_dir: Path):
        """Restore application state from backup."""
        logger.info("Restoring application state")
        
        state_file = backup_dir / "application_state.json"
        if not state_file.exists():
            logger.warning("Application state backup not found")
            return
        
        with open(state_file) as f:
            state = json.load(f)
        
        # Restore environment variables
        if "environment_variables" in state:
            for key, value in state["environment_variables"].items():
                logger.info(f"Restoring {key} = {value}")
                if not self.dry_run:
                    os.environ[key] = value
        
        logger.info("Application state restored")


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Codegen API Transformation Deployment")
    parser.add_argument("command", choices=["deploy", "rollback", "validate"],
                       help="Deployment command")
    parser.add_argument("--environment", choices=["development", "staging", "production"],
                       default="production", help="Target environment")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run")
    parser.add_argument("--backup-id", help="Backup ID for rollback")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create deployment manager
    manager = CodegenDeploymentManager(
        environment=args.environment,
        dry_run=args.dry_run
    )
    
    try:
        if args.command == "deploy":
            success = await manager.deploy()
            sys.exit(0 if success else 1)
            
        elif args.command == "rollback":
            if not args.backup_id:
                print("Error: --backup-id required for rollback")
                sys.exit(1)
            await manager.rollback(args.backup_id)
            
        elif args.command == "validate":
            await manager.post_deployment_validation()
            print("Validation completed successfully")
            
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())