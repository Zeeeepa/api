"""
Codegen_adapter.py

Comprehensive adapter for integrating Zeeeepa/api with codegen CLI functionality.
This package provides a unified interface for all codegen operations, combining
CLI commands with API endpoints for seamless agentic usage and UI project management.

Author: Claude Code Integration
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json

# Core codegen imports
try:
    from codegen.cli.api.client import get_api_client, is_api_client_available
    from codegen.cli.api.endpoints import *
    from codegen.cli.api.schemas import *
except ImportError as e:
    print(f"Warning: Could not import codegen API client: {e}")
    print("Please ensure codegen is installed as a dependency")

# CLI Command imports
try:
    # Agent management
    from codegen.cli.commands.agent.main import create as create_agent, list_agents
    
    # Integration management
    from codegen.cli.commands.integrations.main import list_integrations, add_integration
    
    # Authentication
    from codegen.cli.commands.login.main import login
    from codegen.cli.commands.logout.main import logout
    from codegen.cli.commands.profile.main import profile_main
    from codegen.cli.commands.org.main import org
    
    # Repository management
    from codegen.cli.commands.repo.main import repo, _list_repositories, _set_default_repository
    
    # Claude integration
    from codegen.cli.commands.claude.main import _run_claude_interactive, _run_claude_background, claude
    from codegen.cli.commands.claude.hooks import ensure_claude_hook, cleanup_claude_hook, get_codegen_url
    from codegen.cli.commands.claude.claude_session_api import create_claude_session
    from codegen.cli.commands.claude.claude_log_watcher import ClaudeLogWatcher, ClaudeLogWatcherManager
    
    # Tools and MCP
    from codegen.cli.commands.tools.main import tools
    from codegen.cli.mcp.tools.dynamic import register_dynamic_tools
    from codegen.cli.mcp.tools.executor import execute_tool_via_api
    from codegen.cli.mcp.tools.static import register_static_tools
    from codegen.cli.mcp.api_client import get_api_client as get_mcp_api_client
    from codegen.cli.mcp.resources import register_resources
    from codegen.cli.mcp.runner import run_server
    
    # TUI components
    from codegen.cli.tui.app import MinimalTUI
    from codegen.cli.tui.agent_detail import AgentDetailTUI
    
    # Telemetry
    from codegen.cli.telemetry.viewer import print_span_tree, load_session, analyze_session
    from codegen.cli.telemetry.otel_setup import _get_claude_info, _get_otlp_logs_endpoint
    
    # CLI base
    from codegen.cli.cli import main_callback
    
except ImportError as e:
    print(f"Warning: Could not import some codegen CLI functions: {e}")
    print("Some functionality may be limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodegenAdapterError(Exception):
    """Base exception for Codegen Adapter errors"""
    pass


class AuthenticationError(CodegenAdapterError):
    """Authentication-related errors"""
    pass


class APIError(CodegenAdapterError):
    """API-related errors"""
    pass


class BaseManager:
    """Base class for all manager classes"""
    
    def __init__(self, api_client=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_client = api_client or self._get_api_client()
    
    def _get_api_client(self):
        """Get or create API client instance"""
        try:
            # Check if codegen functions are available
            if 'is_api_client_available' in globals() and is_api_client_available():
                return get_api_client()
            else:
                # Return a mock client for testing environments
                self.logger.warning("Codegen API client not available, using mock client")
                return self._create_mock_client()
        except Exception as e:
            self.logger.error(f"Failed to get API client: {e}")
            # Return mock client instead of raising error for testing
            self.logger.warning("Using mock client due to API client failure")
            return self._create_mock_client()
    
    def _create_mock_client(self):
        """Create a mock API client for testing"""
        class MockClient:
            def __init__(self):
                pass
        return MockClient()


class AgentManager(BaseManager):
    """
    Comprehensive agent management combining CLI functions with API endpoints.
    
    Provides methods for:
    - Creating agent runs via API
    - Listing agent runs with filtering
    - Managing agent status (resume, ban, unban)
    - Getting agent logs and details
    - Background and interactive agent modes
    """
    
    async def create_agent_run(
        self, 
        org_id: int,
        prompt: str,
        images: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        repo_id: Optional[int] = None,
        model: Optional[str] = None,
        agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new agent run via API"""
        try:
            payload = {
                "prompt": prompt,
                "images": images,
                "metadata": metadata,
                "repo_id": repo_id,
                "model": model,
                "agent_type": agent_type
            }
            # Filter out None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            response = await self.api_client.post(f"/v1/organizations/{org_id}/agent/run", json=payload)
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to create agent run: {e}")
            raise APIError(f"Failed to create agent run: {e}")
    
    async def get_agent_run(self, org_id: int, agent_run_id: int) -> Dict[str, Any]:
        """Get agent run status and details"""
        try:
            response = await self.api_client.get(f"/v1/organizations/{org_id}/agent/run/{agent_run_id}")
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get agent run: {e}")
            raise APIError(f"Failed to get agent run: {e}")
    
    async def list_agent_runs(
        self,
        org_id: int,
        user_id: Optional[int] = None,
        source_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> Dict[str, Any]:
        """List agent runs with optional filtering"""
        try:
            params = {
                "skip": skip,
                "limit": limit
            }
            if user_id:
                params["user_id"] = user_id
            if source_type:
                params["source_type"] = source_type
            
            response = await self.api_client.get(
                f"/v1/organizations/{org_id}/agent/runs", 
                params=params
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to list agent runs: {e}")
            raise APIError(f"Failed to list agent runs: {e}")
    
    async def resume_agent_run(
        self,
        org_id: int,
        agent_run_id: int,
        prompt: str,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Resume a paused agent run"""
        try:
            payload = {
                "agent_run_id": agent_run_id,
                "prompt": prompt,
                "images": images
            }
            response = await self.api_client.post(
                f"/v1/organizations/{org_id}/agent/run/resume",
                json=payload
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to resume agent run: {e}")
            raise APIError(f"Failed to resume agent run: {e}")
    
    async def ban_agent_run(self, org_id: int, agent_run_id: int) -> Dict[str, Any]:
        """Ban all checks for an agent run"""
        try:
            payload = {"agent_run_id": agent_run_id}
            response = await self.api_client.post(
                f"/v1/organizations/{org_id}/agent/run/ban",
                json=payload
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to ban agent run: {e}")
            raise APIError(f"Failed to ban agent run: {e}")
    
    async def unban_agent_run(self, org_id: int, agent_run_id: int) -> Dict[str, Any]:
        """Unban all checks for an agent run"""
        try:
            payload = {"agent_run_id": agent_run_id}
            response = await self.api_client.post(
                f"/v1/organizations/{org_id}/agent/run/unban",
                json=payload
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to unban agent run: {e}")
            raise APIError(f"Failed to unban agent run: {e}")
    
    async def get_agent_logs(
        self,
        org_id: int,
        agent_run_id: int,
        skip: int = 0,
        limit: int = 100,
        reverse: bool = False
    ) -> Dict[str, Any]:
        """Get agent run logs with pagination"""
        try:
            params = {
                "skip": skip,
                "limit": limit,
                "reverse": reverse
            }
            response = await self.api_client.get(
                f"/v1/alpha/organizations/{org_id}/agent/run/{agent_run_id}/logs",
                params=params
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get agent logs: {e}")
            raise APIError(f"Failed to get agent logs: {e}")
    
    def create_background_agent(self, prompt: str, **kwargs):
        """Create a background agent using CLI functions"""
        try:
            return create_agent(prompt, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create background agent: {e}")
            raise CodegenAdapterError(f"Failed to create background agent: {e}")
    
    def list_agents_cli(self, **kwargs):
        """List agents using CLI functions"""
        try:
            return list_agents(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to list agents via CLI: {e}")
            raise CodegenAdapterError(f"Failed to list agents via CLI: {e}")


class IntegrationManager(BaseManager):
    """
    Comprehensive integration management for all external services.
    
    Supports:
    - OAuth-based integrations (Slack, Linear, Notion, Figma, ClickUp, Jira, Sentry, Monday.com)
    - GitHub app installations
    - API key-based integrations (CircleCI)  
    - Database connections (PostgreSQL)
    """
    
    async def get_organization_integrations(self, org_id: int) -> Dict[str, Any]:
        """Get all integration statuses for an organization"""
        try:
            response = await self.api_client.get(f"/v1/organizations/{org_id}/integrations")
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get organization integrations: {e}")
            raise APIError(f"Failed to get organization integrations: {e}")
    
    def list_integrations_cli(self, **kwargs):
        """List integrations using CLI functions"""
        try:
            return list_integrations(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to list integrations via CLI: {e}")
            raise CodegenAdapterError(f"Failed to list integrations via CLI: {e}")
    
    def add_integration_cli(self, integration_type: str, **kwargs):
        """Add integration using CLI functions"""
        try:
            return add_integration(integration_type, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to add integration via CLI: {e}")
            raise CodegenAdapterError(f"Failed to add integration via CLI: {e}")


class RepositoryManager(BaseManager):
    """
    Repository and pull request management combining CLI and API functionality.
    
    Features:
    - Repository listing and management
    - Check suite settings configuration
    - Pull request state management
    - Default repository settings
    """
    
    async def get_repositories(
        self, 
        org_id: int, 
        skip: int = 0, 
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get repositories for an organization"""
        try:
            params = {"skip": skip, "limit": limit}
            response = await self.api_client.get(
                f"/v1/organizations/{org_id}/repos",
                params=params
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get repositories: {e}")
            raise APIError(f"Failed to get repositories: {e}")
    
    async def get_check_suite_settings(self, org_id: int, repo_id: int) -> Dict[str, Any]:
        """Get check suite settings for a repository"""
        try:
            response = await self.api_client.get(
                f"/v1/organizations/{org_id}/repos/check-suite-settings",
                params={"repo_id": repo_id}
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get check suite settings: {e}")
            raise APIError(f"Failed to get check suite settings: {e}")
    
    async def update_check_suite_settings(
        self,
        org_id: int,
        repo_id: int,
        settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update check suite settings for a repository"""
        try:
            response = await self.api_client.put(
                f"/v1/organizations/{org_id}/repos/check-suite-settings",
                params={"repo_id": repo_id},
                json=settings
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to update check suite settings: {e}")
            raise APIError(f"Failed to update check suite settings: {e}")
    
    async def edit_pull_request(
        self,
        org_id: int,
        pr_id: int,
        state: str,
        repo_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Edit pull request state"""
        try:
            payload = {"state": state}
            
            # Use simple endpoint if repo_id not provided
            if repo_id is None:
                endpoint = f"/v1/organizations/{org_id}/prs/{pr_id}"
            else:
                endpoint = f"/v1/organizations/{org_id}/repos/{repo_id}/prs/{pr_id}"
            
            response = await self.api_client.patch(endpoint, json=payload)
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to edit pull request: {e}")
            raise APIError(f"Failed to edit pull request: {e}")
    
    def list_repositories_cli(self, **kwargs):
        """List repositories using CLI functions"""
        try:
            return _list_repositories(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to list repositories via CLI: {e}")
            raise CodegenAdapterError(f"Failed to list repositories via CLI: {e}")
    
    def set_default_repository_cli(self, repo_name: str, **kwargs):
        """Set default repository using CLI functions"""
        try:
            return _set_default_repository(repo_name, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to set default repository via CLI: {e}")
            raise CodegenAdapterError(f"Failed to set default repository via CLI: {e}")


class ClaudeManager(BaseManager):
    """
    Claude Code integration and session management.
    
    Features:
    - Interactive and background Claude runs
    - Session management and tracking
    - Log watching and analysis  
    - Hook management for Claude Code
    - Telemetry and monitoring
    """
    
    def run_claude_interactive(self, **kwargs):
        """Run Claude in interactive mode"""
        try:
            return _run_claude_interactive(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to run Claude interactive: {e}")
            raise CodegenAdapterError(f"Failed to run Claude interactive: {e}")
    
    def run_claude_background(self, **kwargs):
        """Run Claude in background mode"""
        try:
            return _run_claude_background(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to run Claude background: {e}")
            raise CodegenAdapterError(f"Failed to run Claude background: {e}")
    
    def create_claude_session(self, **kwargs):
        """Create a new Claude session"""
        try:
            return create_claude_session(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create Claude session: {e}")
            raise CodegenAdapterError(f"Failed to create Claude session: {e}")
    
    def ensure_claude_hook(self, **kwargs):
        """Ensure Claude hook is properly set up"""
        try:
            return ensure_claude_hook(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to ensure Claude hook: {e}")
            raise CodegenAdapterError(f"Failed to ensure Claude hook: {e}")
    
    def cleanup_claude_hook(self, **kwargs):
        """Clean up Claude hook"""
        try:
            return cleanup_claude_hook(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to cleanup Claude hook: {e}")
            raise CodegenAdapterError(f"Failed to cleanup Claude hook: {e}")
    
    def get_codegen_url(self):
        """Get codegen URL for Claude integration"""
        try:
            return get_codegen_url()
        except Exception as e:
            self.logger.error(f"Failed to get codegen URL: {e}")
            raise CodegenAdapterError(f"Failed to get codegen URL: {e}")
    
    def create_log_watcher(self, **kwargs):
        """Create a Claude log watcher"""
        try:
            return ClaudeLogWatcher(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create log watcher: {e}")
            raise CodegenAdapterError(f"Failed to create log watcher: {e}")
    
    def create_log_watcher_manager(self, **kwargs):
        """Create a Claude log watcher manager"""
        try:
            return ClaudeLogWatcherManager(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create log watcher manager: {e}")
            raise CodegenAdapterError(f"Failed to create log watcher manager: {e}")


class MCPManager(BaseManager):
    """
    Model Context Protocol (MCP) server integration.
    
    Features:
    - Dynamic and static tool registration
    - Tool execution via API
    - Resource management
    - Server runner functionality
    """
    
    def register_dynamic_tools(self, **kwargs):
        """Register dynamic tools from API"""
        try:
            return register_dynamic_tools(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to register dynamic tools: {e}")
            raise CodegenAdapterError(f"Failed to register dynamic tools: {e}")
    
    def register_static_tools(self, **kwargs):
        """Register static tools"""
        try:
            return register_static_tools(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to register static tools: {e}")
            raise CodegenAdapterError(f"Failed to register static tools: {e}")
    
    def execute_tool_via_api(self, tool_name: str, **kwargs):
        """Execute a tool via the API"""
        try:
            return execute_tool_via_api(tool_name, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to execute tool via API: {e}")
            raise CodegenAdapterError(f"Failed to execute tool via API: {e}")
    
    def register_resources(self, **kwargs):
        """Register MCP resources"""
        try:
            return register_resources(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to register resources: {e}")
            raise CodegenAdapterError(f"Failed to register resources: {e}")
    
    def run_server(self, **kwargs):
        """Run MCP server"""
        try:
            return run_server(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to run MCP server: {e}")
            raise CodegenAdapterError(f"Failed to run MCP server: {e}")


class AuthManager(BaseManager):
    """
    Authentication and user management.
    
    Features:
    - User login/logout
    - Profile management
    - Organization management
    - Token handling
    """
    
    async def get_current_user(self) -> Dict[str, Any]:
        """Get current user info from API token"""
        try:
            response = await self.api_client.get("/v1/users/me")
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get current user: {e}")
            raise APIError(f"Failed to get current user: {e}")
    
    async def get_organizations(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """Get organizations for authenticated user"""
        try:
            params = {"skip": skip, "limit": limit}
            response = await self.api_client.get("/v1/organizations", params=params)
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get organizations: {e}")
            raise APIError(f"Failed to get organizations: {e}")
    
    async def get_organization_users(
        self,
        org_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get users for specified organization"""
        try:
            params = {"skip": skip, "limit": limit}
            response = await self.api_client.get(
                f"/v1/organizations/{org_id}/users",
                params=params
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get organization users: {e}")
            raise APIError(f"Failed to get organization users: {e}")
    
    async def get_user(self, org_id: int, user_id: int) -> Dict[str, Any]:
        """Get specific user details in organization"""
        try:
            response = await self.api_client.get(
                f"/v1/organizations/{org_id}/users/{user_id}"
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get user: {e}")
            raise APIError(f"Failed to get user: {e}")
    
    def login_cli(self, **kwargs):
        """Login using CLI functions"""
        try:
            return login(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to login via CLI: {e}")
            raise AuthenticationError(f"Failed to login via CLI: {e}")
    
    def logout_cli(self, **kwargs):
        """Logout using CLI functions"""
        try:
            return logout(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to logout via CLI: {e}")
            raise AuthenticationError(f"Failed to logout via CLI: {e}")
    
    def profile_main(self, **kwargs):
        """Access profile management"""
        try:
            return profile_main(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to access profile: {e}")
            raise CodegenAdapterError(f"Failed to access profile: {e}")


class UIManager(BaseManager):
    """
    User Interface management combining TUI components with API data.
    
    Features:
    - Agent list display and management
    - Interactive agent creation
    - Agent detail views
    - Dashboard and kanban interfaces
    """
    
    def create_minimal_tui(self, **kwargs):
        """Create MinimalTUI instance"""
        try:
            return MinimalTUI(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create minimal TUI: {e}")
            raise CodegenAdapterError(f"Failed to create minimal TUI: {e}")
    
    def create_agent_detail_tui(self, **kwargs):
        """Create AgentDetailTUI instance"""
        try:
            return AgentDetailTUI(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create agent detail TUI: {e}")
            raise CodegenAdapterError(f"Failed to create agent detail TUI: {e}")


class TelemetryManager(BaseManager):
    """
    Telemetry and monitoring management.
    
    Features:
    - Session analysis
    - Span tree printing
    - Claude info retrieval
    - OTLP logs endpoint management
    """
    
    def print_span_tree(self, *args, **kwargs):
        """Print spans as a tree structure"""
        try:
            return print_span_tree(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to print span tree: {e}")
            raise CodegenAdapterError(f"Failed to print span tree: {e}")
    
    def load_session(self, session_file: str):
        """Load all records from a session file"""
        try:
            return load_session(session_file)
        except Exception as e:
            self.logger.error(f"Failed to load session: {e}")
            raise CodegenAdapterError(f"Failed to load session: {e}")
    
    def analyze_session(self, session_file: str):
        """Analyze a telemetry session file"""
        try:
            return analyze_session(session_file)
        except Exception as e:
            self.logger.error(f"Failed to analyze session: {e}")
            raise CodegenAdapterError(f"Failed to analyze session: {e}")


class SetupManager(BaseManager):
    """
    Setup and sandbox management.
    
    Features:
    - Setup command generation
    - Sandbox log analysis
    """
    
    async def generate_setup_commands(
        self,
        org_id: int,
        repo_id: int,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate setup commands for a repository"""
        try:
            payload = {
                "repo_id": repo_id,
                "prompt": prompt,
                "trigger_source": "setup-commands"
            }
            payload = {k: v for k, v in payload.items() if v is not None}
            
            response = await self.api_client.post(
                f"/v1/organizations/{org_id}/setup-commands/generate",
                json=payload
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to generate setup commands: {e}")
            raise APIError(f"Failed to generate setup commands: {e}")
    
    async def analyze_sandbox_logs(self, org_id: int, sandbox_id: int) -> Dict[str, Any]:
        """Analyze sandbox setup logs using AI agent"""
        try:
            response = await self.api_client.post(
                f"/v1/organizations/{org_id}/sandbox/{sandbox_id}/analyze-logs"
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to analyze sandbox logs: {e}")
            raise APIError(f"Failed to analyze sandbox logs: {e}")


class CodegenAdapter:
    """
    Main adapter class providing unified access to all codegen functionality.
    
    This is the primary interface that combines all manager classes and provides
    a comprehensive API for both programmatic usage and UI project management.
    """
    
    def __init__(self, api_client=None):
        """Initialize the Codegen Adapter with all manager instances"""
        self.api_client = api_client
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize all manager instances
        self.agent = AgentManager(api_client)
        self.integration = IntegrationManager(api_client)
        self.repository = RepositoryManager(api_client)
        self.claude = ClaudeManager(api_client)
        self.mcp = MCPManager(api_client)
        self.auth = AuthManager(api_client)
        self.ui = UIManager(api_client)
        self.telemetry = TelemetryManager(api_client)
        self.setup = SetupManager(api_client)
        
        self.logger.info("Codegen Adapter initialized successfully")
    
    def get_available_features(self) -> List[str]:
        """Get list of available features/managers"""
        return [
            "agent",
            "integration", 
            "repository",
            "claude",
            "mcp",
            "auth",
            "ui",
            "telemetry",
            "setup"
        ]
    
    def health_check(self) -> Dict[str, bool]:
        """Perform health check on all components"""
        health = {}
        
        try:
            # Check API client availability
            health["api_client"] = is_api_client_available() if 'is_api_client_available' in globals() else False
        except:
            health["api_client"] = False
        
        # Check manager initialization
        managers = ["agent", "integration", "repository", "claude", "mcp", "auth", "ui", "telemetry", "setup"]
        for manager in managers:
            try:
                health[manager] = hasattr(self, manager) and getattr(self, manager) is not None
            except:
                health[manager] = False
        
        return health
    
    def get_version_info(self) -> Dict[str, str]:
        """Get version information"""
        return {
            "codegen_adapter": "1.0.0",
            "codegen": "Unknown",  # Would need to import from codegen package
            "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"
        }


# Factory functions for easy instantiation
def create_adapter(api_client=None) -> CodegenAdapter:
    """Factory function to create a CodegenAdapter instance"""
    return CodegenAdapter(api_client)


def create_agent_manager(api_client=None) -> AgentManager:
    """Factory function to create an AgentManager instance"""
    return AgentManager(api_client)


def create_integration_manager(api_client=None) -> IntegrationManager:
    """Factory function to create an IntegrationManager instance"""
    return IntegrationManager(api_client)


# Example usage and testing
def main():
    """Example usage of the Codegen Adapter"""
    try:
        # Create adapter instance
        adapter = create_adapter()
        
        # Health check
        health = adapter.health_check()
        print("Health Check Results:")
        for component, status in health.items():
            print(f"  {component}: {'✓' if status else '✗'}")
        
        # Version info
        version_info = adapter.get_version_info()
        print(f"\nVersion Info: {version_info}")
        
        # Available features
        features = adapter.get_available_features()
        print(f"Available Features: {', '.join(features)}")
        
        print("\nCodegen Adapter initialized successfully!")
        print("Use adapter.agent, adapter.claude, adapter.repository, etc. to access functionality")
        
    except Exception as e:
        print(f"Error initializing Codegen Adapter: {e}")
        print("Make sure codegen is installed and you're authenticated")


if __name__ == "__main__":
    main()


# Export main classes and functions for easy import
__all__ = [
    "CodegenAdapter",
    "AgentManager",
    "IntegrationManager", 
    "RepositoryManager",
    "ClaudeManager",
    "MCPManager",
    "AuthManager",
    "UIManager",
    "TelemetryManager",
    "SetupManager",
    "create_adapter",
    "create_agent_manager",
    "create_integration_manager",
    "CodegenAdapterError",
    "AuthenticationError", 
    "APIError"
]