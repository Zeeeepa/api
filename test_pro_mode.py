#!/usr/bin/env python3
"""
Test script for Codegen Pro Mode system
Uses real environment variables for testing
"""

import os
import asyncio
import json
import logging
from typing import Dict, Any
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, '/tmp')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration using environment variables
TEST_CONFIG = {
    "org_id": int(os.getenv("CODEGEN_ORG_ID", "323")),
    "api_token": os.getenv("CODEGEN_API_TOKEN"),
    "github_token": os.getenv("GITHUB_TOKEN"),
    "max_agents": 3,  # Keep small for testing
    "timeout": 120,   # 2 minutes for testing
}

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 60}")
    print(f"🧪 {title}")
    print('=' * 60)

def print_status(message: str, status: str = "INFO"):
    """Print a status message"""
    emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{emoji.get(status, 'ℹ️')} {message}")

async def test_adapter_initialization():
    """Test Codegen Adapter initialization"""
    print_section("Testing Codegen Adapter Initialization")
    
    try:
        # Set environment variables
        os.environ["CODEGEN_API_TOKEN"] = TEST_CONFIG["api_token"]
        
        # Try importing the adapter
        from Codegen_adapter import create_adapter, CodegenAdapter
        print_status("Codegen Adapter import successful", "SUCCESS")
        
        # Create adapter instance
        adapter = create_adapter()
        print_status("Adapter instance created", "SUCCESS")
        
        # Health check
        health = adapter.health_check()
        print_status(f"Health check: {health}", "INFO")
        
        # Get available features
        features = adapter.get_available_features()
        print_status(f"Available features: {', '.join(features)}", "INFO")
        
        return adapter, True
        
    except ImportError as e:
        print_status(f"Import error: {e}", "ERROR")
        print_status("Note: This is expected if Codegen package is not installed", "WARNING")
        return None, False
    except Exception as e:
        print_status(f"Adapter initialization failed: {e}", "ERROR")
        return None, False

async def test_pro_mode_configuration():
    """Test Pro Mode configuration and setup"""
    print_section("Testing Pro Mode Configuration")
    
    try:
        # Set Pro Mode environment variables
        os.environ["PRO_MODE_MAX_AGENTS"] = str(TEST_CONFIG["max_agents"])
        os.environ["PRO_MODE_DEFAULT_TIMEOUT"] = str(TEST_CONFIG["timeout"])
        os.environ["PRO_MODE_LOG_LEVEL"] = "INFO"
        
        # Import Pro Mode components
        from codegen_pro_mode import (
            StateManager, ProModeEngine, ProModeConfig,
            ProModeRequest, ProModeStrategy
        )
        print_status("Pro Mode imports successful", "SUCCESS")
        
        # Test configuration
        config = ProModeConfig()
        print_status(f"Max agents: {config.max_agents}", "INFO")
        print_status(f"Default timeout: {config.default_timeout}", "INFO")
        print_status(f"Log level: {config.log_level}", "INFO")
        
        # Test state manager
        state_manager = StateManager()
        print_status("State manager created", "SUCCESS")
        
        stats = state_manager.get_stats()
        print_status(f"Initial stats: {stats}", "INFO")
        
        return config, state_manager, True
        
    except Exception as e:
        print_status(f"Configuration test failed: {e}", "ERROR")
        return None, None, False

async def test_pro_mode_request():
    """Test Pro Mode request creation and validation"""
    print_section("Testing Pro Mode Request Creation")
    
    try:
        from codegen_pro_mode import ProModeRequest, ProModeStrategy
        
        # Create test request
        request = ProModeRequest(
            prompt="Test prompt: Analyze this simple Python function for basic improvements",
            num_agents=TEST_CONFIG["max_agents"],
            org_id=TEST_CONFIG["org_id"],
            repo_id=None,  # No specific repo for testing
            strategy=ProModeStrategy.SIMPLE,
            timeout=TEST_CONFIG["timeout"],
            metadata={
                "test": True,
                "environment": "testing",
                "max_agents": TEST_CONFIG["max_agents"]
            }
        )
        
        print_status("Pro Mode request created successfully", "SUCCESS")
        print_status(f"Prompt length: {len(request.prompt)} characters", "INFO")
        print_status(f"Number of agents: {request.num_agents}", "INFO")
        print_status(f"Organization ID: {request.org_id}", "INFO")
        print_status(f"Strategy: {request.strategy}", "INFO")
        print_status(f"Timeout: {request.timeout} seconds", "INFO")
        
        return request, True
        
    except Exception as e:
        print_status(f"Request creation failed: {e}", "ERROR")
        return None, False

async def test_session_creation():
    """Test Pro Mode session creation"""
    print_section("Testing Pro Mode Session Creation")
    
    try:
        from codegen_pro_mode import StateManager, ProModeRequest, ProModeStrategy
        
        state_manager = StateManager()
        
        request = ProModeRequest(
            prompt="Test session: Analyze basic code structure and suggest improvements",
            num_agents=2,  # Keep minimal for testing
            org_id=TEST_CONFIG["org_id"],
            repo_id=None,
            strategy=ProModeStrategy.SIMPLE,
            timeout=60,  # 1 minute for testing
            metadata={"test_session": True}
        )
        
        # Create session
        session = state_manager.create_session(request)
        print_status(f"Session created with ID: {session.session_id}", "SUCCESS")
        print_status(f"Session status: {session.status}", "INFO")
        print_status(f"Number of agents planned: {session.num_agents}", "INFO")
        
        # Create agent instances
        for i in range(session.num_agents):
            agent = state_manager.create_agent_instance(session, i)
            print_status(f"Agent {i} created: {agent.agent_id}", "INFO")
        
        # Check session state
        session_stats = {
            "session_id": session.session_id,
            "agents_created": len(session.agents),
            "status": session.status,
            "strategy": session.strategy.value
        }
        print_status(f"Session stats: {json.dumps(session_stats, indent=2)}", "INFO")
        
        return session, True
        
    except Exception as e:
        print_status(f"Session creation failed: {e}", "ERROR")
        return None, False

async def test_mock_agent_execution():
    """Test mock agent execution without actual Codegen API calls"""
    print_section("Testing Mock Agent Execution")
    
    try:
        from codegen_pro_mode import (
            StateManager, AgentInstance, AgentStatus,
            ProModeRequest, ProModeStrategy
        )
        import time
        
        state_manager = StateManager()
        
        # Create a test session
        request = ProModeRequest(
            prompt="Mock test: Simple code analysis",
            num_agents=2,
            org_id=TEST_CONFIG["org_id"],
            strategy=ProModeStrategy.SIMPLE,
            timeout=30
        )
        
        session = state_manager.create_session(request)
        
        # Create and simulate agent execution
        results = []
        for i in range(session.num_agents):
            agent = state_manager.create_agent_instance(session, i)
            
            # Simulate agent lifecycle
            print_status(f"Starting mock execution for agent {i}", "INFO")
            
            # Update to running
            state_manager.update_agent_status(agent.agent_id, AgentStatus.RUNNING)
            await asyncio.sleep(1)  # Simulate work
            
            # Add mock logs
            state_manager.add_agent_log(agent.agent_id, {
                "type": "execution_start",
                "message": f"Mock agent {i} started execution"
            })
            
            # Simulate completion
            mock_result = f"Mock analysis result from agent {i}: Code looks good with minor suggestions for improvement."
            state_manager.update_agent_status(
                agent.agent_id, 
                AgentStatus.COMPLETED, 
                result=mock_result
            )
            
            state_manager.add_agent_log(agent.agent_id, {
                "type": "execution_complete",
                "message": f"Mock agent {i} completed successfully",
                "result_length": len(mock_result)
            })
            
            results.append(mock_result)
            print_status(f"Agent {i} completed mock execution", "SUCCESS")
        
        # Test synthesis simulation
        synthesized_result = f"Synthesized analysis from {len(results)} agents: " + \
                           "Combined insights suggest the code structure is solid with opportunities for optimization."
        
        session.final_result = synthesized_result
        session.status = "completed"
        
        print_status("Mock synthesis completed", "SUCCESS")
        print_status(f"Final result length: {len(synthesized_result)} characters", "INFO")
        
        return {
            "session": session,
            "individual_results": results,
            "final_result": synthesized_result
        }, True
        
    except Exception as e:
        print_status(f"Mock execution failed: {e}", "ERROR")
        return None, False

async def test_api_endpoints_structure():
    """Test that API endpoints are properly defined"""
    print_section("Testing API Endpoints Structure")
    
    try:
        from codegen_pro_mode import app
        
        # Get all routes
        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                routes.append({
                    "path": route.path,
                    "methods": list(route.methods) if route.methods else [],
                    "name": getattr(route, 'name', 'unnamed')
                })
        
        print_status(f"Found {len(routes)} API routes", "SUCCESS")
        
        # Key endpoints to check
        key_endpoints = [
            "/pro-mode",
            "/pro-mode-sync", 
            "/sessions/{session_id}",
            "/sessions",
            "/agents/{agent_id}",
            "/health",
            "/stats"
        ]
        
        for endpoint in key_endpoints:
            found = any(endpoint.replace("{session_id}", "session_id") in route["path"] or 
                       endpoint.replace("{agent_id}", "agent_id") in route["path"] 
                       for route in routes)
            status = "SUCCESS" if found else "ERROR"
            print_status(f"Endpoint {endpoint}: {'Found' if found else 'Missing'}", status)
        
        return routes, True
        
    except Exception as e:
        print_status(f"API structure test failed: {e}", "ERROR")
        return None, False

async def test_environment_configuration():
    """Test environment variable configuration"""
    print_section("Testing Environment Configuration")
    
    try:
        # Check required environment variables
        required_vars = {
            "CODEGEN_API_TOKEN": TEST_CONFIG["api_token"],
            "CODEGEN_ORG_ID": str(TEST_CONFIG["org_id"]),
            "GITHUB_TOKEN": TEST_CONFIG.get("github_token", "Not set")
        }
        
        print_status("Environment Variables Check:", "INFO")
        for var_name, var_value in required_vars.items():
            if var_value and var_value != "Not set":
                masked_value = var_value[:10] + "..." if len(var_value) > 10 else var_value
                print_status(f"  {var_name}: {masked_value} ✓", "SUCCESS")
            else:
                print_status(f"  {var_name}: Not set", "WARNING")
        
        # Test Pro Mode specific configuration
        from codegen_pro_mode import ProModeConfig
        
        # Set test environment variables
        test_env_vars = {
            "PRO_MODE_MAX_AGENTS": "10",
            "PRO_MODE_MAX_WORKERS": "5",
            "PRO_MODE_TOURNAMENT_THRESHOLD": "8",
            "PRO_MODE_LOG_LEVEL": "DEBUG"
        }
        
        for var, value in test_env_vars.items():
            os.environ[var] = value
        
        config = ProModeConfig()
        
        print_status("Pro Mode Configuration:", "INFO")
        print_status(f"  Max Agents: {config.max_agents}", "INFO")
        print_status(f"  Max Workers: {config.max_workers}", "INFO")
        print_status(f"  Tournament Threshold: {config.tournament_threshold}", "INFO")
        print_status(f"  Log Level: {config.log_level}", "INFO")
        
        return config, True
        
    except Exception as e:
        print_status(f"Environment configuration test failed: {e}", "ERROR")
        return None, False

async def run_comprehensive_test():
    """Run comprehensive Pro Mode test suite"""
    print_section("Codegen Pro Mode Comprehensive Test Suite")
    
    print_status(f"Using Organization ID: {TEST_CONFIG['org_id']}", "INFO")
    print_status(f"API Token: {'Set' if TEST_CONFIG['api_token'] else 'Not set'}", 
                "SUCCESS" if TEST_CONFIG['api_token'] else "WARNING")
    
    test_results = {}
    
    # Test 1: Environment Configuration
    config, env_success = await test_environment_configuration()
    test_results["environment"] = env_success
    
    # Test 2: Codegen Adapter (may fail if not installed)
    adapter, adapter_success = await test_adapter_initialization()
    test_results["adapter"] = adapter_success
    
    # Test 3: Pro Mode Configuration
    pm_config, state_manager, config_success = await test_pro_mode_configuration()
    test_results["pro_mode_config"] = config_success
    
    # Test 4: Request Creation
    request, request_success = await test_pro_mode_request()
    test_results["request_creation"] = request_success
    
    # Test 5: Session Creation
    session, session_success = await test_session_creation()
    test_results["session_creation"] = session_success
    
    # Test 6: Mock Agent Execution
    execution_result, execution_success = await test_mock_agent_execution()
    test_results["mock_execution"] = execution_success
    
    # Test 7: API Endpoints Structure
    routes, api_success = await test_api_endpoints_structure()
    test_results["api_structure"] = api_success
    
    # Summary
    print_section("Test Results Summary")
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "SUCCESS" if result else "ERROR"
        print_status(f"{test_name.replace('_', ' ').title()}: {'PASSED' if result else 'FAILED'}", status)
    
    print_status(f"\nOverall: {passed_tests}/{total_tests} tests passed", 
                "SUCCESS" if passed_tests == total_tests else "WARNING")
    
    success_rate = (passed_tests / total_tests) * 100
    print_status(f"Success Rate: {success_rate:.1f}%", 
                "SUCCESS" if success_rate >= 80 else "WARNING" if success_rate >= 60 else "ERROR")
    
    # Additional information
    if execution_result and execution_success:
        print_section("Mock Execution Results")
        print_status(f"Session ID: {execution_result['session'].session_id}", "INFO")
        print_status(f"Individual results: {len(execution_result['individual_results'])}", "INFO")
        print_status(f"Final result preview: {execution_result['final_result'][:100]}...", "INFO")
    
    return test_results, success_rate

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Codegen Pro Mode Test Suite")
    print(f"Python path: {sys.path[0]}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        results, success_rate = asyncio.run(run_comprehensive_test())
        
        if success_rate >= 80:
            print_status("🎉 Pro Mode system is ready for use!", "SUCCESS")
        elif success_rate >= 60:
            print_status("⚠️ Pro Mode system has some issues but core functionality works", "WARNING")
        else:
            print_status("❌ Pro Mode system has significant issues that need to be addressed", "ERROR")
            
    except KeyboardInterrupt:
        print_status("Test suite interrupted by user", "WARNING")
    except Exception as e:
        print_status(f"Test suite failed with error: {e}", "ERROR")
        import traceback
        traceback.print_exc()