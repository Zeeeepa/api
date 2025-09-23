#!/usr/bin/env python3
"""
Focused test for Codegen Pro Mode core functionality
Tests individual components without requiring full Codegen installation
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any

# Add current directory to Python path
sys.path.insert(0, '/tmp')

# Set environment variables for testing
os.environ.update({
    "CODEGEN_ORG_ID": "323",
    "CODEGEN_API_TOKEN": os.getenv("CODEGEN_API_TOKEN", "test-token"),
    "PRO_MODE_MAX_AGENTS": "5",
    "PRO_MODE_LOG_LEVEL": "INFO",
    "PRO_MODE_DEFAULT_TIMEOUT": "60"
})

def print_test(name: str, success: bool, message: str = ""):
    """Print test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {name}")
    if message:
        print(f"    {message}")

async def test_imports():
    """Test that all imports work"""
    try:
        # Test Codegen Adapter import (may have warnings but should work)
        from Codegen_adapter import create_adapter, CodegenAdapterError, APIError
        
        # Test Pro Mode imports
        from codegen_pro_mode import (
            StateManager, ProModeEngine, ProModeConfig,
            ProModeRequest, ProModeStrategy, AgentStatus
        )
        
        print_test("Imports", True, "All imports successful")
        return True
    except Exception as e:
        print_test("Imports", False, f"Import error: {e}")
        return False

async def test_configuration():
    """Test configuration loading"""
    try:
        from codegen_pro_mode import ProModeConfig
        
        config = ProModeConfig()
        
        # Check that environment variables are loaded
        assert config.max_agents == 5, f"Expected 5 agents, got {config.max_agents}"
        assert config.log_level == "INFO", f"Expected INFO log level, got {config.log_level}"
        assert config.default_timeout == 60, f"Expected 60s timeout, got {config.default_timeout}"
        
        print_test("Configuration", True, f"Max agents: {config.max_agents}, Log: {config.log_level}")
        return True
    except Exception as e:
        print_test("Configuration", False, f"Config error: {e}")
        return False

async def test_state_manager():
    """Test state manager functionality"""
    try:
        from codegen_pro_mode import StateManager, ProModeRequest, ProModeStrategy
        
        # Create state manager
        state_manager = StateManager()
        
        # Test stats
        stats = state_manager.get_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        assert "adapters" in stats, "Stats should include adapters count"
        
        # Test session creation
        request = ProModeRequest(
            prompt="Test prompt",
            num_agents=3,
            org_id=323,
            strategy=ProModeStrategy.SIMPLE,
            timeout=60
        )
        
        session = state_manager.create_session(request)
        assert session.session_id is not None, "Session should have an ID"
        assert session.num_agents == 3, "Session should have correct agent count"
        
        print_test("State Manager", True, f"Session created: {session.session_id[:8]}...")
        return True
    except Exception as e:
        print_test("State Manager", False, f"State manager error: {e}")
        return False

async def test_agent_creation():
    """Test agent instance creation"""
    try:
        from codegen_pro_mode import (
            StateManager, ProModeRequest, ProModeStrategy, AgentStatus
        )
        
        state_manager = StateManager()
        
        # Create session
        request = ProModeRequest(
            prompt="Test agent creation",
            num_agents=2,
            org_id=323,
            strategy=ProModeStrategy.SIMPLE
        )
        
        session = state_manager.create_session(request)
        
        # Create agent instances
        agents = []
        for i in range(2):
            agent = state_manager.create_agent_instance(session, i)
            agents.append(agent)
            
            # Test agent properties
            assert agent.agent_id is not None, "Agent should have an ID"
            assert agent.status == AgentStatus.PENDING, "New agent should be pending"
            assert agent.org_id == 323, "Agent should have correct org_id"
        
        # Test status update
        agent = agents[0]
        state_manager.update_agent_status(agent.agent_id, AgentStatus.RUNNING)
        assert state_manager.agents[agent.agent_id].status == AgentStatus.RUNNING
        
        # Test log addition
        state_manager.add_agent_log(agent.agent_id, {
            "type": "test",
            "message": "Test log entry"
        })
        
        assert len(state_manager.agents[agent.agent_id].logs) == 1
        
        print_test("Agent Creation", True, f"Created {len(agents)} agents successfully")
        return True
    except Exception as e:
        print_test("Agent Creation", False, f"Agent creation error: {e}")
        return False

async def test_request_validation():
    """Test request validation"""
    try:
        from codegen_pro_mode import ProModeRequest, ProModeStrategy
        
        # Test valid request
        valid_request = ProModeRequest(
            prompt="Valid test prompt",
            num_agents=5,
            org_id=323,
            strategy=ProModeStrategy.ADAPTIVE,
            timeout=120
        )
        
        assert valid_request.prompt == "Valid test prompt"
        assert valid_request.num_agents == 5
        assert valid_request.org_id == 323
        
        # Test invalid request (should raise validation error)
        try:
            invalid_request = ProModeRequest(
                prompt="",  # Empty prompt should fail
                num_agents=0,  # Zero agents should fail
                org_id=323
            )
            print_test("Request Validation", False, "Should have failed validation")
            return False
        except Exception:
            # Expected to fail
            pass
        
        print_test("Request Validation", True, "Validation works correctly")
        return True
    except Exception as e:
        print_test("Request Validation", False, f"Validation error: {e}")
        return False

async def test_fastapi_app_creation():
    """Test FastAPI app creation"""
    try:
        from codegen_pro_mode import app
        
        # Check that app is created
        assert app is not None, "FastAPI app should be created"
        
        # Check routes exist
        route_paths = [route.path for route in app.routes if hasattr(route, 'path')]
        
        expected_routes = ["/pro-mode", "/health", "/stats"]
        for expected_route in expected_routes:
            assert any(expected_route in path for path in route_paths), f"Route {expected_route} not found"
        
        print_test("FastAPI App", True, f"App created with {len(route_paths)} routes")
        return True
    except Exception as e:
        print_test("FastAPI App", False, f"FastAPI error: {e}")
        return False

async def test_mock_synthesis():
    """Test synthesis logic without actual API calls"""
    try:
        from codegen_pro_mode import SynthesisEngine
        
        # Create mock adapter
        class MockAdapter:
            class MockAgent:
                async def create_agent_run(self, **kwargs):
                    return {"id": 123}
            
            def __init__(self):
                self.agent = self.MockAgent()
        
        mock_adapter = MockAdapter()
        synthesis_engine = SynthesisEngine(mock_adapter)
        
        # Test prompt building
        candidates = [
            "First analysis result",
            "Second analysis result", 
            "Third analysis result"
        ]
        
        prompt = synthesis_engine._build_synthesis_prompt(candidates, "Original prompt")
        
        assert "candidate 1" in prompt.lower()
        assert "candidate 2" in prompt.lower()
        assert "candidate 3" in prompt.lower()
        assert "original prompt" in prompt.lower()
        
        print_test("Mock Synthesis", True, f"Synthesis prompt built for {len(candidates)} candidates")
        return True
    except Exception as e:
        print_test("Mock Synthesis", False, f"Synthesis error: {e}")
        return False

async def run_focused_tests():
    """Run focused test suite"""
    print("🧪 Codegen Pro Mode Focused Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("State Manager", test_state_manager),
        ("Agent Creation", test_agent_creation),
        ("Request Validation", test_request_validation),
        ("FastAPI App", test_fastapi_app_creation),
        ("Mock Synthesis", test_mock_synthesis)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print_test(test_name, False, f"Unexpected error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 Core functionality is working well!")
    elif success_rate >= 60:
        print("⚠️  Core functionality mostly works with some issues")
    else:
        print("❌ Significant issues found")
    
    # Show configuration info
    print(f"\n📋 Configuration Info:")
    print(f"  Organization ID: {os.getenv('CODEGEN_ORG_ID')}")
    print(f"  API Token: {'Set' if os.getenv('CODEGEN_API_TOKEN') else 'Not set'}")
    print(f"  Max Agents: {os.getenv('PRO_MODE_MAX_AGENTS')}")
    print(f"  Log Level: {os.getenv('PRO_MODE_LOG_LEVEL')}")
    
    return results, success_rate

if __name__ == "__main__":
    asyncio.run(run_focused_tests())