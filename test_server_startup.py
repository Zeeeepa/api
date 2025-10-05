#!/usr/bin/env python3
"""
Test Pro Mode server startup and basic API functionality
"""

import os
import sys
import asyncio
import httpx
import json
import time
import subprocess
import signal
from typing import Optional

# Add current directory to Python path
sys.path.insert(0, '/tmp')

# Set environment variables
os.environ.update({
    "CODEGEN_ORG_ID": "323",
    "CODEGEN_API_TOKEN": os.getenv("CODEGEN_API_TOKEN", "test-token"),
    "PRO_MODE_MAX_AGENTS": "10",
    "PRO_MODE_LOG_LEVEL": "INFO",
    "PRO_MODE_DEFAULT_TIMEOUT": "60"
})

class ProModeServerTest:
    def __init__(self, port: int = 8001):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.server_process: Optional[subprocess.Popen] = None
        self.client: Optional[httpx.AsyncClient] = None
    
    def start_server(self) -> bool:
        """Start the Pro Mode server"""
        try:
            print(f"🚀 Starting Pro Mode server on port {self.port}...")
            
            # Start server process
            self.server_process = subprocess.Popen([
                sys.executable, "codegen_pro_mode.py", 
                "--host", "127.0.0.1", 
                "--port", str(self.port)
            ], 
            cwd="/tmp",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Wait for server to start
            for i in range(15):  # 15 second timeout
                try:
                    response = httpx.get(f"{self.base_url}/health", timeout=1.0)
                    if response.status_code == 200:
                        print(f"✅ Server started successfully in {i+1} seconds")
                        return True
                except:
                    time.sleep(1)
            
            print("❌ Server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the Pro Mode server"""
        if self.server_process:
            try:
                if os.name == 'nt':
                    self.server_process.terminate()
                else:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                
                self.server_process.wait(timeout=5)
                print("✅ Server stopped successfully")
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    self.server_process.kill()
                else:
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                print("⚠️  Server force killed")
            except Exception as e:
                print(f"⚠️  Error stopping server: {e}")
    
    async def test_health_endpoint(self) -> bool:
        """Test health check endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5.0)
                
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"✅ Health check passed: {health_data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_stats_endpoint(self) -> bool:
        """Test stats endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/stats", timeout=5.0)
                
                if response.status_code == 200:
                    stats_data = response.json()
                    print(f"✅ Stats endpoint working: {len(stats_data)} metrics")
                    return True
                else:
                    print(f"❌ Stats endpoint failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ Stats endpoint error: {e}")
            return False
    
    async def test_sessions_endpoint(self) -> bool:
        """Test sessions listing endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/sessions", timeout=5.0)
                
                if response.status_code == 200:
                    sessions_data = response.json()
                    print(f"✅ Sessions endpoint working: {len(sessions_data)} sessions")
                    return True
                else:
                    print(f"❌ Sessions endpoint failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ Sessions endpoint error: {e}")
            return False
    
    async def test_pro_mode_request_creation(self) -> bool:
        """Test creating a Pro Mode request (without actual execution)"""
        try:
            async with httpx.AsyncClient() as client:
                # Create a simple test request
                request_data = {
                    "prompt": "Test prompt for API endpoint validation",
                    "num_agents": 2,
                    "org_id": 323,
                    "strategy": "simple",
                    "timeout": 60,
                    "metadata": {
                        "test": True,
                        "api_test": "endpoint_validation"
                    }
                }
                
                response = await client.post(
                    f"{self.base_url}/pro-mode", 
                    json=request_data,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    session_id = response_data.get("session_id")
                    print(f"✅ Pro Mode request created: {session_id[:8] if session_id else 'no-id'}...")
                    
                    # Check session status
                    if session_id:
                        await asyncio.sleep(1)  # Brief wait
                        status_response = await client.get(
                            f"{self.base_url}/sessions/{session_id}",
                            timeout=5.0
                        )
                        
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            print(f"✅ Session status check passed: {status_data.get('status', 'unknown')}")
                        else:
                            print(f"⚠️  Session status check failed: {status_response.status_code}")
                    
                    return True
                else:
                    print(f"❌ Pro Mode request failed: {response.status_code}")
                    print(f"    Error details: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Pro Mode request error: {e}")
            return False
    
    async def test_api_endpoints(self) -> dict:
        """Test all API endpoints"""
        print("\n🧪 Testing API Endpoints")
        print("=" * 40)
        
        results = {}
        
        # Test health endpoint
        results["health"] = await self.test_health_endpoint()
        
        # Test stats endpoint
        results["stats"] = await self.test_stats_endpoint()
        
        # Test sessions endpoint
        results["sessions"] = await self.test_sessions_endpoint()
        
        # Test Pro Mode request creation
        results["pro_mode_request"] = await self.test_pro_mode_request_creation()
        
        return results
    
    async def run_complete_test(self) -> bool:
        """Run complete server test"""
        print("🎯 Pro Mode Server Integration Test")
        print("=" * 50)
        
        # Start server
        if not self.start_server():
            return False
        
        try:
            # Wait a moment for server to fully initialize
            await asyncio.sleep(2)
            
            # Test API endpoints
            endpoint_results = await self.test_api_endpoints()
            
            # Summary
            print(f"\n📊 Test Results Summary")
            print("=" * 30)
            
            passed = sum(1 for result in endpoint_results.values() if result)
            total = len(endpoint_results)
            
            for test_name, result in endpoint_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{status} {test_name.replace('_', ' ').title()}")
            
            success_rate = (passed / total) * 100 if total > 0 else 0
            print(f"\nOverall: {passed}/{total} ({success_rate:.1f}%)")
            
            if success_rate >= 80:
                print("🎉 Server is working well!")
            elif success_rate >= 60:
                print("⚠️  Server has some issues but core functionality works")
            else:
                print("❌ Server has significant issues")
            
            return success_rate >= 60
            
        finally:
            # Always stop server
            self.stop_server()

async def main():
    """Main test function"""
    tester = ProModeServerTest(port=8001)
    
    try:
        success = await tester.run_complete_test()
        
        print(f"\n🏁 Integration Test {'PASSED' if success else 'FAILED'}")
        
        if success:
            print("\n🚀 Pro Mode server is ready for use!")
            print("Next steps:")
            print("  1. Run: python codegen_pro_mode.py")
            print("  2. Visit: http://localhost:8000/docs")
            print("  3. Test with: python test_pro_mode_focused.py")
        else:
            print("\n⚠️  Some issues found. Check logs above for details.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        tester.stop_server()
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        tester.stop_server()

if __name__ == "__main__":
    asyncio.run(main())