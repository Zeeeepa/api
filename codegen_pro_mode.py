#!/usr/bin/env python3
"""
Codegen Pro Mode - Advanced Agent Orchestration System

This module provides a FastAPI-based Pro Mode service for Codegen that enables:
- Parallel agent execution with tournament-style synthesis
- State management for Codegen and Claude instances
- Advanced prompt engineering and multi-candidate generation
- Comprehensive monitoring and error handling

Based on the Pro Mode pattern but fully adapted for Codegen's agentic architecture.

Author: Claude Code Integration
Version: 1.0.0
"""

import os
import time
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json
import concurrent.futures as cf

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import the Codegen Adapter
from Codegen_adapter import (
    create_adapter,
    CodegenAdapter,
    CodegenAdapterError,
    AuthenticationError,
    APIError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_AGENTS = 100
MAX_WORKERS = 50
TOURNAMENT_THRESHOLD = 20
GROUP_SIZE = 10
DEFAULT_TIMEOUT = 300  # 5 minutes
STATE_CLEANUP_INTERVAL = 3600  # 1 hour
MAX_RETRIES = 3

# FastAPI app
app = FastAPI(
    title="Codegen Pro Mode - Advanced Agent Orchestration",
    description="Professional-grade agent management with parallel execution and tournament synthesis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
adapter_instances: Dict[str, CodegenAdapter] = {}
agent_states: Dict[str, Dict[str, Any]] = {}
claude_instances: Dict[str, Dict[str, Any]] = {}
session_data: Dict[str, Dict[str, Any]] = {}


# ---------- Enums and Data Classes ----------

class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SYNTHESIZING = "synthesizing"


class ProModeStrategy(str, Enum):
    SIMPLE = "simple"
    TOURNAMENT = "tournament"
    ADAPTIVE = "adaptive"


@dataclass
class AgentInstance:
    """Represents a single agent instance with state tracking"""
    agent_id: str
    org_id: int
    prompt: str
    repo_id: Optional[int] = None
    status: AgentStatus = AgentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_terminal(self) -> bool:
        return self.status in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED]


@dataclass 
class ClaudeInstance:
    """Represents a Claude Code instance with state tracking"""
    instance_id: str
    session_name: str
    project_path: Optional[str] = None
    status: str = "inactive"
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    log_watcher: Optional[Any] = None


@dataclass
class ProModeSession:
    """Represents a Pro Mode execution session"""
    session_id: str
    prompt: str
    num_agents: int
    strategy: ProModeStrategy
    org_id: int
    repo_id: Optional[int] = None
    status: str = "initializing"
    created_at: datetime = field(default_factory=datetime.now)
    agents: List[AgentInstance] = field(default_factory=list)
    synthesis_results: List[str] = field(default_factory=list)
    final_result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------- Pydantic Models ----------

class ProModeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The prompt for agent execution")
    num_agents: int = Field(..., ge=1, le=MAX_AGENTS, description="Number of agents to spawn")
    org_id: int = Field(..., description="Organization ID")
    repo_id: Optional[int] = Field(None, description="Repository ID (optional)")
    strategy: ProModeStrategy = Field(ProModeStrategy.ADAPTIVE, description="Execution strategy")
    timeout: int = Field(DEFAULT_TIMEOUT, ge=30, le=1800, description="Timeout in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ProModeResponse(BaseModel):
    session_id: str
    status: str
    final_result: Optional[str] = None
    candidates: List[str] = []
    synthesis_path: List[str] = []
    stats: Dict[str, Any] = {}
    created_at: datetime
    completed_at: Optional[datetime] = None


class AgentStatusResponse(BaseModel):
    agent_id: str
    status: AgentStatus
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    duration: Optional[float] = None
    logs_count: int = 0


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    prompt: str
    num_agents: int
    strategy: ProModeStrategy
    agents: List[AgentStatusResponse]
    progress: float
    final_result: Optional[str] = None
    stats: Dict[str, Any] = {}


# ---------- State Management ----------

class StateManager:
    """Manages state for Codegen and Claude instances"""
    
    def __init__(self):
        self.adapters: Dict[str, CodegenAdapter] = {}
        self.agents: Dict[str, AgentInstance] = {}
        self.claude_instances: Dict[str, ClaudeInstance] = {}
        self.sessions: Dict[str, ProModeSession] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        
    def get_adapter(self, adapter_id: str = "default") -> CodegenAdapter:
        """Get or create a CodegenAdapter instance"""
        if adapter_id not in self.adapters:
            try:
                self.adapters[adapter_id] = create_adapter()
                logger.info(f"Created new adapter instance: {adapter_id}")
            except Exception as e:
                logger.error(f"Failed to create adapter {adapter_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to create adapter: {e}")
        
        return self.adapters[adapter_id]
    
    def create_session(self, request: ProModeRequest) -> ProModeSession:
        """Create a new Pro Mode session"""
        session_id = str(uuid.uuid4())
        session = ProModeSession(
            session_id=session_id,
            prompt=request.prompt,
            num_agents=request.num_agents,
            strategy=request.strategy,
            org_id=request.org_id,
            repo_id=request.repo_id,
            metadata=request.metadata or {}
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created Pro Mode session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> ProModeSession:
        """Get a session by ID"""
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return self.sessions[session_id]
    
    def create_agent_instance(self, session: ProModeSession, agent_index: int) -> AgentInstance:
        """Create a new agent instance"""
        agent_id = f"{session.session_id}-agent-{agent_index}"
        agent = AgentInstance(
            agent_id=agent_id,
            org_id=session.org_id,
            prompt=session.prompt,
            repo_id=session.repo_id,
            metadata={
                "session_id": session.session_id,
                "agent_index": agent_index,
                "strategy": session.strategy.value
            }
        )
        
        self.agents[agent_id] = agent
        session.agents.append(agent)
        logger.debug(f"Created agent instance: {agent_id}")
        return agent
    
    def create_claude_instance(self, session_name: str, project_path: Optional[str] = None) -> ClaudeInstance:
        """Create a new Claude instance"""
        instance_id = str(uuid.uuid4())
        claude = ClaudeInstance(
            instance_id=instance_id,
            session_name=session_name,
            project_path=project_path
        )
        
        self.claude_instances[instance_id] = claude
        logger.info(f"Created Claude instance: {instance_id}")
        return claude
    
    def update_agent_status(self, agent_id: str, status: AgentStatus, **kwargs):
        """Update agent status and metadata"""
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found for status update")
            return
        
        agent = self.agents[agent_id]
        agent.status = status
        
        if status == AgentStatus.RUNNING and not agent.started_at:
            agent.started_at = datetime.now()
        elif status in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED]:
            agent.completed_at = datetime.now()
        
        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
        
        logger.debug(f"Updated agent {agent_id} status to {status}")
    
    def add_agent_log(self, agent_id: str, log_entry: Dict[str, Any]):
        """Add a log entry to an agent"""
        if agent_id in self.agents:
            self.agents[agent_id].logs.append({
                **log_entry,
                "timestamp": datetime.now().isoformat()
            })
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions and their associated data"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if session.created_at < cutoff:
                sessions_to_remove.append(session_id)
                
                # Clean up associated agents
                for agent in session.agents:
                    if agent.agent_id in self.agents:
                        del self.agents[agent.agent_id]
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            logger.info(f"Cleaned up old session: {session_id}")
        
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
    
    async def start_cleanup_task(self):
        """Start the background cleanup task"""
        if self.cleanup_task and not self.cleanup_task.done():
            return
        
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(STATE_CLEANUP_INTERVAL)
                    self.cleanup_old_sessions()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Started cleanup task")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        return {
            "adapters": len(self.adapters),
            "active_sessions": len(self.sessions),
            "total_agents": len(self.agents),
            "claude_instances": len(self.claude_instances),
            "agent_statuses": {
                status.value: len([a for a in self.agents.values() if a.status == status])
                for status in AgentStatus
            }
        }


# Global state manager
state_manager = StateManager()


# ---------- Core Agent Execution ----------

class AgentExecutor:
    """Handles execution of individual agents"""
    
    def __init__(self, adapter: CodegenAdapter):
        self.adapter = adapter
        
    async def execute_agent(self, agent: AgentInstance) -> str:
        """Execute a single agent and return result"""
        try:
            state_manager.update_agent_status(agent.agent_id, AgentStatus.RUNNING)
            
            # Add some randomization to avoid identical results
            enhanced_prompt = self._enhance_prompt(agent.prompt, agent.metadata.get("agent_index", 0))
            
            # Create agent run via API
            result = await self.adapter.agent.create_agent_run(
                org_id=agent.org_id,
                prompt=enhanced_prompt,
                repo_id=agent.repo_id,
                metadata={
                    **agent.metadata,
                    "pro_mode": True,
                    "execution_timestamp": datetime.now().isoformat()
                }
            )
            
            api_agent_id = result.get("id")
            if not api_agent_id:
                raise Exception("Failed to create agent run - no ID returned")
            
            # Monitor agent execution
            agent_result = await self._monitor_agent_execution(
                agent.org_id, 
                api_agent_id,
                agent.agent_id,
                timeout=DEFAULT_TIMEOUT
            )
            
            state_manager.update_agent_status(
                agent.agent_id, 
                AgentStatus.COMPLETED, 
                result=agent_result
            )
            
            return agent_result
            
        except Exception as e:
            error_msg = str(e)
            state_manager.update_agent_status(
                agent.agent_id, 
                AgentStatus.FAILED, 
                error=error_msg
            )
            logger.error(f"Agent {agent.agent_id} failed: {error_msg}")
            raise
    
    def _enhance_prompt(self, base_prompt: str, agent_index: int) -> str:
        """Enhance prompt with variation for diversity"""
        variations = [
            "Focus on providing a comprehensive and detailed analysis.",
            "Prioritize practical solutions and actionable recommendations.",
            "Consider edge cases and potential challenges in your response.",
            "Emphasize best practices and industry standards.",
            "Think step-by-step and explain your reasoning clearly.",
            "Consider multiple perspectives and alternative approaches.",
            "Focus on efficiency, performance, and maintainability.",
            "Provide specific examples and concrete implementations.",
            "Consider the broader context and system implications.",
            "Emphasize security, reliability, and scalability aspects."
        ]
        
        variation = variations[agent_index % len(variations)]
        return f"{base_prompt}\n\nAdditional guidance: {variation}"
    
    async def _monitor_agent_execution(
        self, 
        org_id: int, 
        api_agent_id: int, 
        local_agent_id: str,
        timeout: int = DEFAULT_TIMEOUT
    ) -> str:
        """Monitor agent execution until completion"""
        start_time = time.time()
        check_interval = 2  # Start with 2 second intervals
        
        while time.time() - start_time < timeout:
            try:
                # Get agent status
                status_result = await self.adapter.agent.get_agent_run(org_id, api_agent_id)
                current_status = status_result.get("status", "unknown")
                
                # Log status update
                state_manager.add_agent_log(local_agent_id, {
                    "type": "status_check",
                    "api_status": current_status,
                    "elapsed_time": time.time() - start_time
                })
                
                if current_status == "completed":
                    result = status_result.get("result") or status_result.get("summary", "")
                    if result:
                        return result
                    else:
                        # Try to get logs for more detailed output
                        logs = await self.adapter.agent.get_agent_logs(org_id, api_agent_id, limit=50)
                        log_results = []
                        for log_entry in logs.get("logs", []):
                            if log_entry.get("observation"):
                                log_results.append(str(log_entry["observation"]))
                        
                        return "\n".join(log_results) if log_results else "Agent completed but no result available"
                
                elif current_status == "failed":
                    error_msg = status_result.get("error", "Agent execution failed")
                    raise Exception(f"Agent failed: {error_msg}")
                
                elif current_status == "cancelled":
                    raise Exception("Agent was cancelled")
                
                # Exponential backoff for polling interval
                await asyncio.sleep(min(check_interval, 10))
                check_interval = min(check_interval * 1.2, 10)
                
            except Exception as e:
                if "failed" in str(e).lower() or "cancelled" in str(e).lower():
                    raise
                logger.warning(f"Error monitoring agent {local_agent_id}: {e}")
                await asyncio.sleep(check_interval)
        
        raise Exception(f"Agent execution timed out after {timeout} seconds")


# ---------- Synthesis Engine ----------

class SynthesisEngine:
    """Handles synthesis of multiple agent results"""
    
    def __init__(self, adapter: CodegenAdapter):
        self.adapter = adapter
    
    async def synthesize_results(
        self, 
        candidates: List[str], 
        original_prompt: str,
        org_id: int,
        repo_id: Optional[int] = None
    ) -> str:
        """Synthesize multiple candidate results into a final result"""
        if not candidates:
            raise Exception("No candidates provided for synthesis")
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Build synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(candidates, original_prompt)
        
        # Execute synthesis agent
        result = await self.adapter.agent.create_agent_run(
            org_id=org_id,
            prompt=synthesis_prompt,
            repo_id=repo_id,
            metadata={
                "type": "synthesis",
                "num_candidates": len(candidates),
                "original_prompt": original_prompt
            }
        )
        
        api_agent_id = result.get("id")
        if not api_agent_id:
            raise Exception("Failed to create synthesis agent")
        
        # Monitor synthesis execution
        executor = AgentExecutor(self.adapter)
        synthesis_result = await executor._monitor_agent_execution(
            org_id, 
            api_agent_id, 
            f"synthesis-{uuid.uuid4()}",
            timeout=DEFAULT_TIMEOUT
        )
        
        return synthesis_result
    
    def _build_synthesis_prompt(self, candidates: List[str], original_prompt: str) -> str:
        """Build a prompt for synthesizing candidate results"""
        numbered_candidates = "\n\n".join(
            f"<candidate {i + 1}>\n{result}\n</candidate {i + 1}>"
            for i, result in enumerate(candidates)
        )
        
        return f"""You are an expert synthesizer and editor. Your task is to analyze multiple candidate responses to a prompt and create the single best final answer.

Original Prompt:
{original_prompt}

Candidate Responses:
{numbered_candidates}

Instructions:
1. Analyze all candidate responses carefully
2. Identify the best insights, solutions, and approaches from each candidate
3. Merge the strongest elements while correcting any errors or inconsistencies
4. Remove redundant information and improve clarity
5. Ensure the final result directly addresses the original prompt
6. Do not mention the synthesis process or refer to the candidates in your response
7. Be decisive, comprehensive, and provide a single authoritative answer

Provide the final synthesized response:"""
    
    async def tournament_synthesis(
        self,
        candidates: List[str],
        original_prompt: str,
        org_id: int,
        repo_id: Optional[int] = None,
        group_size: int = GROUP_SIZE
    ) -> Tuple[str, List[str]]:
        """Perform tournament-style synthesis with group winners"""
        if len(candidates) <= group_size:
            final_result = await self.synthesize_results(candidates, original_prompt, org_id, repo_id)
            return final_result, [final_result]
        
        # Group candidates
        groups = [candidates[i:i + group_size] for i in range(0, len(candidates), group_size)]
        
        # Synthesize each group in parallel
        group_synthesis_tasks = []
        for i, group in enumerate(groups):
            group_prompt = f"{original_prompt}\n\n[Group {i+1} synthesis from {len(group)} candidates]"
            task = self.synthesize_results(group, group_prompt, org_id, repo_id)
            group_synthesis_tasks.append(task)
        
        group_winners = await asyncio.gather(*group_synthesis_tasks)
        
        # Final synthesis
        final_result = await self.synthesize_results(group_winners, original_prompt, org_id, repo_id)
        
        return final_result, group_winners


# ---------- Pro Mode Engine ----------

class ProModeEngine:
    """Main Pro Mode execution engine"""
    
    def __init__(self):
        self.adapter = state_manager.get_adapter()
        self.executor = AgentExecutor(self.adapter)
        self.synthesizer = SynthesisEngine(self.adapter)
    
    async def execute_pro_mode(self, session: ProModeSession) -> ProModeResponse:
        """Execute Pro Mode with the specified strategy"""
        try:
            session.status = "running"
            
            # Create agent instances
            for i in range(session.num_agents):
                state_manager.create_agent_instance(session, i)
            
            # Execute based on strategy
            if session.strategy == ProModeStrategy.SIMPLE or session.num_agents <= TOURNAMENT_THRESHOLD:
                result = await self._execute_simple_mode(session)
            elif session.strategy == ProModeStrategy.TOURNAMENT:
                result = await self._execute_tournament_mode(session)
            else:  # ADAPTIVE
                result = await self._execute_adaptive_mode(session)
            
            session.status = "completed"
            session.final_result = result["final_result"]
            session.synthesis_results = result.get("synthesis_path", [])
            
            return self._build_response(session)
            
        except Exception as e:
            session.status = "failed"
            session.error = str(e)
            logger.error(f"Pro Mode execution failed for session {session.session_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _execute_simple_mode(self, session: ProModeSession) -> Dict[str, Any]:
        """Execute simple mode with direct synthesis"""
        # Execute all agents in parallel
        agent_tasks = [
            self.executor.execute_agent(agent) 
            for agent in session.agents
        ]
        
        # Use semaphore to limit concurrent agents
        semaphore = asyncio.Semaphore(MAX_WORKERS)
        
        async def limited_agent_execution(agent_task):
            async with semaphore:
                return await agent_task
        
        limited_tasks = [limited_agent_execution(task) for task in agent_tasks]
        results = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {i} failed: {result}")
            else:
                successful_results.append(result)
        
        if not successful_results:
            raise Exception("All agents failed")
        
        # Synthesize results
        final_result = await self.synthesizer.synthesize_results(
            successful_results,
            session.prompt,
            session.org_id,
            session.repo_id
        )
        
        return {
            "final_result": final_result,
            "candidates": successful_results,
            "synthesis_path": [final_result]
        }
    
    async def _execute_tournament_mode(self, session: ProModeSession) -> Dict[str, Any]:
        """Execute tournament mode with group synthesis"""
        # Execute all agents in parallel (same as simple mode)
        simple_result = await self._execute_simple_mode(session)
        candidates = simple_result["candidates"]
        
        # Perform tournament synthesis
        final_result, synthesis_path = await self.synthesizer.tournament_synthesis(
            candidates,
            session.prompt,
            session.org_id,
            session.repo_id
        )
        
        return {
            "final_result": final_result,
            "candidates": candidates,
            "synthesis_path": synthesis_path
        }
    
    async def _execute_adaptive_mode(self, session: ProModeSession) -> Dict[str, Any]:
        """Execute adaptive mode - chooses strategy based on conditions"""
        if session.num_agents <= TOURNAMENT_THRESHOLD:
            return await self._execute_simple_mode(session)
        else:
            return await self._execute_tournament_mode(session)
    
    def _build_response(self, session: ProModeSession) -> ProModeResponse:
        """Build the final Pro Mode response"""
        # Collect successful results
        candidates = []
        for agent in session.agents:
            if agent.status == AgentStatus.COMPLETED and agent.result:
                candidates.append(agent.result)
        
        # Calculate statistics
        completed_agents = len([a for a in session.agents if a.status == AgentStatus.COMPLETED])
        failed_agents = len([a for a in session.agents if a.status == AgentStatus.FAILED])
        total_duration = sum(
            a.duration for a in session.agents 
            if a.duration is not None
        ) or 0
        
        stats = {
            "total_agents": len(session.agents),
            "completed_agents": completed_agents,
            "failed_agents": failed_agents,
            "success_rate": completed_agents / len(session.agents) if session.agents else 0,
            "total_duration": total_duration,
            "average_duration": total_duration / completed_agents if completed_agents > 0 else 0,
            "strategy_used": session.strategy.value
        }
        
        return ProModeResponse(
            session_id=session.session_id,
            status=session.status,
            final_result=session.final_result,
            candidates=candidates,
            synthesis_path=session.synthesis_results,
            stats=stats,
            created_at=session.created_at,
            completed_at=datetime.now() if session.status in ["completed", "failed"] else None
        )


# ---------- FastAPI Dependencies ----------

def get_state_manager() -> StateManager:
    """Dependency to get the state manager"""
    return state_manager


def get_pro_mode_engine() -> ProModeEngine:
    """Dependency to get the Pro Mode engine"""
    return ProModeEngine()


# ---------- API Routes ----------

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Codegen Pro Mode API")
    
    # Start cleanup task
    await state_manager.start_cleanup_task()
    
    # Verify adapter connectivity
    try:
        adapter = state_manager.get_adapter()
        health = adapter.health_check()
        logger.info(f"Adapter health check: {health}")
        
        if not health.get("api_client", False):
            logger.warning("API client not available - some functionality may be limited")
        
    except Exception as e:
        logger.error(f"Failed to initialize adapter: {e}")


@app.post("/pro-mode", response_model=ProModeResponse)
async def create_pro_mode_session(
    request: ProModeRequest,
    background_tasks: BackgroundTasks,
    engine: ProModeEngine = Depends(get_pro_mode_engine)
) -> ProModeResponse:
    """Create and execute a new Pro Mode session"""
    
    # Validate request
    if request.num_agents > MAX_AGENTS:
        raise HTTPException(
            status_code=400, 
            detail=f"Number of agents ({request.num_agents}) exceeds maximum ({MAX_AGENTS})"
        )
    
    # Create session
    session = state_manager.create_session(request)
    
    # Execute in background for async operation
    async def execute_session():
        try:
            await engine.execute_pro_mode(session)
        except Exception as e:
            logger.error(f"Background execution failed for session {session.session_id}: {e}")
    
    background_tasks.add_task(execute_session)
    
    # Return immediate response with session info
    return ProModeResponse(
        session_id=session.session_id,
        status="initializing",
        final_result=None,
        candidates=[],
        synthesis_path=[],
        stats={"total_agents": request.num_agents},
        created_at=session.created_at,
        completed_at=None
    )


@app.post("/pro-mode-sync", response_model=ProModeResponse) 
async def create_pro_mode_session_sync(
    request: ProModeRequest,
    engine: ProModeEngine = Depends(get_pro_mode_engine)
) -> ProModeResponse:
    """Create and execute a Pro Mode session synchronously"""
    
    # Validate request
    if request.num_agents > MAX_AGENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Number of agents ({request.num_agents}) exceeds maximum ({MAX_AGENTS})"
        )
    
    # Create session
    session = state_manager.create_session(request)
    
    # Execute synchronously
    return await engine.execute_pro_mode(session)


@app.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str) -> SessionStatusResponse:
    """Get the status of a Pro Mode session"""
    session = state_manager.get_session(session_id)
    
    # Build agent status responses
    agent_responses = []
    for agent in session.agents:
        agent_responses.append(AgentStatusResponse(
            agent_id=agent.agent_id,
            status=agent.status,
            result=agent.result,
            error=agent.error,
            created_at=agent.created_at,
            duration=agent.duration,
            logs_count=len(agent.logs)
        ))
    
    # Calculate progress
    completed_count = len([a for a in session.agents if a.is_terminal])
    progress = completed_count / len(session.agents) if session.agents else 0.0
    
    # Build stats
    stats = {
        "total_agents": len(session.agents),
        "completed_agents": len([a for a in session.agents if a.status == AgentStatus.COMPLETED]),
        "failed_agents": len([a for a in session.agents if a.status == AgentStatus.FAILED]),
        "running_agents": len([a for a in session.agents if a.status == AgentStatus.RUNNING]),
        "strategy": session.strategy.value,
        "progress": progress
    }
    
    return SessionStatusResponse(
        session_id=session.session_id,
        status=session.status,
        prompt=session.prompt,
        num_agents=session.num_agents,
        strategy=session.strategy,
        agents=agent_responses,
        progress=progress,
        final_result=session.final_result,
        stats=stats
    )


@app.get("/sessions", response_model=List[SessionStatusResponse])
async def list_sessions(limit: int = 50, offset: int = 0) -> List[SessionStatusResponse]:
    """List all Pro Mode sessions"""
    sessions = list(state_manager.sessions.values())
    
    # Sort by creation time (newest first)
    sessions.sort(key=lambda s: s.created_at, reverse=True)
    
    # Apply pagination
    paginated_sessions = sessions[offset:offset + limit]
    
    # Build responses
    responses = []
    for session in paginated_sessions:
        # Build agent status responses
        agent_responses = []
        for agent in session.agents:
            agent_responses.append(AgentStatusResponse(
                agent_id=agent.agent_id,
                status=agent.status,
                result=agent.result,
                error=agent.error,
                created_at=agent.created_at,
                duration=agent.duration,
                logs_count=len(agent.logs)
            ))
        
        # Calculate progress
        completed_count = len([a for a in session.agents if a.is_terminal])
        progress = completed_count / len(session.agents) if session.agents else 0.0
        
        responses.append(SessionStatusResponse(
            session_id=session.session_id,
            status=session.status,
            prompt=session.prompt,
            num_agents=session.num_agents,
            strategy=session.strategy,
            agents=agent_responses,
            progress=progress,
            final_result=session.final_result,
            stats={"progress": progress}
        ))
    
    return responses


@app.delete("/sessions/{session_id}")
async def cancel_session(session_id: str) -> Dict[str, str]:
    """Cancel a Pro Mode session"""
    session = state_manager.get_session(session_id)
    
    # Cancel all running agents
    for agent in session.agents:
        if agent.status == AgentStatus.RUNNING:
            state_manager.update_agent_status(agent.agent_id, AgentStatus.CANCELLED)
    
    session.status = "cancelled"
    
    return {"message": f"Session {session_id} cancelled"}


@app.get("/agents/{agent_id}", response_model=AgentStatusResponse)
async def get_agent_status(agent_id: str) -> AgentStatusResponse:
    """Get the status of a specific agent"""
    if agent_id not in state_manager.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = state_manager.agents[agent_id]
    
    return AgentStatusResponse(
        agent_id=agent.agent_id,
        status=agent.status,
        result=agent.result,
        error=agent.error,
        created_at=agent.created_at,
        duration=agent.duration,
        logs_count=len(agent.logs)
    )


@app.get("/agents/{agent_id}/logs")
async def get_agent_logs(agent_id: str) -> Dict[str, Any]:
    """Get logs for a specific agent"""
    if agent_id not in state_manager.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = state_manager.agents[agent_id]
    
    return {
        "agent_id": agent_id,
        "logs": agent.logs,
        "logs_count": len(agent.logs)
    }


@app.get("/claude-instances", response_model=List[Dict[str, Any]])
async def list_claude_instances() -> List[Dict[str, Any]]:
    """List all Claude Code instances"""
    instances = []
    for instance_id, claude in state_manager.claude_instances.items():
        instances.append({
            "instance_id": instance_id,
            "session_name": claude.session_name,
            "project_path": claude.project_path,
            "status": claude.status,
            "created_at": claude.created_at,
            "last_activity": claude.last_activity,
            "metadata": claude.metadata
        })
    
    return instances


@app.post("/claude-instances")
async def create_claude_instance(
    session_name: str,
    project_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new Claude Code instance"""
    try:
        adapter = state_manager.get_adapter()
        
        # Create Claude session
        session_info = adapter.claude.create_claude_session(
            name=session_name,
            description=f"Pro Mode Claude instance - {session_name}"
        )
        
        # Create instance tracking
        claude = state_manager.create_claude_instance(session_name, project_path)
        claude.status = "active"
        claude.metadata = {"session_info": session_info}
        
        return {
            "instance_id": claude.instance_id,
            "session_name": session_name,
            "project_path": project_path,
            "status": "active",
            "session_info": session_info
        }
        
    except Exception as e:
        logger.error(f"Failed to create Claude instance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        adapter = state_manager.get_adapter()
        adapter_health = adapter.health_check()
        stats = state_manager.get_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "adapter_health": adapter_health,
            "system_stats": stats,
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "version": "1.0.0"
        }


@app.get("/stats")
async def get_system_stats() -> Dict[str, Any]:
    """Get detailed system statistics"""
    stats = state_manager.get_stats()
    
    # Add performance metrics
    performance_stats = {
        "avg_session_duration": 0,
        "success_rate": 0,
        "total_sessions": len(state_manager.sessions)
    }
    
    if state_manager.sessions:
        completed_sessions = [s for s in state_manager.sessions.values() if s.status == "completed"]
        if completed_sessions:
            # Calculate average duration for completed sessions
            durations = []
            for session in completed_sessions:
                session_agents = [a for a in session.agents if a.duration]
                if session_agents:
                    avg_duration = sum(a.duration for a in session_agents) / len(session_agents)
                    durations.append(avg_duration)
            
            if durations:
                performance_stats["avg_session_duration"] = sum(durations) / len(durations)
            
            performance_stats["success_rate"] = len(completed_sessions) / len(state_manager.sessions)
    
    return {
        **stats,
        "performance": performance_stats,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/cleanup")
async def trigger_cleanup() -> Dict[str, str]:
    """Manually trigger cleanup of old sessions"""
    try:
        state_manager.cleanup_old_sessions(max_age_hours=1)
        return {"message": "Cleanup completed successfully"}
    except Exception as e:
        logger.error(f"Manual cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Configuration and Optimization ----------

class ProModeConfig:
    """Configuration class for Pro Mode optimization"""
    
    def __init__(self):
        self.max_agents = int(os.getenv("PRO_MODE_MAX_AGENTS", MAX_AGENTS))
        self.max_workers = int(os.getenv("PRO_MODE_MAX_WORKERS", MAX_WORKERS))
        self.tournament_threshold = int(os.getenv("PRO_MODE_TOURNAMENT_THRESHOLD", TOURNAMENT_THRESHOLD))
        self.group_size = int(os.getenv("PRO_MODE_GROUP_SIZE", GROUP_SIZE))
        self.default_timeout = int(os.getenv("PRO_MODE_DEFAULT_TIMEOUT", DEFAULT_TIMEOUT))
        self.cleanup_interval = int(os.getenv("PRO_MODE_CLEANUP_INTERVAL", STATE_CLEANUP_INTERVAL))
        
        # Advanced configuration
        self.enable_adaptive_timeout = os.getenv("PRO_MODE_ADAPTIVE_TIMEOUT", "true").lower() == "true"
        self.enable_result_caching = os.getenv("PRO_MODE_RESULT_CACHING", "false").lower() == "true"
        self.enable_load_balancing = os.getenv("PRO_MODE_LOAD_BALANCING", "true").lower() == "true"
        
        # Logging configuration
        self.log_level = os.getenv("PRO_MODE_LOG_LEVEL", "INFO")
        self.enable_detailed_logging = os.getenv("PRO_MODE_DETAILED_LOGGING", "false").lower() == "true"
        
        # Performance tuning
        self.agent_poll_interval = float(os.getenv("PRO_MODE_POLL_INTERVAL", "2.0"))
        self.synthesis_timeout_multiplier = float(os.getenv("PRO_MODE_SYNTHESIS_TIMEOUT_MULTIPLIER", "1.5"))


# Global configuration
config = ProModeConfig()


# ---------- Usage Examples and CLI ----------

def create_example_request() -> ProModeRequest:
    """Create an example Pro Mode request for testing"""
    return ProModeRequest(
        prompt="Analyze the codebase and provide comprehensive recommendations for improving code quality, performance, and maintainability. Focus on identifying potential bugs, security issues, and optimization opportunities.",
        num_agents=5,
        org_id=123,  # Replace with actual org ID
        repo_id=456,  # Replace with actual repo ID
        strategy=ProModeStrategy.ADAPTIVE,
        timeout=300,
        metadata={
            "example": True,
            "priority": "high",
            "analysis_type": "comprehensive"
        }
    )


async def run_example_session():
    """Run an example Pro Mode session"""
    print("🚀 Running Codegen Pro Mode Example")
    
    # Create request
    request = create_example_request()
    print(f"📝 Created request with {request.num_agents} agents using {request.strategy} strategy")
    
    # Initialize engine
    engine = ProModeEngine()
    
    # Create session
    session = state_manager.create_session(request)
    print(f"🎯 Created session: {session.session_id}")
    
    try:
        # Execute Pro Mode
        print("⚡ Executing Pro Mode session...")
        response = await engine.execute_pro_mode(session)
        
        print("✅ Pro Mode execution completed!")
        print(f"📊 Results:")
        print(f"  - Status: {response.status}")
        print(f"  - Candidates: {len(response.candidates)}")
        print(f"  - Success Rate: {response.stats.get('success_rate', 0):.2%}")
        print(f"  - Total Duration: {response.stats.get('total_duration', 0):.2f}s")
        
        if response.final_result:
            print(f"🎉 Final Result Preview:")
            preview = response.final_result[:200] + "..." if len(response.final_result) > 200 else response.final_result
            print(f"  {preview}")
        
        return response
        
    except Exception as e:
        print(f"❌ Example session failed: {e}")
        return None


# ---------- Main Application ----------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Codegen Pro Mode - Advanced Agent Orchestration")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)), help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--example", action="store_true", help="Run example session instead of starting server")
    parser.add_argument("--config", action="store_true", help="Show configuration and exit")
    
    args = parser.parse_args()
    
    if args.config:
        print("🔧 Codegen Pro Mode Configuration:")
        print(f"  Max Agents: {config.max_agents}")
        print(f"  Max Workers: {config.max_workers}")
        print(f"  Tournament Threshold: {config.tournament_threshold}")
        print(f"  Group Size: {config.group_size}")
        print(f"  Default Timeout: {config.default_timeout}s")
        print(f"  Cleanup Interval: {config.cleanup_interval}s")
        print(f"  Adaptive Timeout: {config.enable_adaptive_timeout}")
        print(f"  Result Caching: {config.enable_result_caching}")
        print(f"  Load Balancing: {config.enable_load_balancing}")
        print(f"  Log Level: {config.log_level}")
        exit(0)
    
    if args.example:
        print("🧪 Running Pro Mode Example")
        asyncio.run(run_example_session())
        exit(0)
    
    # Configure logging based on config
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    
    if config.enable_detailed_logging:
        logging.getLogger("uvicorn").setLevel(logging.DEBUG)
        logging.getLogger("fastapi").setLevel(logging.DEBUG)
    
    print(f"🚀 Starting Codegen Pro Mode API")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print(f"📚 Docs: http://{args.host}:{args.port}/docs")
    print(f"⚡ Max Agents: {config.max_agents}")
    print(f"🔧 Strategy: Adaptive tournament synthesis")
    
    uvicorn.run(
        "codegen_pro_mode:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=config.log_level.lower()
    )
