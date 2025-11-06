# Codegen API Transformation

This document describes the comprehensive transformation of the ComfyUI-focused API to fully align with the Codegen project's organization-centric architecture.

## 🎯 Overview

The transformation implements a complete architectural shift from ComfyUI workflow patterns to Codegen's `/v1/organizations/{org_id}/*` API structure while maintaining backward compatibility.

### Key Features

- **Organization-Centric Architecture**: All endpoints follow `/v1/organizations/{org_id}/*` patterns
- **Advanced Authentication**: Organization-scoped API keys with role-based permissions
- **Rate Limiting**: 60 requests per 30-second window per organization (matching Codegen patterns)
- **Pro Mode Integration**: Tournament synthesis system with organization context
- **Comprehensive Validation**: Full Pydantic model validation matching Codegen schemas
- **Backward Compatibility**: Legacy ComfyUI endpoints remain functional
- **Database Migration**: Safe migration with rollback capabilities

## 🏗️ Architecture Components

### 1. Database Models (`codegen_models.py`)

New database schema implementing organization-centric patterns:

- **CodegenOrganization**: Core organization entity
- **CodegenOrganizationMembership**: User-organization relationships with roles
- **CodegenProject**: Organization-scoped projects  
- **CodegenAPIKey**: Organization-scoped authentication
- **CodegenSession**: Pro Mode session management
- **CodegenAgentInstance**: Agent execution tracking
- **CodegenRateLimit**: Rate limiting enforcement

```python
# Example: Creating an organization
org = CodegenOrganization(
    name="Acme Corp",
    display_name="ACME Corporation",
    status=OrganizationStatus.ACTIVE,
    rate_limit_per_minute=60
)
```

### 2. API Schemas (`codegen_schemas.py`)

Pydantic models matching Codegen's exact API patterns:

- **Response Models**: `UserResponse`, `OrganizationResponse`, `ProjectResponse`
- **Pagination**: `Page_UserResponse_`, `PaginationMeta`
- **Error Handling**: `APIRateLimitErrorResponse`, `PermissionsErrorResponse`
- **Request Models**: `CreateProjectRequest`, `ProModeRequest`

```python
# Example: Paginated user response
page = Page_UserResponse_(
    items=[UserResponse(id=1, name="John Doe", ...)],
    meta=PaginationMeta(total=1, skip=0, limit=100, has_more=False)
)
```

### 3. Middleware Stack (`codegen_middleware.py`)

Three-layer middleware implementing Codegen patterns:

#### CodegenAuthMiddleware
- Validates organization-scoped API keys
- Injects `OrganizationContext` into requests
- Supports role-based permissions (Owner, Admin, Member, Viewer)

#### CodegenRateLimitMiddleware  
- Enforces 60 requests per 30-second window
- Per-organization and per-user granularity
- Automatic cleanup of expired entries

#### CodegenPermissionMiddleware
- Fine-grained permission checking
- Endpoint-specific permission requirements
- Role hierarchy enforcement

```python
# Example: Checking permissions
if org_context.has_permission("write"):
    # Allow project creation
    pass
```

### 4. API Routes (`codegen_routes.py`)

Complete implementation of Codegen endpoint patterns:

- `GET /v1/organizations/{org_id}/users` - List organization users
- `GET /v1/organizations/{org_id}/users/{user_id}` - Get specific user
- `POST /v1/organizations/{org_id}/projects` - Create project
- `GET /v1/organizations/{org_id}/projects/{project_id}/sessions` - List sessions
- `POST /v1/organizations/{org_id}/projects/{project_id}/pro-mode` - Create Pro Mode session

### 5. Integration Layer (`codegen_integration.py`)

Seamless integration with existing ComfyUI API:

- **Feature Flags**: Gradual rollout control
- **Exception Handlers**: Codegen-compatible error formats
- **OpenAPI Integration**: Unified documentation
- **Health Checks**: Component-specific monitoring

## 🚀 Deployment

### Prerequisites

1. **Python 3.8+** with asyncio support
2. **PostgreSQL 12+** database
3. **Existing ComfyUI API** installation
4. **Environment Variables**:
   ```bash
   DATABASE_URL=postgresql://user:pass@host:port/db
   ENV=production
   CODEGEN_ORG_AUTH_ENABLED=true
   CODEGEN_RATE_LIMITING_ENABLED=true
   CODEGEN_PRO_MODE_V2_ENABLED=true
   ```

### Step 1: Database Migration

```bash
# Run migrations
cd database_migrations
python migrate.py run --verbose

# Validate migrations
python migrate.py validate

# Check status
python migrate.py status
```

### Step 2: Production Deployment

```bash
# Dry run deployment
python deploy_codegen_transformation.py deploy --environment production --dry-run

# Execute deployment
python deploy_codegen_transformation.py deploy --environment production --verbose
```

### Step 3: Validation

```bash
# Run validation
python deploy_codegen_transformation.py validate --environment production

# Run comprehensive tests
python -m pytest tests/test_codegen_api_transformation.py -v
```

## 📋 API Usage Examples

### Authentication

All organization endpoints require Bearer authentication:

```bash
curl -H "Authorization: Bearer your-org-api-key" \
     https://api.comfydeploy.com/api/v1/organizations/123/users
```

### Creating a Project

```bash
curl -X POST \
     -H "Authorization: Bearer your-org-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "My Project",
       "description": "A new project",
       "repository_url": "https://github.com/user/repo"
     }' \
     https://api.comfydeploy.com/api/v1/organizations/123/projects
```

### Pro Mode Session

```bash
curl -X POST \
     -H "Authorization: Bearer your-org-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Create a web application",
       "agents": ["general-purpose", "web-developer"],
       "max_iterations": 3,
       "timeout_minutes": 60
     }' \
     https://api.comfydeploy.com/api/v1/organizations/123/projects/456/pro-mode
```

## 🔄 Migration Strategy

### Phase 1: Infrastructure (Completed)
- ✅ Database schema migration
- ✅ Middleware implementation
- ✅ Authentication system
- ✅ Rate limiting

### Phase 2: API Endpoints (Completed)
- ✅ Organization management endpoints
- ✅ User management endpoints  
- ✅ Project management endpoints
- ✅ Pro Mode integration

### Phase 3: Integration (Completed)
- ✅ Backward compatibility
- ✅ Feature flags
- ✅ Error handling standardization
- ✅ OpenAPI documentation

## 🧪 Testing

### Unit Tests
```bash
# Run all codegen transformation tests
pytest tests/test_codegen_api_transformation.py -v

# Run specific test classes
pytest tests/test_codegen_api_transformation.py::TestCodegenModels -v
pytest tests/test_codegen_api_transformation.py::TestCodegenMiddleware -v
```

### Integration Tests
```bash
# Test with real database (requires test database)
pytest tests/test_codegen_api_transformation.py --integration

# Test Pro Mode integration
pytest tests/test_codegen_api_transformation.py::TestProModeIntegration -v
```

### Load Testing
```bash
# Test rate limiting
pytest tests/test_codegen_api_transformation.py::TestRateLimiting -v

# Performance testing
python -m pytest tests/performance/ --benchmark-only
```

## 📊 Monitoring

### Health Checks

The system provides multiple health check endpoints:

- `/v1/health` - Overall system health
- `/v1/stats` - System statistics  
- `/api/v1/health/codegen` - Codegen component health

### Metrics

Key metrics to monitor:

- **Request Rate**: Requests per second per organization
- **Rate Limit Hits**: Number of rate limit violations
- **Authentication Failures**: Failed auth attempts
- **Database Performance**: Query execution times
- **Pro Mode Sessions**: Active and completed sessions

### Logging

Structured logging with key events:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "component": "codegen_auth",
  "organization_id": 123,
  "user_id": "user-456",
  "event": "authentication_success",
  "endpoint": "/v1/organizations/123/projects"
}
```

## 🔒 Security

### Authentication
- Organization-scoped API keys with SHA-256 hashing
- Role-based access control (Owner, Admin, Member, Viewer)
- JWT token validation for session management

### Authorization
- Endpoint-specific permission requirements
- Organization membership validation
- Resource ownership verification

### Rate Limiting
- Organization-level rate limiting (60 req/30s)
- User-level granular limits
- Automatic request throttling

### Data Protection
- SQL injection prevention with parameterized queries
- Input validation with Pydantic models
- Secure password handling (external auth systems)

## 🔧 Troubleshooting

### Common Issues

#### Migration Failures
```bash
# Check migration status
python database_migrations/migrate.py status

# Rollback if needed
python database_migrations/migrate.py rollback --migration-id 001_create_codegen_tables
```

#### Authentication Errors
```bash
# Verify API key
curl -H "Authorization: Bearer your-key" https://api.comfydeploy.com/api/v1/health

# Check organization membership
SELECT * FROM codegen_organization_memberships WHERE user_id = 'your-user-id';
```

#### Rate Limiting Issues
```bash
# Check rate limit entries
SELECT * FROM codegen_rate_limits WHERE organization_id = 123;

# Clear rate limit cache
DELETE FROM codegen_rate_limits WHERE last_request_at < NOW() - INTERVAL '1 hour';
```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export CODEGEN_DEBUG=true
```

## 🚀 Performance Optimization

### Database Optimization
- Proper indexing on organization_id columns
- Connection pooling with SQLAlchemy
- Query optimization for pagination

### Caching Strategy
- Redis for rate limit tracking
- Organization context caching
- API response caching for read-heavy endpoints

### Scaling Considerations
- Horizontal scaling with load balancers
- Database read replicas for read-heavy operations
- Microservice decomposition for high-volume organizations

## 📈 Future Enhancements

### Planned Features
- [ ] Advanced analytics dashboard
- [ ] Multi-region deployment support  
- [ ] Enhanced Pro Mode algorithms
- [ ] Real-time collaboration features
- [ ] Advanced security features (2FA, SSO)

### API Versioning
- Current: v1 (Codegen-aligned)
- Legacy: ComfyUI endpoints (maintained for compatibility)
- Future: v2 with enhanced features

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd api

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Standards
- **Python**: Follow PEP 8 with Black formatting
- **API Design**: Follow RESTful principles
- **Testing**: Minimum 80% code coverage
- **Documentation**: Comprehensive docstrings and comments

## 📞 Support

### Getting Help
- 📧 Email: support@comfydeploy.com
- 📖 Documentation: Internal wiki
- 🐛 Issues: GitHub Issues
- 💬 Chat: Internal Slack #api-support

### Emergency Contacts
- **Production Issues**: On-call engineer
- **Security Issues**: Security team immediately
- **Database Issues**: DBA team

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Database backup created
- [ ] Migration files validated
- [ ] Environment variables set
- [ ] Health checks passing
- [ ] Load balancer configured

### Deployment
- [ ] Maintenance window scheduled
- [ ] Database migrations executed
- [ ] Application deployed
- [ ] Feature flags enabled
- [ ] Health checks validated

### Post-Deployment
- [ ] All endpoints responding
- [ ] Rate limiting functional
- [ ] Authentication working
- [ ] Pro Mode integration tested
- [ ] Monitoring alerts configured
- [ ] Documentation updated

---

*This transformation represents a complete architectural evolution from ComfyUI patterns to Codegen's organization-centric approach while maintaining full backward compatibility and providing enhanced features for collaborative development.*