# SentinelPM — Release Management & Governance

## 1. Versioning Policy

SentinelPM follows [Semantic Versioning 2.0.0](https://semver.org/):

$$\text{MAJOR}.\text{MINOR}.\text{PATCH}$$

- **MAJOR (v1.0.0):** Incompatible architectural changes, API contract breaks, or radical schema restructuring.
- **MINOR (v0.2.0):** Backward-compatible new features (e.g., adding a new model architecture or dashboard tab).
- **PATCH (v0.1.1):** Backward-compatible bug fixes, performance optimizations, or documentation revisions.

---

## 2. Pre-Release Quality Gates

Before any version tag or production release is approved, all seven quality gates must pass:

| Gate | Verification Check | Command | Pass Criteria |
| --- | --- | --- | --- |
| **1. Code Quality** | Static analysis & linting | `ruff check .` | 0 errors, 0 warnings |
| **2. Formatting** | PEP 8 style formatting | `ruff format --check .` | 100% compliant |
| **3. Test Suite** | Full unit and integration tests | `pytest --cov=app` | 100% passed (220+ tests) |
| **4. Smoke Test** | End-to-end service verification | `python -m scripts.smoke_test` | 6/6 checks PASSED |
| **5. Container** | Docker configuration syntax | `docker compose config` | Exit code 0 |
| **6. Benchmarks** | Latency & SLA verification | `python -m scripts.benchmark` | p95 < 250ms, Error rate = 0% |
| **7. Security** | Credentials & path leak audit | `git status` & code scan | 0 secrets, 0 hardcoded absolute paths |

---

## 3. Step-by-Step Release Procedure

### Step 1: Execute Full Automated Verification
```powershell
# 1. Verify code formatting and linting
ruff check .
ruff format --check .

# 2. Run complete test suite with coverage
pytest --cov=app --cov-report=term-missing

# 3. Execute system smoke test
python -m scripts.smoke_test

# 4. Generate empirical performance benchmarks
python -m scripts.benchmark
```

### Step 2: Update Version Metadata
Update version strings in:
1. `pyproject.toml` (`version = "X.Y.Z"`)
2. `app/__init__.py` (`__version__ = "X.Y.Z"`)
3. `README.md` and release notes.

### Step 3: Git Tagging & Commit
```bash
git add pyproject.toml app/__init__.py README.md docs/
git commit -m "chore(release): bump version to vX.Y.Z"
git tag -a vX.Y.Z -m "Release vX.Y.Z: SentinelPM Industrial Predictive Maintenance Platform"
git push origin main --tags
```

### Step 4: Container Build & Verification
```bash
docker compose build
docker compose up -d
python -m scripts.smoke_test --url http://127.0.0.1:8000
docker compose down
```

---

## 4. Rollback Plan

If a defect or performance regression is identified post-deployment:

1. **Service Rollback:** In Docker Compose, roll back to the previously tagged container image:
   ```bash
   docker compose down
   # Point docker-compose.yml to previous image tag: sentinelpm:vX.Y.Y
   docker compose up -d
   ```
2. **Model Rollback:** Set `SENTINELPM_ACTIVE_MODEL=random_forest` or restore previous approved artifact from `./artifacts/models/`.
3. **Hotfix Protocol:** Create a hotfix branch from the release tag, fix the defect with accompanying tests, and issue a patch release `vX.Y.Z+1`.
