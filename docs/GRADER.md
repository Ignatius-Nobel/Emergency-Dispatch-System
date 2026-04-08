# Fix: OpenEnv Phase 2 Validation - Missing Graders

## Context

The dispatch_grid OpenEnv environment failed phase 2 validation during Hugging Face Spaces submission with the error:
> "Your submission must include at least 3 tasks with graders."

**Root Cause:** The environment has 4 tasks defined in `openenv.yaml` and a `/grader` HTTP endpoint, but lacks the required `graders/` directory with Python grader classes that OpenEnv's submission system expects.

## Problem Analysis

Current state:
- `openenv.yaml` defines 4 tasks: easy, medium, hard, crisis ✓
- `server/app.py` has a `/grader` endpoint ✓
- `server/dispatch_grid_environment.py` has `compute_reward()` function ✓
- **Missing:** `graders/` directory with grader class implementations ✗

Reference: The Vittal-M/openenv-hackathon example shows the expected structure:
- `graders/__init__.py` - exports grader classes
- `graders/grader_detection.py` - Task 1 grader implementation
- `graders/grader_classification.py` - Task 2 grader implementation
- `graders/grader_fix.py` - Task 3 grader implementation

## Implementation Plan

### 1. Create `graders/` directory structure

Files to create:
- `graders/__init__.py`
- `graders/grader_easy.py` (for easy task)
- `graders/grader_medium.py` (for medium task)
- `graders/grader_hard.py` (for hard task)

### 2. Implement grader classes

Each grader class should:
- Have a `grade(action, ground_truth)` method returning a float reward
- Store `last_breakdown` dict for training feedback
- Normalize common synonyms in responses
- Use the existing `compute_reward()` function from the environment

**Grader structure (based on reference implementation):**

```python
class EasyGrader:
    """Grade Task 1: Basic Emergency Dispatch."""

    def __init__(self) -> None:
        self.last_breakdown: dict[str, Any] = {}

    def grade(self, action: Action, ground_truth: dict[str, Any]) -> float:
        # Use compute_reward from environment
        # Return normalized score 0.0-1.0
```

### 3. Update `openenv.yaml` to reference graders

Add grader references to each task definition:

```yaml
tasks:
  - id: easy
    name: Basic Emergency Dispatch
    difficulty: easy
    description: ...
    grader: graders.EasyGrader  # Add this line

  - id: medium
    name: Ambiguous & Multi-Type Dispatch
    difficulty: medium
    description: ...
    grader: graders.MediumGrader

  - id: hard
    name: Multi-Hazard Cascading Emergencies
    difficulty: hard
    description: ...
    grader: graders.HardGrader
```

### 4. Files to Modify/Create

**Create:**
- `graders/__init__.py` - Package init exporting all graders
- `graders/grader_easy.py` - Easy task grader
- `graders/grader_medium.py` - Medium task grader
- `graders/grader_hard.py` - Hard task grader

**Modify:**
- `openenv.yaml` - Add grader references to task definitions

### 5. Grader Implementation Details

Each grader will:
1. Import `compute_reward` from `server.dispatch_grid_environment`
2. Import `DispatchGridAction` from `models`
3. Import `EmergencyCall` with ground truth data
4. Call `compute_reward(action, call)` to get raw score
5. Normalize to 0.0-1.0 range
6. Return breakdown with feedback

### 6. Verification

After implementation:
1. Run `openenv validate --verbose` locally
2. Test graders can be imported: `python -c "from graders import EasyGrader, MediumGrader, HardGrader"`
3. Verify each grader's `grade()` method works with sample data
4. Attempt `openenv push` to verify phase 2 validation passes

## Key Files

- `graders/__init__.py`
- `graders/grader_easy.py`
- `graders/grader_medium.py`
- `graders/grader_hard.py`
- `openenv.yaml` (modified)
- `server/dispatch_grid_environment.py` (existing compute_reward to reuse)
- `models.py` (existing Action/Observation schemas to reuse)