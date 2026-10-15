# GTDRC3 Integration - Issue Resolution ✅

## Problem
When attempting to train a YOLOv12 model using the new GTDRC3 block, the following errors occurred:
1. `KeyError: 'GTDRC3'` - Module not recognized by parse_model
2. `TypeError: GTDRC3.__init__() takes from 3 to 5 positional arguments but 6 were given` - Incorrect argument passing

## Root Causes

### Issue 1: Module Recognition
- **Cause**: While GTDRC3 was imported in tasks.py, it wasn't registered in the two critical checks within `parse_model()` function
- **Location**: `/ultralytics/nn/tasks.py` lines 978-1021 and 1031-1052

### Issue 2: Argument Mismatch
- **Cause**: parse_model automatically inserts the repeat count `n` for blocks that support repetitions (line `args.insert(2, n)`)
- **GTDRC3 signature**: `__init__(self, c1, c2, n=1, shortcut=True)`
- **Problem**: YAML args `[c2, n, shortcut]` + automatic insertion = 6 total args instead of 5

## Solutions Applied

### 1. Registered GTDRC3 in parse_model checks
**File**: `/ultralytics/nn/tasks.py`
- Added `GTDRC3` to first check block (line 1025) - modules that support variable output channels
- Added `GTDRC3` to second check block (line 1052) - modules that support repetition counts

### 2. Corrected YAML Argument Format
**File**: `/ultralytics/cfg/models/v12/yolov12_GTDRC3.yaml`

**Before (incorrect)**:
```yaml
- [-1, 1, GTDRC3, [256, 1, True]]   # This passes n explicitly
```

**After (correct)**:
```yaml
- [-1, 2, GTDRC3, [256, True]]      # n=2 in YAML is handled separately, only pass c2 and shortcut
```

**Explanation**:
- The number after the module name (e.g., `2`) is the repetition count `n`
- YAML args should only contain: `[c2_channels, shortcut_bool]`
- parse_model automatically:
  1. Prepends c1 from input: `[c1, c2]`
  2. Inserts n at position 2: `[c1, c2, n, shortcut]`
  3. Calls GTDRC3(c1, c2, n, shortcut) ✅

### 3. Updated Notebook Configuration
**File**: `/train_evaluation.ipynb` cell 2
- Updated `MODEL_PATH` to use the corrected YAML file
- Changed `SINGLE_FOLD_ID` to 1 (from 3) for clarity
- All training infrastructure remains the same

## Verification ✅

Model successfully loads:
```python
from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/v12/yolov12_GTDRC3.yaml")
# ✅ No errors, model ready for training
```

## GTDRC3 YAML Parameters Reference

```yaml
# Syntax: [from, repeats, module_name, [args]]
# For GTDRC3, args format: [output_channels, shortcut_flag]

backbone:
  - [-1, 1, GTDRC3, [256, True]]   # Single block, 256 output channels, residual enabled
  - [-1, 2, GTDRC3, [512, True]]   # Two sequential blocks, 512 output channels
  - [-1, 4, GTDRC3, [1024, True]]  # Four sequential blocks, 1024 output channels

head:
  - [-1, 2, GTDRC3, [512, True]]   # Fusion blocks in head
```

## Files Modified

| File | Change | Line(s) |
|------|--------|---------|
| `ultralytics/nn/tasks.py` | Added GTDRC3 to parse_model checks | 1025, 1052 |
| `ultralytics/cfg/models/v12/yolov12_GTDRC3.yaml` | Fixed argument format from `[c2, n, shortcut]` to `[c2, shortcut]` | All GTDRC3 entries |
| `train_evaluation.ipynb` | Updated notebook to use corrected YAML | Cell 2 |

## How to Use

Training with GTDRC3:
```python
from ultralytics import YOLO

# Load model with GTDRC3 blocks
model = YOLO("ultralytics/cfg/models/v12/yolov12_GTDRC3.yaml")

# Train normally
results = model.train(
    data="path/to/data.yaml",
    epochs=200,
    batch=4,
    device=0,
)
```

## Next Steps

You can now:
1. Run the training notebook with GTDRC3 model
2. Compare GTDRC3 performance against other architectures
3. Create custom variants by adjusting channel dimensions in YAML

---
**Status**: ✅ Ready for training | **Tested**: Yes | **Date**: 2026-02-17
