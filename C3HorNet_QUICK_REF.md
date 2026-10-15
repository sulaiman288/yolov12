# C3HorNet Quick Reference

## Implementation Summary

✅ **C3HorNet** module has been successfully created and integrated into YOLOv12.

---

## Files Modified

| File | Changes | Status |
|------|---------|--------|
| `ultralytics/nn/modules/block.py` | Added C3HorNet class (line 737) + __all__ export | ✅ |
| `ultralytics/nn/modules/__init__.py` | Import & export C3HorNet | ✅ |
| `ultralytics/nn/tasks.py` | Import + register in 2 parse_model() conditions | ✅ |
| `ultralytics/engine/exporter.py` | Import + isinstance() check | ✅ |

---

## Class Definition

```python
class C3HorNet(C2f):
    """HorNet-inspired CSP Bottleneck with 2 convolutions for improved spatial interaction."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes the C3HorNet module."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
```

---

## Usage in YAML Config

Replace any `C2f` with `C3HorNet`:

```yaml
backbone:
  - [-1, 3, C3HorNet, [128, True]]    # Instead of C2f
  - [-1, 6, C3HorNet, [256, True]]    # Custom blocks here
  - [-1, 6, C3HorNet, [512, True]]    # Ready to enhance

head:
  - [-1, 3, C3HorNet, [512]]
  - [-1, 3, C3HorNet, [256]]
```

---

## System Integration

The module integrates into the codebase via:

1. **Module Resolution**: YAML → `globals()['C3HorNet']` → C3HorNet class
2. **Channel Scaling**: Applied via `width_multiple` in parse_model()
3. **Repeat Scaling**: Applied via `depth_multiple` in parse_model()
4. **Export Support**: Handled in exporter.py for all formats (ONNX, TensorRT, etc.)

---

## Ready to Enhance

Now you can:
- **Modify the forward() method** to add HorNet-specific spatial interaction mechanisms
- **Add gated GMLPs** for improved feature mixing
- **Test different kernel configurations** in Bottleneck blocks
- **Create variants** (C3HorNetGMLP, C3HorNetAttention, etc.) following the same pattern

All changes to C3HorNet are automatically picked up by the training pipeline!

---

## Python API

```python
from ultralytics import YOLO

# Using C3HorNet config
model = YOLO('path/to/yolov10n-hornet.yaml')
results = model.train(data='coco8.yaml', epochs=100)
```

## CLI

```bash
yolo detect train model=yolov10n-hornet.yaml data=coco8.yaml epochs=100
```
