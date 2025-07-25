# Contributing to AI Photo Editor

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/apathi/ai-photo-editor.git
   cd ai-photo-editor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and single-purpose
- Maximum line length: 100 characters

Example:
```python
def process_image(
    image: Image.Image,
    size: int = 512
) -> Image.Image:
    """
    Resize image to square dimensions.

    Args:
        image: Input PIL Image
        size: Target size in pixels

    Returns:
        Resized PIL Image
    """
    return image.resize((size, size))
```

## Testing

All contributions should include tests:

```bash
# Run tests
python tests/test_pipeline.py

# Add new tests to tests/
# Test naming: test_<functionality>.py
```

## Commit Messages

Use clear, descriptive commit messages:

```
✅ Good:
- "Add support for batch inpainting"
- "Fix memory leak in segmentation pipeline"
- "Improve mask visualization in frontend"

❌ Bad:
- "update code"
- "fix bug"
- "changes"
```

## Pull Request Process

1. **Update documentation** if needed (README, docstrings, comments)
2. **Add tests** for new functionality
3. **Ensure all tests pass** before submitting
4. **Update CHANGELOG** with your changes
5. **Create pull request** with clear description

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests passing
```

## Areas for Contribution

### High Priority
- [ ] Performance optimizations (TensorRT, ONNX conversion)
- [ ] Additional evaluation metrics (FID, LPIPS)
- [ ] Batch processing support
- [ ] Progress callbacks for long operations

### Medium Priority
- [ ] Additional segmentation models (SAM-HQ, FastSAM)
- [ ] Alternative inpainting models (Kandinsky, ControlNet)
- [ ] UI/UX improvements
- [ ] Mobile responsiveness enhancements

### Low Priority
- [ ] Additional themes (dark mode)
- [ ] Export formats (TIFF, WebP)
- [ ] Keyboard shortcuts
- [ ] Undo/redo functionality

## Code Review

All submissions require review. We will:
- Check code quality and style
- Verify tests are comprehensive
- Ensure documentation is complete
- Test functionality

## Questions?

Open an issue for:
- Feature requests
- Bug reports
- Questions about implementation
- Suggestions for improvement

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making AI Photo Editor better! 🎨
