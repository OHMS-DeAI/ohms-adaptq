# Contributing to OHMS-AdaptQ

Thank you for your interest in contributing to OHMS-AdaptQ! This document provides guidelines for contributing to the Adaptive Progressive Quantization engine.

## Development Setup

### Prerequisites
- Rust 1.70+
- Python 3.8+ (for HuggingFace integration)
- Git

### Setup
```bash
git clone <repo-url>
cd ohms-adaptq
cargo build
cargo test
```

## Code Guidelines

### Rust Standards
- Use `cargo fmt` for formatting
- Run `cargo clippy` for linting  
- Maintain deterministic behavior (no system randomness)
- Document public APIs with rustdoc
- Add unit tests for quantization algorithms

### Determinism Requirements
- All randomness must be seeded and reproducible
- Floating point operations should be stable across platforms
- Hash calculations must be consistent
- No system-dependent behavior

### Testing
- Unit tests for quantization functions
- Integration tests for full APQ pipeline
- Golden model comparisons for accuracy
- Determinism verification across runs

## Artifact Standards

### Manifest Schema
- Follow the JSON schema in `schemas/manifest.schema.json`
- Include all required fields
- Validate hashes match actual shard contents

### Verification Reports
- Include OVERALL_STATUS field (PASS/FAIL/WARNING)
- Document per-layer error thresholds
- Provide perplexity comparisons vs FP16 baseline

### Sharding
- Maintain ≤2 MiB chunk size limit
- Ensure chunks can be loaded independently
- Include chunk metadata in manifest

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes following the guidelines above
4. Add/update tests as needed
5. Ensure all tests pass (`cargo test`)
6. Run formatting and linting (`cargo fmt && cargo clippy`)
7. Commit with conventional commit format
8. Push to your fork and create a pull request

### Commit Format
```
type(scope): description

Examples:
feat(quantization): add 3-bit dynamic allocation
fix(manifest): correct chunk hash validation
docs(readme): update APQ method description
```

## Review Criteria

- Code follows Rust best practices
- Maintains deterministic behavior
- Includes appropriate tests
- Documentation is clear and complete
- No breaking changes to artifact format without versioning
- Performance impact is considered and documented

## Model Support

When adding support for new model architectures:

1. Ensure compatibility with HuggingFace transformers
2. Add architecture-specific calibration logic
3. Update documentation with supported models
4. Include test cases with known good outputs
5. Validate quantization quality meets thresholds

## Security Considerations

- No network access during quantization
- Validate all inputs before processing
- Use safe Rust practices (avoid unsafe blocks)
- Hash all artifacts for integrity
- Document any security assumptions

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- General questions

For security issues, please follow responsible disclosure practices.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).