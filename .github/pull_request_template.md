## Description

Brief description of changes made.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## APQ-Specific Checklist

- [ ] Quantization changes maintain determinism
- [ ] Artifacts include correct hashes
- [ ] Model accuracy within acceptable thresholds
- [ ] Chunk sizes remain ≤2 MiB
- [ ] Manifest schema compatibility maintained
- [ ] Verification status logic updated if needed

## Testing

- [ ] Unit tests added/updated and passing
- [ ] Integration tests passing
- [ ] Golden model comparisons updated if needed
- [ ] Determinism verified across multiple runs
- [ ] Performance impact measured and documented

## Code Quality

- [ ] `cargo fmt` applied
- [ ] `cargo clippy` passes without warnings
- [ ] Documentation updated (rustdoc)
- [ ] CHANGELOG.md updated if applicable
- [ ] Breaking changes noted in commit message

## Security

- [ ] No network access added to quantization pipeline
- [ ] Input validation added where appropriate
- [ ] No unsafe code blocks (or justified with comments)
- [ ] Cryptographic hashes verified
- [ ] No secrets or credentials in code

## Artifacts

If this change affects artifact format:

- [ ] Schema version updated
- [ ] Backward compatibility considered
- [ ] Migration path documented
- [ ] Test artifacts updated

## Performance

- [ ] Performance impact assessed
- [ ] Memory usage within bounds
- [ ] No performance regressions
- [ ] Benchmarks updated if needed

## Additional Notes

Add any additional context, screenshots, or links that reviewers should know about.

## Review Checklist (for reviewers)

- [ ] Code follows project standards
- [ ] Tests are comprehensive
- [ ] Documentation is clear
- [ ] Security implications considered
- [ ] Performance impact acceptable
- [ ] Artifacts remain verifiable