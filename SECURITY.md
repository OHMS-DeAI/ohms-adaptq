# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Security Considerations

### Quantization Integrity
- All artifacts include cryptographic hashes (SHA-256)
- Deterministic output ensures reproducibility
- No external network access during quantization
- Input validation on all model files

### Model Supply Chain
- Verify HuggingFace model checksums before processing
- Log all quantization parameters for audit trail
- Sandbox quantization environment
- No arbitrary code execution from model files

### Artifact Security
- Immutable artifacts after generation
- Hash verification at every stage
- Secure artifact storage recommendations
- Version pinning for dependencies

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability:

1. **Do not** open a public issue
2. Email: security@ohms-project.org
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Process
- **24 hours**: Acknowledgment of report
- **48 hours**: Initial assessment
- **7 days**: Detailed response with fix timeline
- **30 days**: Public disclosure (coordinated)

## Security Best Practices

### For Users
- Verify artifact hashes before use
- Use official APQ releases only
- Keep dependencies updated
- Validate model sources

### For Developers
- Follow secure coding practices
- No unsafe Rust blocks without review
- Validate all inputs
- Use seeded randomness only
- Hash all outputs

### For Auditors
- Independent verification of determinism
- Validate hash consistency
- Check for information leakage
- Verify no unauthorized network access

## Dependencies

APQ maintains a minimal dependency tree:
- Regular security audits of dependencies
- Pinned versions in Cargo.lock
- Automated vulnerability scanning
- Quick patching process for critical issues

## Compliance

APQ is designed to support:
- Reproducible builds
- Audit trail requirements
- Air-gapped environments
- Deterministic execution

## Contact

For security questions or concerns:
- Email: security@ohms-project.org
- GPG Key: [Available on request]

Thank you for helping keep OHMS secure!