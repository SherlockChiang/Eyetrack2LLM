# Security Policy

## Reporting

Do not open a public issue containing credentials, tokens, unpublished participant-level data, restricted corpus files, or exploitable security details.

Use GitHub private vulnerability reporting when it is enabled for this repository. If it is unavailable, contact the repository owner privately through the owner's GitHub profile. Include reproduction steps, affected versions, impact, and a minimal proof of concept, but do not attach participant data or secrets.

For non-sensitive bugs, use the public bug-report template.

Release candidates are size- and secret-scanned by `python scripts/check_release.py --public`. This check reduces accidental disclosure risk but is not a substitute for reviewing `PUBLIC_FILES.txt` before staging.
