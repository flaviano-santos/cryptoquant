# Security policy

## Reporting a vulnerability

Please do not open a public issue for security problems, especially anything
involving credentials. Email the maintainer directly.

## Handling API keys

This project never reads credentials from configuration files. Keys come from
environment variables only:

```bash
export CQ_API_KEY=...
export CQ_API_SECRET=...
```

When creating exchange API keys:

- Enable **trading only**. Disable withdrawals.
- Restrict to a specific IP address.
- Use a separate key for testnet and live.
- Rotate them periodically, and immediately if a machine is compromised.

`.gitignore` excludes `.env`, `*.key`, `*.pem` and common service-account
filename patterns. The `detect-private-key` pre-commit hook is a second line of
defence. Neither replaces checking `git diff --staged` before you commit.

## Trading safety defaults

`config.yaml` ships with `live.testnet: true` and `live.dry_run: true`. Both
must be changed deliberately. The kill switch in `cryptoquant.trading.risk`
flattens positions and halts on a daily-loss or drawdown breach, and requires a
manual reset. Do not remove these guards to make testing more convenient.
