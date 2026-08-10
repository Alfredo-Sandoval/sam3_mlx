# Papercuts

## 2026-08-10T15:51:29.507Z — gpt-5.6-sol — Alfredo Sandoval

While checking references after removing sam3_mlx/agent, an rg pattern containing backticks was interpreted by Bash and flooded the output. Pass regex patterns with shell-safe single quoting.

## 2026-08-10T15:52:30.131Z — gpt-5.6-sol — Alfredo Sandoval

The documented full pytest command cannot collect on this Linux host because the installed MLX package cannot load libmlx.so; this checkout's runtime target is macOS Apple Silicon. Run the MLX suite on its supported host, and keep host-compatible structural checks available for Linux cleanup work.

## 2026-08-10T16:26:51.009Z — gpt-5.6-sol — Alfredo Sandoval

Before publishing the sam3_mlx branch, local Git reported an upstream tracking ref that GitHub no longer had, and a chained divergence command masked the failed targeted fetch. Fetch the remote namespace before trusting stale remote-tracking refs, and keep fetch as a standalone gate.
