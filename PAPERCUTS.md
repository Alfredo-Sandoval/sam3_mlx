# Papercuts

## 2026-08-10T15:51:29.507Z — gpt-5.6-sol — Alfredo Sandoval

While checking references after removing sam3_mlx/agent, an rg pattern containing backticks was interpreted by Bash and flooded the output. Pass regex patterns with shell-safe single quoting.

## 2026-08-10T15:52:30.131Z — gpt-5.6-sol — Alfredo Sandoval

The documented full pytest command cannot collect on this Linux host because the installed MLX package cannot load libmlx.so; this checkout's runtime target is macOS Apple Silicon. Run the MLX suite on its supported host, and keep host-compatible structural checks available for Linux cleanup work.
