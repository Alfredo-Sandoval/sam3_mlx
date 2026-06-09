"""Single-image agent inference compatibility entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from sam3_mlx.agent.agent_core import agent_inference


def run_single_image_inference(
    image_path,
    text_prompt,
    llm_config,
    send_generate_request,
    call_sam_service,
    output_dir="agent_output",
    debug=False,
):
    """Run official-shaped SAM3 agent inference on a single image.

    The orchestration mirrors the official SAM3 agent inference at upstream
    commit ``2814fa619404a722d03e9a012e083e4f293a4e53`` while keeping the SAM
    service and LLM client injectable for the MLX port.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    llm_name = llm_config["name"]
    prompt_for_filename = text_prompt.replace("/", "_").replace(" ", "_")
    base_filename = f"{image_path.stem}_{prompt_for_filename}_agent_{llm_name}"
    output_json_path = output_path / f"{base_filename}_pred.json"
    output_image_path = output_path / f"{base_filename}_pred.png"
    agent_history_path = output_path / f"{base_filename}_history.json"

    if output_json_path.exists():
        return None

    agent_history, final_output_dict, rendered_final_output = agent_inference(
        str(image_path),
        text_prompt,
        send_generate_request=send_generate_request,
        call_sam_service=call_sam_service,
        output_dir=str(output_path),
        debug=debug,
    )

    final_output_dict["text_prompt"] = text_prompt
    final_output_dict["image_path"] = str(image_path)

    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(final_output_dict, handle, indent=4)
    with agent_history_path.open("w", encoding="utf-8") as handle:
        json.dump(agent_history, handle, indent=4)
    rendered_final_output.save(output_image_path)
    return str(output_image_path)
