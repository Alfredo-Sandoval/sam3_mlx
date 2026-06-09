"""Agent orchestration compatibility helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from sam3_mlx.agent.client_llm import send_generate_request
from sam3_mlx.agent.client_sam3 import call_sam_service
from sam3_mlx.agent.viz import visualize


_SYSTEM_PROMPT_DIR = Path(__file__).resolve().parent / "system_prompts"


def _read_system_prompt(filename: str) -> str:
    path = _SYSTEM_PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing vendored SAM3 agent prompt resource: {path}. "
            "Copy it from third_party/facebook-sam3/sam3/agent/system_prompts."
        )
    return path.read_text(encoding="utf-8").strip()


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=4)


def _call_segment_service(
    service: Callable[..., str],
    *,
    image_path: str,
    text_prompt: str,
    output_folder_path: str,
) -> str:
    try:
        return service(
            image_path=image_path,
            text_prompt=text_prompt,
            output_folder_path=output_folder_path,
        )
    except TypeError as exc:
        raise TypeError(
            "agent_inference expects call_sam_service to accept keyword arguments "
            "image_path, text_prompt, and output_folder_path. For the local MLX "
            "client, pass a bound wrapper such as "
            "lambda **kw: call_sam_service(processor, **kw)."
        ) from exc


def save_debug_messages(messages_list, debug, debug_folder_path, debug_jsonl_path):
    """Save messages to a debug JSONL-like file if debug is enabled."""
    if debug and debug_jsonl_path:
        Path(debug_folder_path).mkdir(parents=True, exist_ok=True)
        with Path(debug_jsonl_path).open("w", encoding="utf-8") as handle:
            for msg in messages_list:
                handle.write(json.dumps(msg, indent=4) + "\n")


def cleanup_debug_files(debug, debug_folder_path, debug_jsonl_path):
    """Clean up debug files when a run successfully returns."""
    if not (debug and debug_folder_path):
        return
    debug_jsonl = Path(debug_jsonl_path) if debug_jsonl_path else None
    if debug_jsonl is not None and debug_jsonl.exists():
        debug_jsonl.unlink()
    debug_folder = Path(debug_folder_path)
    if debug_folder.exists():
        debug_folder.rmdir()


def count_images(messages):
    """Count image content items in a message history."""
    total = 0
    for message in messages:
        if "content" in message and isinstance(message["content"], list):
            total += sum(
                1
                for content_item in message["content"]
                if isinstance(content_item, dict)
                and content_item.get("type") == "image"
            )
    return total


def _prune_messages_for_next_round(
    messages_list,
    used_text_prompts,
    latest_sam3_text_prompt,
    img_path,
    initial_text_prompt,
):
    """Return the official pruned message subset used between agent rounds."""
    if len(messages_list) >= 10:
        raise AssertionError("There should not be more than 10 messages in history")

    part1 = copy.deepcopy(messages_list[:2])
    part2_start_idx = None
    for idx in range(len(messages_list) - 1, 1, -1):
        msg = messages_list[idx]
        if msg.get("role") != "assistant" or "content" not in msg:
            continue
        for content in msg["content"]:
            if (
                isinstance(content, dict)
                and content.get("type") == "text"
                and "<tool>" in content.get("text", "")
                and "segment_phrase" in content.get("text", "")
            ):
                part2_start_idx = idx
                break
        if part2_start_idx is not None:
            break

    part2 = messages_list[part2_start_idx:] if part2_start_idx is not None else []
    previously_used = (
        [p for p in used_text_prompts if p != latest_sam3_text_prompt]
        if latest_sam3_text_prompt
        else list(used_text_prompts)
    )
    if part2 and len(previously_used) > 0:
        warning_text = (
            "Note that we have previously called the segment_phrase tool with each "
            f'"text_prompt" in this list: {list(previously_used)}, but none of the '
            "generated results were satisfactory. So make sure that you do not use "
            'any of these phrases as the "text_prompt" to call the segment_phrase tool again.'
        )
        part1[1] = {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": (
                        f"The above image is the raw input image. The initial user input query is: "
                        f"'{initial_text_prompt}'. {warning_text}"
                    ),
                },
            ],
        }
    return [*part1, *part2]


def agent_inference(
    img_path: str,
    initial_text_prompt: str,
    debug: bool = False,
    send_generate_request=send_generate_request,
    call_sam_service=call_sam_service,
    max_generations: int = 100,
    output_dir="../../sam3_agent_out",
):
    """
    Run the official single-image SAM3 agent loop with MLX-local SAM calls.

    The control flow is ported from the official SAM3 agent core at upstream
    commit ``2814fa619404a722d03e9a012e083e4f293a4e53``. The SAM service callable is intentionally
    injectable so callers can bind a local MLX processor while retaining the
    official tool-call protocol.
    """
    output_root = Path(output_dir)
    sam_output_dir = output_root / "sam_out"
    error_save_dir = output_root / "none_out"
    debug_save_dir = output_root / "agent_debug_out"
    sam_output_dir.mkdir(parents=True, exist_ok=True)
    error_save_dir.mkdir(parents=True, exist_ok=True)
    debug_save_dir.mkdir(parents=True, exist_ok=True)

    path_to_latest_output_json = ""
    latest_sam3_text_prompt = ""
    used_text_prompts: set[str] = set()
    generation_count = 0

    debug_folder_path = None
    debug_jsonl_path = None
    if debug:
        debug_folder_path = debug_save_dir / Path(img_path).stem
        debug_jsonl_path = debug_folder_path / "debug_history.json"
        debug_folder_path.mkdir(parents=True, exist_ok=True)

    system_prompt = _read_system_prompt("system_prompt.txt")
    iterative_checking_system_prompt = _read_system_prompt(
        "system_prompt_iterative_checking.txt"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": (
                        "The above image is the raw input image. "
                        f"The initial user input query is: '{initial_text_prompt}'."
                    ),
                },
            ],
        },
    ]

    generated_text = send_generate_request(messages)
    while generated_text is not None:
        save_debug_messages(messages, debug, debug_folder_path, debug_jsonl_path)
        if "<tool>" not in generated_text:
            raise ValueError(
                f"Generated text does not contain <tool> tag: {generated_text}"
            )
        generated_text = generated_text.split("</tool>", 1)[0] + "</tool>"
        tool_call_json_str = (
            generated_text.split("<tool>")[-1]
            .split("</tool>")[0]
            .strip()
            .replace(r"}}}", r"}}")
        )
        try:
            tool_call = json.loads(tool_call_json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in tool call: {tool_call_json_str}"
            ) from exc

        tool_name = tool_call["name"]
        if path_to_latest_output_json == "" and tool_name not in {
            "segment_phrase",
            "report_no_mask",
        }:
            raise AssertionError(
                "The first tool call must be segment_phrase or report_no_mask"
            )

        if tool_name == "segment_phrase":
            if list(tool_call["parameters"].keys()) != ["text_prompt"]:
                raise ValueError("segment_phrase expects exactly text_prompt")

            current_text_prompt = tool_call["parameters"]["text_prompt"]
            if current_text_prompt in used_text_prompts:
                duplicate_prompt_message = (
                    f"You have previously used '{current_text_prompt}' as your "
                    "text_prompt to call the segment_phrase tool. You may not use it "
                    "again. Please call the segment_phrase tool again with a different, "
                    "perhaps more general, or more creative simple noun phrase prompt, "
                    "while adhering to all the rules stated in the system prompt. You "
                    "must also never use any of the following text_prompt(s): "
                    f"{str(list(used_text_prompts))}."
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": generated_text}],
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": duplicate_prompt_message}],
                    }
                )
            else:
                used_text_prompts.add(current_text_prompt)
                latest_sam3_text_prompt = current_text_prompt
                path_to_latest_output_json = _call_segment_service(
                    call_sam_service,
                    image_path=img_path,
                    text_prompt=current_text_prompt,
                    output_folder_path=str(sam_output_dir),
                )
                sam3_outputs = _read_json(path_to_latest_output_json)
                sam3_output_image_path = sam3_outputs["output_image_path"]
                num_masks = len(sam3_outputs["pred_boxes"])

                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": generated_text}],
                    }
                )
                if num_masks == 0:
                    sam3_output_text_message = (
                        "The segment_phrase tool did not generate any masks for the "
                        f"text_prompt '{current_text_prompt}'. Now, please call the "
                        "segment_phrase tool again with a different, perhaps more "
                        "general, or more creative simple noun phrase text_prompt, "
                        "while adhering to all the rules stated in the system prompt. "
                        f"Please be reminded that the original user query was "
                        f"'{initial_text_prompt}'."
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": sam3_output_text_message}
                            ],
                        }
                    )
                else:
                    sam3_output_text_message = (
                        f"The segment_phrase tool generated {num_masks} available "
                        f"masks. All {num_masks} available masks are rendered in this "
                        f"image below, now you must analyze the {num_masks} available "
                        "mask(s) carefully, compare them against the raw input image "
                        "and the original user query, and determine your next action. "
                        f"Please be reminded that the original user query was "
                        f"'{initial_text_prompt}'."
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": sam3_output_text_message},
                                {"type": "image", "image": sam3_output_image_path},
                            ],
                        }
                    )

        elif tool_name == "examine_each_mask":
            if latest_sam3_text_prompt == "":
                raise AssertionError("examine_each_mask requires a prior SAM3 prompt")
            if messages[-1]["content"][1]["type"] != "image":
                raise AssertionError("Second content element should be an image")
            messages.pop()
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "The segment_phrase tool generated several masks. Now "
                                "you must analyze the mask(s) carefully, compare them "
                                "against the raw input image and the original user "
                                "query, and determine your next action."
                            ),
                        }
                    ],
                }
            )

            current_outputs = _read_json(path_to_latest_output_json)
            num_masks = len(current_outputs["pred_masks"])
            masks_to_keep: list[int] = []
            prompt_safe = latest_sam3_text_prompt.replace("/", "_")

            for index in range(num_masks):
                image_w_mask_i, image_w_zoomed_in_mask_i = visualize(
                    current_outputs,
                    index,
                )
                image_w_zoomed_in_mask_i_path = (
                    sam_output_dir / f"{prompt_safe}_zoom_in_mask_{index + 1}.png"
                )
                image_w_mask_i_path = (
                    sam_output_dir / f"{prompt_safe}_selected_mask_{index + 1}.png"
                )
                image_w_zoomed_in_mask_i.save(image_w_zoomed_in_mask_i_path)
                image_w_mask_i.save(image_w_mask_i_path)

                iterative_checking_messages = [
                    {"role": "system", "content": iterative_checking_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "The raw input image: "},
                            {"type": "image", "image": img_path},
                            {
                                "type": "text",
                                "text": (
                                    "The initial user input query is: "
                                    f"'{initial_text_prompt}'"
                                ),
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Image with the predicted segmentation mask "
                                    "rendered on it: "
                                ),
                            },
                            {"type": "image", "image": str(image_w_mask_i_path)},
                            {"type": "text", "text": "Image with the zoomed-in mask: "},
                            {
                                "type": "image",
                                "image": str(image_w_zoomed_in_mask_i_path),
                            },
                        ],
                    },
                ]
                checking_generated_text = send_generate_request(
                    iterative_checking_messages
                )
                if checking_generated_text is None:
                    raise ValueError(
                        "Generated text is None during mask checking. Check the LLM "
                        "server and input parameters."
                    )
                verdict = (
                    checking_generated_text.split("<verdict>")[-1]
                    .split("</verdict>")[0]
                    .strip()
                )
                if "Accept" in verdict:
                    if "Reject" in verdict:
                        raise ValueError(
                            f"Ambiguous verdict in generated text: {checking_generated_text}"
                        )
                    masks_to_keep.append(index)
                elif "Reject" in verdict:
                    if "Accept" in verdict:
                        raise ValueError(
                            f"Ambiguous verdict in generated text: {checking_generated_text}"
                        )
                else:
                    raise ValueError(
                        f"Unexpected verdict in generated text: {checking_generated_text}. "
                        "Expected 'Accept' or 'Reject'."
                    )

            updated_outputs = {
                "original_image_path": current_outputs["original_image_path"],
                "orig_img_h": current_outputs["orig_img_h"],
                "orig_img_w": current_outputs["orig_img_w"],
                "pred_boxes": [current_outputs["pred_boxes"][i] for i in masks_to_keep],
                "pred_scores": [
                    current_outputs["pred_scores"][i] for i in masks_to_keep
                ],
                "pred_masks": [current_outputs["pred_masks"][i] for i in masks_to_keep],
            }

            image_w_check_masks = visualize(updated_outputs)
            kept_suffix = (
                "none"
                if len(masks_to_keep) == 0
                else "_".join(str(i + 1) for i in masks_to_keep)
            )
            image_w_check_masks_path = (
                sam_output_dir / f"{prompt_safe}_selected_masks_{kept_suffix}.png"
            )
            image_w_check_masks.save(image_w_check_masks_path)

            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text}],
                }
            )
            if len(masks_to_keep) == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"The original user query was: '{initial_text_prompt}'. "
                                    "The examine_each_mask tool examined and rejected all "
                                    "of the masks generated by the segment_phrase tool. "
                                    "Now, please call the segment_phrase tool again with a "
                                    "different, perhaps more general, or more creative "
                                    "simple noun phrase text_prompt, while adhering to all "
                                    "the rules stated in the system prompt."
                                ),
                            }
                        ],
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"The original user query was: '{initial_text_prompt}'. "
                                    "After calling the examine_each_mask tool on the "
                                    "available masks, the number of available masks is "
                                    f"now {len(masks_to_keep)}. All {len(masks_to_keep)} "
                                    "available masks are rendered in this image below, now "
                                    f"you must analyze the {len(masks_to_keep)} available "
                                    "mask(s) carefully, compare them against the raw input "
                                    "image and the original user query, and determine your "
                                    "next action."
                                ),
                            },
                            {"type": "image", "image": str(image_w_check_masks_path)},
                        ],
                    }
                )

            base_path = Path(path_to_latest_output_json)
            if "masks_" in str(base_path):
                base_path = Path(str(base_path).split("masks_")[0] + ".json")
            if len(masks_to_keep) == 0:
                path_to_latest_output_json = str(
                    base_path.with_name(base_path.stem + "masks_none.json")
                )
            else:
                path_to_latest_output_json = str(
                    base_path.with_name(
                        base_path.stem
                        + f"masks_{'_'.join(map(str, masks_to_keep))}.json"
                    )
                )
            _write_json(path_to_latest_output_json, updated_outputs)

        elif tool_name == "select_masks_and_return":
            current_outputs = _read_json(path_to_latest_output_json)
            if list(tool_call["parameters"].keys()) != ["final_answer_masks"]:
                raise ValueError(
                    "select_masks_and_return expects exactly final_answer_masks"
                )
            available_masks = set(range(1, len(current_outputs["pred_masks"]) + 1))
            masks_to_keep = sorted(
                {
                    i
                    for i in tool_call["parameters"]["final_answer_masks"]
                    if i in available_masks
                }
            )
            final_outputs = {
                "original_image_path": current_outputs["original_image_path"],
                "orig_img_h": current_outputs["orig_img_h"],
                "orig_img_w": current_outputs["orig_img_w"],
                "pred_boxes": [
                    current_outputs["pred_boxes"][i - 1] for i in masks_to_keep
                ],
                "pred_scores": [
                    current_outputs["pred_scores"][i - 1] for i in masks_to_keep
                ],
                "pred_masks": [
                    current_outputs["pred_masks"][i - 1] for i in masks_to_keep
                ],
            }
            rendered_final_output = visualize(final_outputs)
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text}],
                }
            )
            cleanup_debug_files(debug, debug_folder_path, debug_jsonl_path)
            return messages, final_outputs, rendered_final_output

        elif tool_name == "report_no_mask":
            with Image.open(img_path) as image:
                width, height = image.size
                rendered_final_output = image.convert("RGB").copy()
            final_outputs = {
                "original_image_path": img_path,
                "orig_img_h": height,
                "orig_img_w": width,
                "pred_boxes": [],
                "pred_scores": [],
                "pred_masks": [],
            }
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": generated_text}],
                }
            )
            cleanup_debug_files(debug, debug_folder_path, debug_jsonl_path)
            return messages, final_outputs, rendered_final_output

        else:
            raise ValueError(f"Unknown tool call: {tool_name}")

        for message in messages:
            if message["role"] == "assistant" and "content" in message:
                for content in message["content"]:
                    if (
                        isinstance(content, dict)
                        and content.get("type") == "text"
                        and "text" in content
                    ):
                        content["text"] = (
                            content["text"].split("</tool>", 1)[0] + "</tool>\n\n"
                        )

        messages = _prune_messages_for_next_round(
            messages,
            used_text_prompts,
            latest_sam3_text_prompt,
            img_path,
            initial_text_prompt,
        )
        if count_images(messages) > 2:
            raise AssertionError("There can never be more than 2 images in context")
        generation_count += 1
        if generation_count > max_generations:
            raise ValueError(
                f"Exceeded maximum number of allowed generation requests ({max_generations})"
            )
        generated_text = send_generate_request(messages)

    error_save_path = error_save_dir / f"{Path(img_path).stem}_error_history.json"
    _write_json(error_save_path, messages)
    raise ValueError(
        "Generated text is None, which is unexpected. Check the LLM server and the "
        f"input parameters for image path: {img_path} and initial text prompt: "
        f"{initial_text_prompt}."
    )
