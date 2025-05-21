# agentictest/ClientAgenticArchitecture/worker.py
import logging
import time
import base64
import os
from io import BytesIO
from typing import List, Optional, Dict, Tuple, Any
from PIL import Image
import openai # For VLM client
import re # For string manipulation

from .utils.logging_utils import setup_logger
from .utils.screenshot import capture_screenshot, detect_primary_screen_size
from .utils.parser import parse_llm_output

FINISH_WORD = "finished"
WAIT_WORD = "wait"
CALL_USER = "call_user"

WORKER_ACTION_SPACE = """
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s (as per your action_executor) and take a screenshot.
finished() # Use this when the current task is complete.
"""

WORKER_USER_PROMPT_TEMPLATE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
Thought: ...
Action: ...

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{sub_task_instruction}
"""
DEFAULT_WORKER_HISTORY_N = 5
DEFAULT_WORKER_VLM_MODEL = "ui-tars"



class Worker:
    def __init__(
        self,
        ui_tars_server_url: str,
        ui_tars_api_key: str = "empty",
        primary_screen_size: Optional[Tuple[int, int]] = None,
        logger: Optional[logging.Logger] = None,
        language: str = "English",
        worker_history_n: int = DEFAULT_WORKER_HISTORY_N,
        ui_tars_model_name: str = DEFAULT_WORKER_VLM_MODEL
    ):
        self.logger = logger or setup_logger("Worker", logging.INFO)
        self.language = language
        self.worker_history_n = worker_history_n
        self.ui_tars_model_name = ui_tars_model_name
        self.logger.info(f"Worker using VLM model: {self.ui_tars_model_name}")

        self.vlm_base_url = f"{ui_tars_server_url.rstrip('/')}/v1"
        try:
            self.vlm_client = openai.OpenAI(
                base_url=self.vlm_base_url,
                api_key=ui_tars_api_key,
            )
        except Exception as e:
            self.logger.critical(f"Failed to initialize OpenAI client for Worker VLM: {e}", exc_info=True)
            raise

        if primary_screen_size:
            self.primary_width, self.primary_height = primary_screen_size
        else:
            try:
                self.primary_width, self.primary_height = detect_primary_screen_size()
            except Exception as e:
                self.logger.warning(f"Failed to auto-detect screen size, defaulting to 1920x1080. Error: {e}")
                self.primary_width, self.primary_height = 1920, 1080
        self.logger.info(f"Worker using primary screen size: {self.primary_width}x{self.primary_height}")
        self.action_space_prompt_segment = WORKER_ACTION_SPACE

    def _pil_image_to_bytes(self, image: Image.Image) -> bytes:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return buffered.getvalue()

    def _escape_string_for_pyautogui_code(self, text_to_escape: Optional[str]) -> str:
        if text_to_escape is None: return ""
        return text_to_escape.replace("\\", "\\\\").replace("'", "\\'")

    def _generate_pyautogui_command_string(
        self,
        parsed_action_item: Dict,
        extracted_thought: Optional[str],
        overall_plan: Optional[List[str]] = None,
        current_sub_task_idx_in_plan: Optional[int] = None
    ) -> str:
        action_type = parsed_action_item.get("type", "unknown").lower()
        py_commands = [] 
        action_details_for_log = [f"ActionType='{action_type}'"]

        def to_pixels(vlm_coords_0_1000: Optional[List[Any]]) -> Optional[Tuple[int, int]]:
            if vlm_coords_0_1000 and isinstance(vlm_coords_0_1000, list) and len(vlm_coords_0_1000) == 2:
                try:
                    x_norm, y_norm = float(vlm_coords_0_1000[0]), float(vlm_coords_0_1000[1])
                    abs_x = int((x_norm / 1000.0) * self.primary_width)
                    abs_y = int((y_norm / 1000.0) * self.primary_height)
                    abs_x = max(0, min(abs_x, self.primary_width - 1))
                    abs_y = max(0, min(abs_y, self.primary_height - 1))
                    return abs_x, abs_y
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Could not parse/convert VLM coordinates: {vlm_coords_0_1000}. Error: {e}")
            return None

        if action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            coords_0_1000 = parsed_action_item.get("coordinates")
            pixel_coords = to_pixels(coords_0_1000)
            if pixel_coords:
                x, y = pixel_coords
                action_details_for_log.append(f"TargetPx=({x},{y}) from VLMCoords={coords_0_1000}")
                py_commands.append(f"pyautogui.moveTo({x}, {y}, duration=0.2)")
                if action_type in ["click", "left_single"]: py_commands.append(f"pyautogui.click(button='left')")
                elif action_type == "left_double": py_commands.append(f"pyautogui.doubleClick(button='left')")
                elif action_type == "right_single": py_commands.append(f"pyautogui.click(button='right')")
            else:
                py_commands.append(f"# CodeGen Error: Missing/invalid coordinates for {action_type}. Data: {parsed_action_item.get('coordinates')}")
        
        elif action_type == "type":
            content = parsed_action_item.get("content", "") 
            action_details_for_log.append(f"Content(len={len(content)})='{content[:30].replace(chr(10), '/n')}...'")
            escaped_content_for_code = self._escape_string_for_pyautogui_code(content)
            py_commands.append(f"pyautogui.write('{escaped_content_for_code}', interval=0.02)")
            if '\n' in content: 
                 py_commands.append("pyautogui.press('enter')")
        
        elif action_type == "hotkey":
            key_combination = parsed_action_item.get("key")
            action_details_for_log.append(f"Keys='{key_combination}'")
            if key_combination and isinstance(key_combination, str) and key_combination.strip():
                keys_as_list_str = ", ".join([f"'{k.strip().lower()}'" for k in key_combination.split('+')])
                py_commands.append(f"pyautogui.hotkey({keys_as_list_str})")
            else:
                py_commands.append(f"# CodeGen Error: Missing or invalid key for hotkey. Data: {key_combination}")

        elif action_type == "drag":
            start_coords_0_1000 = parsed_action_item.get("start_coordinates")
            end_coords_0_1000 = parsed_action_item.get("end_coordinates")
            start_pixel_coords = to_pixels(start_coords_0_1000)
            end_pixel_coords = to_pixels(end_coords_0_1000)
            if start_pixel_coords and end_pixel_coords:
                sx, sy = start_pixel_coords; ex, ey = end_pixel_coords
                action_details_for_log.append(f"StartPx=({sx},{sy}), EndPx=({ex},{ey}) from VLM S={start_coords_0_1000}, E={end_coords_0_1000}")
                py_commands.append(f"pyautogui.moveTo({sx}, {sy}, duration=0.2)")
                py_commands.append(f"pyautogui.dragTo({ex}, {ey}, duration=0.5, button='left')")
            else:
                py_commands.append(f"# CodeGen Error: Missing coords for drag. S:{start_coords_0_1000}, E:{end_coords_0_1000}")

        elif action_type == "scroll":
            direction = str(parsed_action_item.get("direction", "")).lower()
            coords_0_1000 = parsed_action_item.get("coordinates")
            pixel_coords = to_pixels(coords_0_1000)
            scroll_clicks = 5 
            scroll_amount_per_click = 40 
            total_scroll = scroll_clicks * scroll_amount_per_click
            action_details_for_log.append(f"Direction='{direction}'")
            scroll_target_args_list = []
            if pixel_coords:
                x,y = pixel_coords
                py_commands.append(f"pyautogui.moveTo({x}, {y}, duration=0.2)")
                action_details_for_log.append(f"TargetPx=({x},{y}) from VLMCoords={coords_0_1000}")
                scroll_target_args_list.extend([f"x={x}", f"y={y}"])
            scroll_target_args_str = (", " + ", ".join(scroll_target_args_list)) if scroll_target_args_list else ""
            if "up" in direction: py_commands.append(f"pyautogui.scroll({total_scroll}{scroll_target_args_str})")
            elif "down" in direction: py_commands.append(f"pyautogui.scroll({-total_scroll}{scroll_target_args_str})")
            elif "left" in direction: py_commands.append(f"pyautogui.hscroll({-total_scroll}{scroll_target_args_str})")
            elif "right" in direction: py_commands.append(f"pyautogui.hscroll({total_scroll}{scroll_target_args_str})")
            else: py_commands.append(f"# CodeGen Error: Invalid scroll direction. Data: {direction}")
        
        elif action_type == WAIT_WORD:
            py_commands.append("time.sleep(5) # Worker decided to wait")
            action_details_for_log.append("Wait=5s")
        elif action_type == FINISH_WORD:
            py_commands.append(f"# Action: {FINISH_WORD} (Worker signals sub-task completion to coordinator)")
            action_details_for_log.append("SignalToCoordinator=finished_sub_task")
        elif action_type == CALL_USER:
            py_commands.append(f"# Action: {CALL_USER} (Worker signals need for help to coordinator)")
            action_details_for_log.append("SignalToCoordinator=call_user")
        elif action_type.startswith("fail_"): 
             py_commands.append(f"# Action: {action_type.upper()} (Internal Worker/VLM issue or Planner failure)")
             action_details_for_log.append(f"SignalToCoordinator={action_type}")
        else:
            py_commands.append(f"# CodeGen Warning: Unhandled action type '{action_type}'. Parsed: {parsed_action_item}")

        full_command_string = "import pyautogui\nimport time\n\n"
        full_command_string += "'''\n--- Current Overall Plan (Coordinator Perspective) ---\n"
        if overall_plan:
            for i, step_desc in enumerate(overall_plan):
                prefix = ">>> " if current_sub_task_idx_in_plan is not None and i == current_sub_task_idx_in_plan else "    "
                plan_step_cleaned = step_desc.strip().replace('\n', ' ').replace("'''", "' ' '")
                full_command_string += f"{prefix}{i+1}. {plan_step_cleaned}\n"
        else:
            full_command_string += "    No overall plan context provided to worker for this step.\n"
        full_command_string += "'''\n\n"
        full_command_string += f"'''\n--- Worker Thought for this Action Set ---\n"
        full_command_string += f"{extracted_thought or 'No specific worker thought extracted for this action set.'}\n"
        full_command_string += f"--- Action Details Log ---\n"
        full_command_string += f"{'; '.join(action_details_for_log)}\n"
        full_command_string += f"Raw Parsed Action from VLM: {parsed_action_item}\n'''\n\n"
        if py_commands:
            full_command_string += "\n".join(py_commands)
        else:
            full_command_string += "# No explicit pyautogui commands generated for this step.\n"
        full_command_string += "\ntime.sleep(0.5) # Default small pause after action(s) for UI stabilization\n"
        return full_command_string.strip()

    def execute_one_vlm_step_for_sub_task(
        self,
        sub_task_description: str,
        current_screenshot_pil: Optional[Image.Image],
        vlm_responses_history_for_subtask: List[str], # History of assistant responses BEFORE this step
        images_history_bytes_for_subtask: List[bytes],  # History of image_bytes BEFORE this step (not including current_screenshot_pil)
        overall_plan: Optional[List[str]] = None,
        current_sub_task_idx_in_plan: Optional[int] = None
    ) -> Dict[str, Any]:
        self.logger.info(f"Worker: Executing ONE VLM step for sub-task: '{sub_task_description[:100]}...'")

        if not current_screenshot_pil:
            self.logger.error("Worker: Missing current_screenshot_pil for VLM step.")
            fail_thought = "Error: Missing screenshot for worker VLM step."
            fail_action_str = self._generate_pyautogui_command_string(
                {"type":"FAIL_INTERNAL", "details": "Missing screenshot"},
                fail_thought, overall_plan, current_sub_task_idx_in_plan)
            return {"error": "Worker: Missing current screenshot.", "vlm_response_raw": None,
                    "pyautogui_actions_strings": [fail_action_str], "extracted_thought": fail_thought,
                    "updated_vlm_responses_history": vlm_responses_history_for_subtask,
                    "updated_images_history_bytes": images_history_bytes_for_subtask,
                    "signaled_finish_subtask": False, "signaled_call_user": False}
        try:
            current_screenshot_bytes = self._pil_image_to_bytes(current_screenshot_pil)
        except Exception as e: 
            self.logger.exception("Worker: Failed to convert PIL image to bytes.")
            fail_thought = f"Error processing screenshot: {e}"
            fail_action_str = self._generate_pyautogui_command_string(
                {"type":"FAIL_INTERNAL", "details": f"Screenshot processing error: {e}"},
                fail_thought, overall_plan, current_sub_task_idx_in_plan)
            return {"error": f"Worker: Failed to process screenshot: {e}", "vlm_response_raw": None,
                    "pyautogui_actions_strings": [fail_action_str], "extracted_thought": fail_thought,
                    "updated_vlm_responses_history": vlm_responses_history_for_subtask,
                    "updated_images_history_bytes": images_history_bytes_for_subtask,
                    "signaled_finish_subtask": False, "signaled_call_user": False}

        # --- History Preparation ---
        # images_for_vlm_call will contain historical images + the current one.
        images_for_vlm_call = list(images_history_bytes_for_subtask)
        images_for_vlm_call.append(current_screenshot_bytes)

        # Create a mutable copy of incoming response history for potential trimming
        responses_for_vlm_call_history = list(vlm_responses_history_for_subtask)

        # Trim if history + current image exceeds worker_history_n for images
        if len(images_for_vlm_call) > self.worker_history_n:
            trim_count = len(images_for_vlm_call) - self.worker_history_n
            images_for_vlm_call = images_for_vlm_call[trim_count:]
            # Trim corresponding responses from the front
            if len(responses_for_vlm_call_history) >= trim_count:
                responses_for_vlm_call_history = responses_for_vlm_call_history[trim_count:]
            else: # if history is consistent
                self.logger.warning("Response history shorter than image history trim count during trimming.")
                responses_for_vlm_call_history = []
        
        # Ensure responses_for_vlm_call_history matches the number of images it should correspond to
        expected_response_count = max(0, len(images_for_vlm_call) - 1)
        if len(responses_for_vlm_call_history) > expected_response_count:
            self.logger.warning(f"Response history ({len(responses_for_vlm_call_history)}) too long for image history ({len(images_for_vlm_call)}), trimming responses.")
            responses_for_vlm_call_history = responses_for_vlm_call_history[-expected_response_count:] # Keep latest
        elif len(responses_for_vlm_call_history) < expected_response_count and images_history_bytes_for_subtask: # only warn if there was incoming history
             self.logger.warning(f"Response history ({len(responses_for_vlm_call_history)}) shorter than expected ({expected_response_count}) for image history ({len(images_for_vlm_call)}). Some images may lack paired responses.")


        # --- Prompt Construction  ---
        full_text_prompt_for_vlm = WORKER_USER_PROMPT_TEMPLATE.format(
            action_space=self.action_space_prompt_segment,
            language=self.language,
            sub_task_instruction=sub_task_description
        )

        # --- Message Assembly for API ---
        messages_for_api: List[Dict[str, Any]] = []

        for i, img_bytes_hist in enumerate(images_for_vlm_call): # Iterates through all images to be sent
            base64_encoded_img = base64.b64encode(img_bytes_hist).decode("utf-8")
            user_turn_content: List[Dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_encoded_img}"}}
            ]

            # Add the full text prompt with the OLDEST image in the current history window
            if i == 0:
                user_turn_content.append({"type": "text", "text": full_text_prompt_for_vlm})

            messages_for_api.append({"role": "user", "content": user_turn_content})

            # Add assistant response if it exists for this historical user turn
            # responses_for_vlm_call_history corresponds to images before the current one.
            if i < len(responses_for_vlm_call_history):
                messages_for_api.append({"role": "assistant", "content": [{"type": "text", "text": responses_for_vlm_call_history[i]}]})
        
        # --- VLM Call ---
        vlm_prediction_raw = None
        current_vlm_call_error = None
        api_retry_attempts = 2
        for attempt in range(api_retry_attempts):
            try:
                self.logger.debug(f"Worker calling UI-TARS (Attempt {attempt + 1}/{api_retry_attempts}). Model: {self.ui_tars_model_name}. Msgs: {len(messages_for_api)}")
                if self.logger.level == logging.DEBUG:
                    for msg_idx, msg_turn in enumerate(messages_for_api):
                        self.logger.debug(f"  Msg Turn {msg_idx+1} Role: {msg_turn['role']}")
                        for content_item_idx, content_item in enumerate(msg_turn['content']):
                            if content_item['type'] == 'text': self.logger.debug(f"    Content[{content_item_idx}] Text: {content_item['text'][:150].replace(chr(10),' ')}...")
                            else: self.logger.debug(f"    Content[{content_item_idx}] Image Hash (last 20 of b64): ...{content_item['image_url']['url'][-20:]}")

                api_response = self.vlm_client.chat.completions.create(
                    model=self.ui_tars_model_name, messages=messages_for_api, temperature=0.0, max_tokens=512)
                vlm_prediction_raw = api_response.choices[0].message.content.strip()
                current_vlm_call_error = None; break
            except Exception as e:
                current_vlm_call_error = f"Worker UI-TARS API error: {e}"
                self.logger.error(f"{current_vlm_call_error} (Attempt {attempt + 1})", exc_info=True if attempt == api_retry_attempts -1 else False)
            if attempt < api_retry_attempts - 1: time.sleep(1.5 * (attempt + 1))

        if current_vlm_call_error or not vlm_prediction_raw:
            err_msg = current_vlm_call_error or "UI-TARS did not return a prediction."
            fail_thought = f"VLM Call Error: {err_msg}"
            fail_action_str = self._generate_pyautogui_command_string(
                {"type":"FAIL_VLM_CALL", "details": err_msg},
                fail_thought, overall_plan, current_sub_task_idx_in_plan)
            return {"error": err_msg, "vlm_response_raw": None, "pyautogui_actions_strings": [fail_action_str],
                    "extracted_thought": fail_thought,
                    # Return original UNMODIFIED history on fail, as this step didn't complete successfully
                    "updated_vlm_responses_history": vlm_responses_history_for_subtask, 
                    "updated_images_history_bytes": images_history_bytes_for_subtask,    
                    "signaled_finish_subtask": False, "signaled_call_user": False}


        updated_subtask_images_history = images_for_vlm_call 

        # `responses_for_vlm_call_history` contains the trimmed historical responses. Append the new one.
        updated_subtask_responses_history = responses_for_vlm_call_history + [vlm_prediction_raw]

        # --- Parse VLM output and Generate PyAutoGUI strings ---
        parsed_actions_list, extracted_thought = parse_llm_output(vlm_prediction_raw, self.logger)
        if extracted_thought: self.logger.info(f"Worker Thought for step: {extracted_thought}")

        pyautogui_strings_for_this_vlm_step = []
        signaled_finish_subtask_this_vlm_step = False
        signaled_call_user_this_vlm_step = False

        if not parsed_actions_list:
            self.logger.warning("Worker UI-TARS returned no parsable actions from VLM. Checking for keywords in raw response.")
            temp_action_type = WAIT_WORD
            if FINISH_WORD in vlm_prediction_raw.lower():
                signaled_finish_subtask_this_vlm_step = True; temp_action_type = FINISH_WORD
            elif CALL_USER in vlm_prediction_raw.lower():
                signaled_call_user_this_vlm_step = True; temp_action_type = CALL_USER
            
            cmd_str = self._generate_pyautogui_command_string(
                {"type": temp_action_type}, extracted_thought or "No parsable action; decided based on keywords or to wait.",
                overall_plan, current_sub_task_idx_in_plan)
            pyautogui_strings_for_this_vlm_step.append(cmd_str)
        else:
            for parsed_action_item in parsed_actions_list:
                action_type = parsed_action_item.get("type", "unknown").lower()
                cmd_str = self._generate_pyautogui_command_string(
                    parsed_action_item, extracted_thought, overall_plan, current_sub_task_idx_in_plan)
                pyautogui_strings_for_this_vlm_step.append(cmd_str)

                if action_type == FINISH_WORD: signaled_finish_subtask_this_vlm_step = True
                if action_type == CALL_USER: signaled_call_user_this_vlm_step = True
                if signaled_finish_subtask_this_vlm_step or signaled_call_user_this_vlm_step:
                    break
        return {
            "vlm_response_raw": vlm_prediction_raw,
            "pyautogui_actions_strings": pyautogui_strings_for_this_vlm_step,
            "signaled_finish_subtask": signaled_finish_subtask_this_vlm_step,
            "signaled_call_user": signaled_call_user_this_vlm_step,
            "error": None, "extracted_thought": extracted_thought,
            "updated_vlm_responses_history": updated_subtask_responses_history,
            "updated_images_history_bytes": updated_subtask_images_history
        }


    def capture_current_screenshot_pil(self, display_image: bool = False) -> Optional[Image.Image]:
        try:
            return capture_screenshot(self.primary_width, self.primary_height, display_image=display_image)
        except Exception as e:
            self.logger.exception(f"Worker: Error during screenshot capture via utils.screenshot: {e}")
            return None