import logging
import time
import base64
import pyautogui
import os
from io import BytesIO
from typing import List, Optional, Dict, Tuple, Any
from PIL import Image
import openai 

from utils.logging_utils import setup_logger
from utils.screenshot import capture_screenshot, detect_primary_screen_size, pil_to_base64
from utils.parser import parse_llm_output
from utils.action_executor import execute_action

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
    """
    Executes a single sub-task using the UI-TARS VLM
    to interact with the GUI.
    """
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

        self.vlm_base_url = f"{ui_tars_server_url.rstrip('/')}/v1" # Standard for OpenAI-compatible APIs

        self.vlm_client = openai.OpenAI(
            base_url=self.vlm_base_url,
            api_key=ui_tars_api_key,
        )

        if primary_screen_size:
            self.primary_width, self.primary_height = primary_screen_size
        else:
            self.primary_width, self.primary_height = detect_primary_screen_size()
        self.logger.info(f"Worker using primary screen size: {self.primary_width}x{self.primary_height}")

        self.action_space_prompt_segment = WORKER_ACTION_SPACE

    def _pil_image_to_bytes(self, image: Image.Image) -> bytes:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return buffered.getvalue()

    def capture_current_screenshot_pil(self, display_image: bool = False) -> Optional[Image.Image]:
        try:
            return capture_screenshot(self.primary_width, self.primary_height, display_image=display_image)
        except Exception as e:
            self.logger.exception("Worker: Error during screenshot capture.")
            return None

    def run_sub_task(
        self,
        sub_task_description: str,
        max_vlm_steps: int = 7
    ) -> Dict[str, Any]:
        self.logger.info(f"Worker starting sub-task: '{sub_task_description}' (max_steps: {max_vlm_steps})")

        sub_task_history_images_bytes: List[bytes] = []
        sub_task_history_vlm_responses: List[str] = []
        actions_performed_records: List[Dict] = []
        thought_process_summary: List[str] = []
        
        signaled_finish = False
        signaled_call_user = False
        current_error = None
        final_screenshot_pil = None

        current_screenshot_pil = self.capture_current_screenshot_pil()
        if not current_screenshot_pil:
            error_msg = "Worker: Failed to capture initial screenshot for sub-task."
            self.logger.error(error_msg)
            return {
                "signaled_finish": False, "signaled_call_user": False,
                "actions_performed_records": [], "final_screenshot_pil": None,
                "error": error_msg, "thought_process_summary": []
            }
        final_screenshot_pil = current_screenshot_pil
        try:
            img_bytes = self._pil_image_to_bytes(current_screenshot_pil)
            sub_task_history_images_bytes.append(img_bytes)
        except Exception as e:
            error_msg = f"Worker: Failed to process initial screenshot: {e}"
            self.logger.exception(error_msg)
            return {
                "signaled_finish": False, "signaled_call_user": False,
                "actions_performed_records": [], "final_screenshot_pil": final_screenshot_pil,
                "error": error_msg, "thought_process_summary": []
            }

        initial_user_prompt_text_for_vlm = WORKER_USER_PROMPT_TEMPLATE.format(
            action_space=self.action_space_prompt_segment,
            language=self.language,
            sub_task_instruction=sub_task_description
        )

        for step_num in range(1, max_vlm_steps + 1):
            self.logger.info(f"Worker: Sub-task step {step_num}/{max_vlm_steps}")

            # --- Prepare messages for UI-TARS VLM ---
            if len(sub_task_history_images_bytes) > self.worker_history_n:
                trim_amount = len(sub_task_history_images_bytes) - self.worker_history_n
                sub_task_history_images_bytes = sub_task_history_images_bytes[trim_amount:]
                if len(sub_task_history_vlm_responses) >= trim_amount:
                    sub_task_history_vlm_responses = sub_task_history_vlm_responses[trim_amount:]
                else:
                    self.logger.warning("Worker UI-TARS history (responses) misalignment during trimming.")
                    sub_task_history_vlm_responses = []

            messages_for_api: List[Dict[str, Any]] = []
            
            for i, img_bytes_hist in enumerate(sub_task_history_images_bytes):
                base64_encoded_img = base64.b64encode(img_bytes_hist).decode("utf-8")
                user_content_for_turn: List[Dict[str, Any]] = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_encoded_img}"}}
                ]
                
                if i == 0: 
                    user_content_for_turn.append({"type": "text", "text": initial_user_prompt_text_for_vlm})
                
                messages_for_api.append({"role": "user", "content": user_content_for_turn})
                
                if i < len(sub_task_history_vlm_responses):
                    messages_for_api.append({"role": "assistant", "content": [{"type": "text", "text": sub_task_history_vlm_responses[i]}]})
            
            vlm_prediction_raw = None
            current_vlm_call_error = None
            api_retry_attempts = 3
            for attempt in range(api_retry_attempts):
                try:
                    self.logger.debug(f"Worker calling UI-TARS (Attempt {attempt + 1}/{api_retry_attempts}). Sending {len(messages_for_api)} messages.")
                    api_response = self.vlm_client.chat.completions.create(
                        model=self.ui_tars_model_name, 
                        messages=messages_for_api,
                        temperature=0.0,
                        max_tokens=10240,
                    )
                    vlm_prediction_raw = api_response.choices[0].message.content.strip()
                    self.logger.debug(f"Raw UI-TARS prediction for sub-task:\n{vlm_prediction_raw}")
                    current_vlm_call_error = None
                    break 
                except openai.APIError as e:
                    current_vlm_call_error = f"Worker UI-TARS API error: {e}"
                    self.logger.error(f"{current_vlm_call_error} (Attempt {attempt + 1})")
                except Exception as e:
                    current_vlm_call_error = f"Unexpected error during Worker UI-TARS call: {e}"
                    self.logger.exception(f"{current_vlm_call_error} (Attempt {attempt + 1})")
                
                if attempt < api_retry_attempts - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                     self.logger.error("Max API retries reached for Worker UI-TARS call.")

            if current_vlm_call_error or vlm_prediction_raw is None:
                current_error = current_vlm_call_error or "UI-TARS did not return a prediction for sub-task."
                self.logger.error(f"Halting sub-task due to UI-TARS call failure: {current_error}")
                break 

            sub_task_history_vlm_responses.append(vlm_prediction_raw)

            parsed_actions_list, extracted_thought = parse_llm_output(vlm_prediction_raw, self.logger)
            if extracted_thought:
                thought_text = f"Sub-task Step {step_num} Thought: {extracted_thought}"
                thought_process_summary.append(thought_text)
                self.logger.info(thought_text)
            
            if not parsed_actions_list:
                self.logger.warning("Worker UI-TARS returned no parsable actions for sub-task.")
                if FINISH_WORD in vlm_prediction_raw.lower() and not any(a.get('type') == FINISH_WORD for a in parsed_actions_list):
                    self.logger.info("Heuristic: Detected 'finished' in raw UI-TARS prediction but not parsed. Assuming finish.")
                    signaled_finish = True
                    actions_performed_records.append({
                        "action": FINISH_WORD, "step_in_sub_task": step_num, 
                        "thought_for_action": extracted_thought or "Inferred finish",
                        "details": "Finished signal inferred from raw text."
                    })
                    break

            action_executed_successfully_this_step = False
            for i_action, parsed_action_item in enumerate(parsed_actions_list):
                action_type = parsed_action_item.get("type", "unknown")
                self.logger.info(f"Executing action {i_action+1}/{len(parsed_actions_list)} for sub-task: {action_type}")

                should_continue_script, action_record = execute_action(
                    parsed_action=parsed_action_item,
                    raw_action=vlm_prediction_raw,
                    primary_width=self.primary_width,
                    primary_height=self.primary_height,
                    logger=self.logger
                )
                
                if action_record:
                    action_record["step_in_sub_task"] = step_num
                    action_record["thought_for_action"] = extracted_thought
                    actions_performed_records.append(action_record)
                    action_executed_successfully_this_step = not str(action_record.get("action","")).startswith("FAIL_")
                else:
                    self.logger.warning(f"Action '{action_type}' did not result in an action record from execute_action.")
                    action_executed_successfully_this_step = False

                if action_type == FINISH_WORD:
                    signaled_finish = True; break
                if action_type == CALL_USER:
                    signaled_call_user = True; break
                
                if not should_continue_script:
                    self.logger.warning(f"execute_action indicated stop for action '{action_type}'. Halting actions for this step.")
                    if action_record and action_record.get("action","").startswith("FAIL_"):
                         current_error = action_record.get("details", f"Action {action_type} failed during execution.")
                    elif not action_record :
                         current_error = f"Action {action_type} failed pre-execution or was unknown to executor."
                    else:
                         current_error = f"Action {action_type} indicated script should not continue."
                    break 
            
            if signaled_finish or signaled_call_user or current_error:
                break

            if not action_executed_successfully_this_step and parsed_actions_list:
                 self.logger.warning("No action in the parsed list was executed successfully this step.")

            next_screenshot_pil = self.capture_current_screenshot_pil()
            if not next_screenshot_pil:
                current_error = "Worker: Failed to capture intermediate screenshot for sub-task."
                self.logger.error(current_error)
                break 
            
            final_screenshot_pil = next_screenshot_pil
            try:
                img_bytes = self._pil_image_to_bytes(next_screenshot_pil)
                sub_task_history_images_bytes.append(img_bytes)
            except Exception as e:
                current_error = f"Worker: Failed to process intermediate screenshot: {e}"
                self.logger.exception(current_error)
                break

        if not current_error and not signaled_finish and not signaled_call_user and step_num >= max_vlm_steps:
            self.logger.warning(f"Sub-task '{sub_task_description}' reached max_vlm_steps ({max_vlm_steps}) without explicit finish/call_user signal.")

        return {
            "signaled_finish": signaled_finish,
            "signaled_call_user": signaled_call_user,
            "actions_performed_records": actions_performed_records,
            "final_screenshot_pil": final_screenshot_pil,
            "error": current_error,
            "thought_process_summary": thought_process_summary
        }