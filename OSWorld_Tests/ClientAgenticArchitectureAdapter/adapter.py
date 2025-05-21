# agentictest/ClientAgenticArchitectureAdapter/adapter.py
import logging
import os
import sys
from PIL import Image 
from io import BytesIO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ClientAgenticArchitecture.coordinator import Coordinator
    from ClientAgenticArchitecture.planner import Planner
    from ClientAgenticArchitecture.worker import Worker, FINISH_WORD as WORKER_FINISH_WORD, CALL_USER as WORKER_CALL_USER, WAIT_WORD as WORKER_WAIT_WORD
    from ClientAgenticArchitecture.utils.logging_utils import setup_logger as client_arch_setup_logger
except ImportError as e:
    print(f"CRITICAL: Failed to import from ClientAgenticArchitecture. Ensure it's in the Python path. Current sys.path: {sys.path}")
    raise e

DESKTOPENV_DONE = "DONE"
DESKTOPENV_FAIL = "FAIL"
DESKTOPENV_WAIT = "WAIT"
DESKTOPENV_FAIL_CALL_USER = "FAIL_CALL_USER"

class ClientAgenticAdapter:
    def __init__(self, model_name_from_config, max_tokens_from_config,
                 observation_type_from_config, action_space_from_config,
                 max_trajectory_length_from_config, **kwargs):

        adapter_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.logger = client_arch_setup_logger("ClientAgenticAdapterInit", adapter_log_level)
        self.logger.info("ClientAgenticAdapter initializing...")
        self.logger.info(f"Framework observation_type: {observation_type_from_config}")
        self.logger.info(f"Framework action_space: {action_space_from_config}")

        planner_api_key = os.getenv("OPENAI_API_KEY")
        planner_model_name = os.getenv("PLANNER_MODEL_NAME", "gpt-4o")
        ui_tars_server_url = os.getenv("UI_TARS_SERVER_URL")
        ui_tars_api_key = os.getenv("UI_TARS_API_KEY", "empty")
        ui_tars_model_name = os.getenv("UI_TARS_MODEL_NAME", "ui-tars")
        log_level_for_components = os.getenv("LOG_LEVEL", "INFO").upper()

        if not planner_api_key:
            self.logger.critical("Planner API Key (OPENAI_API_KEY) is not set. Exiting.")
            raise ValueError("Planner API Key (OPENAI_API_KEY) is missing.")
        if not ui_tars_server_url:
            self.logger.critical("UI_TARS_SERVER_URL is not set. Exiting.")
            raise ValueError("UI_TARS_SERVER_URL is missing.")

        self.logger.debug(f"Internal Planner Config: Model='{planner_model_name}'")
        self.logger.debug(f"Internal Worker Config: Server='{ui_tars_server_url}', Model='{ui_tars_model_name}'")

        try:
            self.worker = Worker(
                ui_tars_server_url=ui_tars_server_url,
                ui_tars_api_key=ui_tars_api_key,
                ui_tars_model_name=ui_tars_model_name, 
                logger=client_arch_setup_logger("ClientArchWorker", log_level_for_components)
            )
            self.planner = Planner(
                api_key=planner_api_key,
                model_name=planner_model_name,
                logger=client_arch_setup_logger("ClientArchPlanner", log_level_for_components)
            )
            # Pass max_steps from framework to coordinator's max_overall_actions_performed
            self.coordinator = Coordinator(
                planner=self.planner,
                worker=self.worker,
                logger=client_arch_setup_logger("ClientArchCoordinator", log_level_for_components),
                max_overall_actions_performed = kwargs.get('max_steps', 25) 
            )
            self.logger.info("ClientAgenticArchitecture components initialized.")
        except Exception as e:
            self.logger.critical(f"Failed to initialize ClientAgenticArchitecture components: {e}", exc_info=True)
            raise

        self.observation_type_from_framework = observation_type_from_config
        self.current_overall_goal_instruction: Optional[str] = None
        self.adapter_internal_task_state: Dict[str, Any] = {}
        self.logger.info("ClientAgenticAdapter initialization complete.")

    def reset(self, runtime_logger: logging.Logger):
        self.logger = runtime_logger 
        self.logger.info("ClientAgenticAdapter reset called.")
        if hasattr(self.coordinator, '_reset_goal_state'):
            self.coordinator._reset_goal_state()
            self.logger.info("Internal Coordinator state has been reset via _reset_goal_state().")
        self.current_overall_goal_instruction = None
        self.adapter_internal_task_state = {} 
        self.logger.info("Adapter internal state reset for new task.")

    def predict(self, instruction: str, obs: dict):
        self.logger.info(f"Adapter predict: Instruction: '{instruction}'")
        screenshot_bytes = obs.get('screenshot')

        if not screenshot_bytes:
            self.logger.error("No screenshot provided in observation. Signaling FAIL to framework.")
            return "Error: No screenshot.", [DESKTOPENV_FAIL]

        coordinator_output: Optional[Dict[str, Any]] = None

        is_first_call_for_this_instruction = (self.current_overall_goal_instruction != instruction) or \
                                             self.adapter_internal_task_state.get('is_first_call_for_goal', True)

        if is_first_call_for_this_instruction:
            self.logger.info(f"Adapter: Processing as first call for goal: '{instruction}'.")
            if self.current_overall_goal_instruction != instruction and self.current_overall_goal_instruction is not None:
                 self.logger.warning(f"Instruction changed from '{self.current_overall_goal_instruction}' to '{instruction}'. Full reset.")
                 self.reset(self.logger) 
            
            self.current_overall_goal_instruction = instruction
            self.adapter_internal_task_state['is_first_call_for_goal'] = False 

            coordinator_output = self.coordinator.initiate_task_and_get_first_actions(
                overall_task_description=instruction,
                initial_screenshot_bytes=screenshot_bytes
            )
        else: 
            self.logger.info(f"Adapter: Continuing existing overall goal: '{self.current_overall_goal_instruction}'")
            previous_status = self.adapter_internal_task_state.get('status')
            if previous_status in ["task_completed_all_plans", "task_failed_critical", "task_call_user",
                                   "task_failed_worker_error", "task_failed_no_plan", 
                                   "task_failed_empty_plan", "task_failed_replan_failed",
                                   "task_failed_max_actions", "task_failed_subtask_max_attempts"]:
                self.logger.warning(f"Task was already in a terminal state '{previous_status}'. Signaling appropriately to framework and resetting for potential next task.")
                last_vlm_response = self.adapter_internal_task_state.get('last_vlm_response', f"Task previously ended with status: {previous_status}")
                # This task is over, reset for the next one from DesktopEnv
                self.current_overall_goal_instruction = None 
                self.adapter_internal_task_state = {}
                if previous_status == "task_call_user": return last_vlm_response, [DESKTOPENV_FAIL_CALL_USER]
                if previous_status == "task_completed_all_plans": return last_vlm_response, [DESKTOPENV_DONE]
                return last_vlm_response, [DESKTOPENV_FAIL]

            coordinator_output = self.coordinator.execute_next_coordinator_step(
                current_screenshot_bytes=screenshot_bytes
            )
        
        self.adapter_internal_task_state['status'] = coordinator_output.get("final_overall_status", "task_ongoing")
        self.adapter_internal_task_state['last_vlm_response'] = coordinator_output.get("vlm_response")

        raw_vlm_response_for_log = coordinator_output.get("vlm_response", "Coordinator: VLM response N/A for this step.")
        pyautogui_actions_from_coordinator = coordinator_output.get("pyautogui_actions", [])
        current_task_status_from_coordinator = coordinator_output.get("final_overall_status", "task_ongoing")

        actions_to_return_to_framework = pyautogui_actions_from_coordinator # Default

        if current_task_status_from_coordinator in [
            "task_failed_worker_error", "task_failed_no_plan", "task_failed_empty_plan",
            "task_failed_replan_failed", "task_failed_max_actions", "task_failed_subtask_max_attempts"
        ]:
            self.logger.info(f"Coordinator status '{current_task_status_from_coordinator}' indicates overall task failure. Signaling FAIL to framework.")
            # The detailed failure (pyautogui_actions_from_coordinator) is logged by coordinator/worker internally.
            # And will be in coordinator_summary.json
            actions_to_return_to_framework = [DESKTOPENV_FAIL]
            self.current_overall_goal_instruction = None # Reset for next task from framework
            self.adapter_internal_task_state = {}


        elif current_task_status_from_coordinator == "task_call_user":
            self.logger.info("Coordinator status indicates CALL_USER. Signaling FAIL_CALL_USER to framework.")
            actions_to_return_to_framework = [DESKTOPENV_FAIL_CALL_USER]
            self.current_overall_goal_instruction = None 
            self.adapter_internal_task_state = {}
        
        elif current_task_status_from_coordinator == "task_completed_all_plans":
            self.logger.info("Coordinator status indicates all plans completed. Signaling DONE to framework.")
            actions_to_return_to_framework = [DESKTOPENV_DONE]
            self.current_overall_goal_instruction = None
            self.adapter_internal_task_state = {}

        elif not pyautogui_actions_from_coordinator and current_task_status_from_coordinator == "task_ongoing":
            self.logger.warning("Coordinator step is 'ongoing' but resulted in no executable pyautogui actions. Generating WAIT.")
            extracted_thought_from_coord = coordinator_output.get("extracted_thought", "Coordinator/Worker decided to wait as no actions were produced.")
            wait_action_str = self.worker._generate_pyautogui_command_string(
                {"type": WORKER_WAIT_WORD}, 
                extracted_thought_from_coord,
                self.coordinator.current_plan, 
                self.coordinator.current_sub_task_index
            )
            actions_to_return_to_framework = [wait_action_str]
        
        if not actions_to_return_to_framework: 
            self.logger.error(f"actions_to_return_to_framework is empty after all checks (status: {current_task_status_from_coordinator}). Defaulting to sending a WAIT signal with context.")
            default_thought = f"Adapter: No actions generated by coordinator/worker for status {current_task_status_from_coordinator}, defaulting to WAIT."
            wait_action_str = self.worker._generate_pyautogui_command_string(
                {"type": WORKER_WAIT_WORD}, default_thought,
                self.coordinator.current_plan, self.coordinator.current_sub_task_index
            )
            actions_to_return_to_framework = [wait_action_str]

        # Logging what is actually being returned
        is_simple_signal = False
        if len(actions_to_return_to_framework) == 1:
            action_content = actions_to_return_to_framework[0]
            if action_content in [DESKTOPENV_DONE, DESKTOPENV_FAIL, DESKTOPENV_FAIL_CALL_USER]:
                is_simple_signal = True
            elif WORKER_WAIT_WORD in action_content and f"# Action: {WORKER_WAIT_WORD}" in action_content :
                is_simple_signal = True 

        if is_simple_signal:
            self.logger.info(f"Returning terminal or WAIT signal to DesktopEnv: {actions_to_return_to_framework}")
        else:
            self.logger.info(f"Returning {len(actions_to_return_to_framework)} pyautogui command string(s) to DesktopEnv for execution.")
            for i, cmd_str in enumerate(actions_to_return_to_framework):
                 self.logger.debug(f"  Adapter Action string to framework #{i+1} (len={len(cmd_str)}):\n{cmd_str[:700].strip()}...")

        return raw_vlm_response_for_log, actions_to_return_to_framework