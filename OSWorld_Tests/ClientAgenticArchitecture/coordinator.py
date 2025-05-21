# agentictest/ClientAgenticArchitecture/coordinator.py
import logging
from typing import List, Optional, Dict, Any
from PIL import Image
from io import BytesIO
import os

from .planner import Planner
from .worker import Worker, FINISH_WORD as WORKER_FINISH_WORD, CALL_USER as WORKER_CALL_USER, WAIT_WORD as WORKER_WAIT_WORD
from .utils.logging_utils import setup_logger

DEFAULT_MAX_SUB_TASK_ATTEMPTS_BY_COORDINATOR = 1
DEFAULT_REPLAN_AFTER_N_SUCCESSFUL_SUBTASKS = 3
DEFAULT_REPLAN_AFTER_N_GLOBAL_SUBTASK_CALLS_FOR_REPLAN = 5
DEFAULT_MAX_OVERALL_ACTIONS_PERFORMED = 25


class Coordinator:
    def __init__(
        self,
        planner: Planner,
        worker: Worker,
        logger: Optional[logging.Logger] = None,
        max_sub_task_attempts: int = DEFAULT_MAX_SUB_TASK_ATTEMPTS_BY_COORDINATOR,
        replan_after_n_successful_subtasks: int = DEFAULT_REPLAN_AFTER_N_SUCCESSFUL_SUBTASKS,
        replan_after_n_global_subtask_calls_for_replan: int = DEFAULT_REPLAN_AFTER_N_GLOBAL_SUBTASK_CALLS_FOR_REPLAN,
        max_overall_actions_performed: int = DEFAULT_MAX_OVERALL_ACTIONS_PERFORMED
    ):
        self.planner = planner
        self.worker = worker
        self.logger = logger or setup_logger("Coordinator", logging.INFO)

        self.max_sub_task_attempts_by_coordinator = max_sub_task_attempts
        self.replan_after_n_successful_subtasks = replan_after_n_successful_subtasks
        self.replan_after_n_global_subtask_calls_for_replan = replan_after_n_global_subtask_calls_for_replan
        
        try:
            env_max_overall_actions = os.getenv("MAX_OVERALL_ACTIONS_PERFORMED")
            if env_max_overall_actions is not None:
                self.max_overall_actions_performed = int(env_max_overall_actions)
                self.logger.info(f"Max overall actions set from ENV: {self.max_overall_actions_performed}")
            else:
                self.max_overall_actions_performed = int(max_overall_actions_performed)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid MAX_OVERALL_ACTIONS_PERFORMED. Using default: {DEFAULT_MAX_OVERALL_ACTIONS_PERFORMED}")
            self.max_overall_actions_performed = DEFAULT_MAX_OVERALL_ACTIONS_PERFORMED

        self.logger.info(f"Coordinator initialized.")
        self.logger.info(f"  Max sub-task attempts (Coordinator level): {self.max_sub_task_attempts_by_coordinator}")
        self.logger.info(f"  Replan after N successful sub-tasks: {self.replan_after_n_successful_subtasks}")
        self.logger.info(f"  Replan after N global sub-task VLM calls for current plan segment: {self.replan_after_n_global_subtask_calls_for_replan}")
        self.logger.info(f"  Max overall generated pyautogui actions limit (Coordinator level): {self.max_overall_actions_performed}")

        self.overall_goal_description: Optional[str] = None
        self.current_plan: Optional[List[str]] = None
        self.current_sub_task_index: int = 0
        self.current_worker_vlm_steps_for_sub_task: int = 0
        self.max_worker_vlm_steps_per_sub_task: int = 3

        self.global_vlm_calls_since_last_replan: int = 0
        self.consecutive_sub_tasks_completed_count: int = 0
        self.current_plan_execution_history: List[Dict[str, Any]] = []
        self.all_executed_actions_generated_code: List[Dict] = []
        self.all_worker_thoughts: List[Dict] = []
        self.all_plans_archive: List[Dict[str, Any]] = []
        
        self.worker_vlm_responses_for_current_subtask: List[str] = []
        self.worker_images_bytes_for_current_subtask: List[bytes] = []
        self.total_pyautogui_actions_generated_this_goal: int = 0

    def _reset_goal_state(self):
        self.overall_goal_description = None
        self.current_plan = None
        self.current_sub_task_index = 0
        self.current_worker_vlm_steps_for_sub_task = 0
        self._reset_replan_triggers_and_plan_history()
        self.all_executed_actions_generated_code = []
        self.all_worker_thoughts = []
        self.all_plans_archive = []
        self._reset_current_subtask_worker_vlm_history()
        self.total_pyautogui_actions_generated_this_goal = 0
        self.logger.debug("Full Coordinator state reset by _reset_goal_state().")

    def _reset_replan_triggers_and_plan_history(self):
        self.global_vlm_calls_since_last_replan = 0
        self.consecutive_sub_tasks_completed_count = 0
        self.current_plan_execution_history = []
        self.logger.debug("Re-planning trigger counters and current plan execution history reset.")

    def _reset_current_subtask_worker_vlm_history(self):
        self.worker_vlm_responses_for_current_subtask = []
        self.worker_images_bytes_for_current_subtask = []
        self.current_worker_vlm_steps_for_sub_task = 0
        self.logger.debug("Worker's VLM history and VLM step count for current sub-task attempt reset.")

    def _add_to_current_plan_execution_history(
        self, sub_task_desc: str, outcome: str, reason: Optional[str] = None,
        pyautogui_actions_generated_count: Optional[int] = 0):
        entry = {"sub_task": sub_task_desc, "outcome": outcome}
        if reason: entry["reason"] = reason
        entry["pyautogui_actions_generated_count"] = pyautogui_actions_generated_count
        self.current_plan_execution_history.append(entry)
        self.logger.debug(f"Added to current plan execution history: {entry}")

    def _get_screenshot_bytes_for_planner(self, provided_screenshot_bytes: Optional[bytes]) -> Optional[bytes]:
        if provided_screenshot_bytes:
            self.logger.debug("Coordinator using provided screenshot for Planner.")
            return provided_screenshot_bytes
        self.logger.warning("Coordinator: No screenshot bytes provided to _get_screenshot_bytes_for_planner.")
        return None
    
    def _archive_current_plan(self, trigger_reason: str, history_context_for_replan: Optional[Dict] = None, raw_planner_response: Optional[str] = None):
        if self.current_plan:
            self.all_plans_archive.append({
                "plan_segment_number": len(self.all_plans_archive) + 1,
                "plan_steps": list(self.current_plan),
                "activated_at_overall_pyautogui_action_count": self.total_pyautogui_actions_generated_this_goal,
                "trigger_reason": trigger_reason,
                "history_context_for_this_replan": history_context_for_replan,
                "raw_planner_response": raw_planner_response
            })
            self.logger.info(f"Archived current plan segment #{len(self.all_plans_archive)} triggered by: {trigger_reason}")

    def _handle_worker_output(self, worker_output: Dict, current_sub_task_desc: str) -> Dict[str, Any]:
        vlm_response = worker_output.get("vlm_response_raw")
        pyautogui_actions = worker_output.get("pyautogui_actions_strings", [])
        num_pyautogui_actions_this_step = len(pyautogui_actions)
        final_overall_status = "task_ongoing"

        if worker_output.get("extracted_thought"):
            self.all_worker_thoughts.append({
                "contextual_total_pyautogui_actions": self.total_pyautogui_actions_generated_this_goal, # Actions *before* this worker step
                "current_sub_task_index": self.current_sub_task_index,
                "worker_vlm_step_for_subtask": self.current_worker_vlm_steps_for_sub_task,
                "thought_from_worker": worker_output["extracted_thought"]
            })
        
        for idx, action_code_str in enumerate(pyautogui_actions):
            self.all_executed_actions_generated_code.append({
                "overall_pyautogui_action_index": self.total_pyautogui_actions_generated_this_goal + idx + 1,
                "sub_task_description": current_sub_task_desc, "sub_task_index": self.current_sub_task_index,
                "worker_vlm_step_for_subtask": self.current_worker_vlm_steps_for_sub_task,
                "sequence_in_vlm_step_output": idx + 1, "action_code_generated": action_code_str,
                "thought_for_action_set": worker_output.get("extracted_thought")
            })
        
        self.worker_vlm_responses_for_current_subtask = worker_output.get("updated_vlm_responses_history", [])
        self.worker_images_bytes_for_current_subtask = worker_output.get("updated_images_history_bytes", [])

        if worker_output.get("error"):
            error_msg = worker_output['error']
            self.logger.error(f"Worker error on sub-task '{current_sub_task_desc}': {error_msg}")
            self._add_to_current_plan_execution_history(current_sub_task_desc, "failure_worker_vlm_step", reason=error_msg, pyautogui_actions_generated_count=num_pyautogui_actions_this_step)
            final_overall_status = "task_failed_worker_error"
            
        elif worker_output.get("signaled_call_user"):
            self.logger.warning(f"Worker signaled CALL_USER for sub-task '{current_sub_task_desc}'.")
            self._add_to_current_plan_execution_history(current_sub_task_desc, "call_user_requested_by_worker", pyautogui_actions_generated_count=num_pyautogui_actions_this_step)
            final_overall_status = "task_call_user"
            self.consecutive_sub_tasks_completed_count = 0 

        elif worker_output.get("signaled_finish_subtask"):
            self.logger.info(f"Worker signaled FINISH for sub-task '{current_sub_task_desc}'.")
            self._add_to_current_plan_execution_history(current_sub_task_desc, "success_worker_signal", pyautogui_actions_generated_count=num_pyautogui_actions_this_step)
            self.consecutive_sub_tasks_completed_count += 1
            self.current_sub_task_index += 1 
            self._reset_current_subtask_worker_vlm_history() 

            if self.current_sub_task_index >= len(self.current_plan or []):
                self.logger.info("All sub-tasks in the current plan appear completed based on worker signals.")
                final_overall_status = "task_completed_all_plans"
        else: 
            self.logger.info(f"Worker completed a VLM step for sub-task '{current_sub_task_desc}', sub-task ongoing.")
            self._add_to_current_plan_execution_history(current_sub_task_desc, "ongoing_worker_vlm_step", pyautogui_actions_generated_count=num_pyautogui_actions_this_step)

        return {
            "success": not bool(worker_output.get("error")), 
            "reason": worker_output.get("error"),
            "vlm_response": vlm_response,
            "pyautogui_actions": pyautogui_actions,
            "final_overall_status": final_overall_status,
            "actions_in_this_step_count": num_pyautogui_actions_this_step,
            "extracted_thought": worker_output.get("extracted_thought") # Pass worker's thought up
        }

    def initiate_task_and_get_first_actions(self, overall_task_description: str, initial_screenshot_bytes: Optional[bytes]) -> Dict[str, Any]:
        self.logger.info(f"Coordinator: Initiating task '{overall_task_description}'.")
        self._reset_goal_state()
        self.overall_goal_description = overall_task_description
        self.total_pyautogui_actions_generated_this_goal = 0

        planner_screenshot_bytes = self._get_screenshot_bytes_for_planner(initial_screenshot_bytes)
        plan, plan_meta = self.planner.get_plan(
            overall_goal=overall_task_description,
            current_screenshot_bytes=planner_screenshot_bytes,
            history_context=None
        )

        if not plan:
            reason = "Planner failed: No initial plan."
            if plan_meta and plan_meta.get("error_details"): reason += f" Details: {plan_meta['error_details']}"
            self.logger.error(reason)
            fail_action_str = self.worker._generate_pyautogui_command_string(
                {"type":"FAIL_PLANNER", "details": reason}, reason, None, None)
            return {"success": False, "reason": reason, "vlm_response": plan_meta.get("raw_response") if plan_meta else None,
                    "pyautogui_actions": [fail_action_str], "final_overall_status": "task_failed_no_plan"}

        self.current_plan = plan
        self.current_sub_task_index = 0
        self._reset_current_subtask_worker_vlm_history()
        self._archive_current_plan("Initial plan generation", raw_planner_response=plan_meta.get("raw_response"))
        self._reset_replan_triggers_and_plan_history()

        if not self.current_plan or self.current_sub_task_index >= len(self.current_plan):
            self.logger.warning("Initial plan is valid but empty, or index out of bounds.")
            empty_plan_reason = "Initial plan from planner was empty or invalid."
            fail_action_str = self.worker._generate_pyautogui_command_string(
                {"type": "FAIL_PLANNER", "details": empty_plan_reason}, empty_plan_reason, self.current_plan, self.current_sub_task_index)
            return {"success": False, "reason": empty_plan_reason, "vlm_response": None,
                    "pyautogui_actions": [fail_action_str], "final_overall_status": "task_failed_empty_plan"}

        current_sub_task_desc = self.current_plan[self.current_sub_task_index]
        self.logger.info(f"Coordinator: First sub-task from new plan: '{current_sub_task_desc}'")
        
        self.current_worker_vlm_steps_for_sub_task = 1
        self.global_vlm_calls_since_last_replan +=1

        pil_image_for_worker = None
        if initial_screenshot_bytes:
            try:
                pil_image_for_worker = Image.open(BytesIO(initial_screenshot_bytes))
            except Exception as e:
                self.logger.error(f"Failed to convert initial_screenshot_bytes to PIL for Worker: {e}")
                fail_action_str = self.worker._generate_pyautogui_command_string({"type":"FAIL_INTERNAL", "details": f"Screenshot error: {e}"}, f"Screenshot error: {e}", self.current_plan, self.current_sub_task_index)
                return {"success": False, "reason": "Screenshot processing error", "vlm_response": None, "pyautogui_actions": [fail_action_str], "final_overall_status": "task_failed"}
        
        worker_output = self.worker.execute_one_vlm_step_for_sub_task(
            sub_task_description=current_sub_task_desc,
            current_screenshot_pil=pil_image_for_worker,
            vlm_responses_history_for_subtask=[], 
            images_history_bytes_for_subtask=[],    
            overall_plan=self.current_plan,
            current_sub_task_idx_in_plan=self.current_sub_task_index
        )
        
        processed_output = self._handle_worker_output(worker_output, current_sub_task_desc)
        # Update total generated actions after handling worker output
        self.total_pyautogui_actions_generated_this_goal += processed_output.get("actions_in_this_step_count", 0)
        return processed_output

    def _should_replan(self, current_sub_task_failed_max_worker_vlm_steps: bool) -> bool:
        if self.total_pyautogui_actions_generated_this_goal >= self.max_overall_actions_performed:
            self.logger.info("Re-plan Check: Max overall pyautogui actions limit reached. No re-planning.")
            return False 
        if current_sub_task_failed_max_worker_vlm_steps:
            self.logger.info("Re-plan Trigger: Current sub-task failed max worker VLM steps.")
            return True
        if self.consecutive_sub_tasks_completed_count > 0 and \
           self.consecutive_sub_tasks_completed_count >= self.replan_after_n_successful_subtasks:
            self.logger.info(f"Re-plan Trigger: {self.consecutive_sub_tasks_completed_count} successful sub-tasks (Threshold: {self.replan_after_n_successful_subtasks}).")
            return True
        if self.global_vlm_calls_since_last_replan > 0 and \
           self.global_vlm_calls_since_last_replan >= self.replan_after_n_global_subtask_calls_for_replan:
            self.logger.info(f"Re-plan Trigger: {self.global_vlm_calls_since_last_replan} global VLM calls for current plan segment (Threshold: {self.replan_after_n_global_subtask_calls_for_replan}).")
            return True
        return False

    def execute_next_coordinator_step(self, current_screenshot_bytes: Optional[bytes]) -> Dict[str, Any]:
        self.logger.info(f"Coordinator: Executing next step. Current sub-task: {self.current_sub_task_index +1}/{len(self.current_plan or [])} (Worker VLM step for this sub-task: {self.current_worker_vlm_steps_for_sub_task +1}). Total PyAutoGUI actions this goal: {self.total_pyautogui_actions_generated_this_goal}")

        if self.total_pyautogui_actions_generated_this_goal >= self.max_overall_actions_performed:
            self.logger.error("Max overall pyautogui actions limit reached during next step. Halting.")
            reason="Max overall pyautogui actions limit reached."
            fail_action_str = self.worker._generate_pyautogui_command_string({"type":"FAIL_MAX_ACTIONS"}, reason, self.current_plan, self.current_sub_task_index)
            return {"success": False, "reason": reason, "vlm_response": "Coordinator: Max actions.",
                    "pyautogui_actions": [fail_action_str], "final_overall_status": "task_failed_max_actions"}

        if not self.current_plan or self.current_sub_task_index >= len(self.current_plan):
            self.logger.info("All sub-tasks in current plan are completed. Task considered finished by coordinator.")
            done_thought = "All plan steps completed."
            done_action_str = self.worker._generate_pyautogui_command_string({"type":WORKER_FINISH_WORD}, done_thought, self.current_plan, self.current_sub_task_index)
            return {"success": True, "reason": "All sub-tasks in final plan completed.", "vlm_response": f"Coordinator: {done_thought}",
                    "pyautogui_actions": [done_action_str], "final_overall_status": "task_completed_all_plans"}

        current_sub_task_desc = self.current_plan[self.current_sub_task_index]
        sub_task_failed_max_worker_vlm_steps = False

        if self.current_worker_vlm_steps_for_sub_task >= self.max_worker_vlm_steps_per_sub_task:
            self.logger.warning(f"Sub-task '{current_sub_task_desc}' has reached max worker VLM steps ({self.max_worker_vlm_steps_per_sub_task}). Marking as failed for re-plan decision.")
            self._add_to_current_plan_execution_history(current_sub_task_desc, "failure_max_worker_vlm_steps", reason=f"Exceeded {self.max_worker_vlm_steps_per_sub_task} worker VLM attempts.")
            self.consecutive_sub_tasks_completed_count = 0 
            sub_task_failed_max_worker_vlm_steps = True
        
        needs_replan = self._should_replan(current_sub_task_failed_max_worker_vlm_steps=sub_task_failed_max_worker_vlm_steps)
        
        if needs_replan:
            self.logger.info(f"Re-planning condition met. Last processed sub-task was '{current_sub_task_desc}'.")
            if not self.overall_goal_description:
                self.logger.error("CRITICAL: Cannot re-plan because overall_goal_description is not set in Coordinator.")
                fail_action_str = self.worker._generate_pyautogui_command_string({"type":"FAIL_INTERNAL"}, "Internal error: goal not set for replan", self.current_plan, self.current_sub_task_index)
                return {"success": False, "reason": "Internal error: goal_description missing for re-plan", "pyautogui_actions": [fail_action_str], "final_overall_status": "task_failed"}

            history_for_planner_api = {
                "previous_plan": list(self.current_plan) if self.current_plan else [],
                "completed_up_to_index": self.current_sub_task_index -1 if sub_task_failed_max_worker_vlm_steps or self.consecutive_sub_tasks_completed_count == 0 else self.current_sub_task_index,
                "failed_sub_task_description": current_sub_task_desc if sub_task_failed_max_worker_vlm_steps or self.consecutive_sub_tasks_completed_count == 0 else None,
                "attempt_summary": list(self.current_plan_execution_history)
            }
            
            planner_screenshot_bytes = self._get_screenshot_bytes_for_planner(current_screenshot_bytes)
            new_plan, new_plan_meta = self.planner.get_plan(
                overall_goal=self.overall_goal_description, current_screenshot_bytes=planner_screenshot_bytes, history_context=history_for_planner_api
            )

            if not new_plan:
                reason = "Planner failed during re-plan."
                if new_plan_meta and new_plan_meta.get("error_details"): reason += f" Details: {new_plan_meta['error_details']}"
                self.logger.error(reason)
                fail_action_str = self.worker._generate_pyautogui_command_string({"type":"FAIL_PLANNER"}, reason, self.current_plan, self.current_sub_task_index)
                return {"success": False, "reason": reason, "vlm_response": new_plan_meta.get("raw_response") if new_plan_meta else None,
                        "pyautogui_actions": [fail_action_str], "final_overall_status": "task_failed_replan_failed"}

            self.logger.info(f"Successfully re-planned. New plan has {len(new_plan)} steps.")
            self.current_plan = new_plan
            self.current_sub_task_index = 0
            self._reset_current_subtask_worker_vlm_history()
            self._archive_current_plan("Re-plan successful", history_context_for_replan=history_for_planner_api, raw_planner_response=new_plan_meta.get("raw_response"))
            self._reset_replan_triggers_and_plan_history()

            if not self.current_plan:
                 fail_action_str = self.worker._generate_pyautogui_command_string({"type":"FAIL_PLANNER"}, "Empty re-plan", None, None)
                 return {"success": False, "reason": "New plan is empty after re-plan (should not happen).", "pyautogui_actions": [fail_action_str], "final_overall_status": "task_failed_empty_replan"}
            
            current_sub_task_desc = self.current_plan[self.current_sub_task_index]
            self.logger.info(f"Coordinator: First sub-task from new re-plan: '{current_sub_task_desc}'")
        
        current_sub_task_desc = self.current_plan[self.current_sub_task_index]
        self.current_worker_vlm_steps_for_sub_task += 1
        self.global_vlm_calls_since_last_replan += 1
        self.logger.info(f"Calling Worker for sub-task '{current_sub_task_desc}' (Worker VLM Attempt for this sub-task: {self.current_worker_vlm_steps_for_sub_task}, Global VLM calls this plan: {self.global_vlm_calls_since_last_replan})")

        pil_image_for_worker = Image.open(BytesIO(current_screenshot_bytes)) if current_screenshot_bytes else None
        worker_output = self.worker.execute_one_vlm_step_for_sub_task(
            sub_task_description=current_sub_task_desc,
            current_screenshot_pil=pil_image_for_worker,
            vlm_responses_history_for_subtask=self.worker_vlm_responses_for_current_subtask,
            images_history_bytes_for_subtask=self.worker_images_bytes_for_current_subtask,
            overall_plan=self.current_plan,
            current_sub_task_idx_in_plan=self.current_sub_task_index
        )
        
        processed_output = self._handle_worker_output(worker_output, current_sub_task_desc)
        self.total_pyautogui_actions_generated_this_goal += processed_output.get("actions_in_this_step_count", 0)
        return processed_output

    def get_final_execution_summary(self) -> Dict[str, Any]:
        final_status_for_summary = "Unknown"
        if self.current_plan and self.current_sub_task_index >= len(self.current_plan):
            final_status_for_summary = "Completed all plan steps."
        elif self.total_pyautogui_actions_generated_this_goal >= self.max_overall_actions_performed:
            final_status_for_summary = "Reached max overall pyautogui actions."
        
        return {
            "overall_goal_description": self.overall_goal_description,
            "final_plan_steps": list(self.current_plan) if self.current_plan else None,
            "sub_tasks_processed_in_final_plan": self.current_sub_task_index,
            "total_pyautogui_actions_generated_in_goal": self.total_pyautogui_actions_generated_this_goal,
            "max_overall_pyautogui_actions_allowed": self.max_overall_actions_performed,
            "final_status_notes": final_status_for_summary,
            "all_plans_archive": self.all_plans_archive,
            "all_executed_actions_generated_code": self.all_executed_actions_generated_code,
            "all_worker_thoughts": self.all_worker_thoughts,
        }