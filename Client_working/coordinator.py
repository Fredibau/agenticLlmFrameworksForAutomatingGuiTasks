import logging
from typing import List, Optional, Dict, Any
from PIL import Image
from io import BytesIO

from planner import Planner
from worker import Worker
from utils.logging_utils import setup_logger

# --- Configuration Defaults ---
DEFAULT_MAX_SUB_TASK_ATTEMPTS_BY_COORDINATOR = 1
DEFAULT_REPLAN_AFTER_N_SUCCESSFUL_SUBTASKS = 2
DEFAULT_REPLAN_AFTER_N_GLOBAL_SUBTASK_CALLS_FOR_REPLAN = 10
DEFAULT_MAX_OVERALL_ACTIONS_PERFORMED = 15

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
            self.max_overall_actions_performed = int(max_overall_actions_performed)
        except ValueError:
            self.logger.error(f"Invalid max_overall_actions_performed value '{max_overall_actions_performed}'. Using default: {DEFAULT_MAX_OVERALL_ACTIONS_PERFORMED}")
            self.max_overall_actions_performed = DEFAULT_MAX_OVERALL_ACTIONS_PERFORMED

        self.logger.info(f"Coordinator initialized.")
        self.logger.info(f"  Max sub-task attempts by Coordinator: {self.max_sub_task_attempts_by_coordinator}")
        self.logger.info(f"  Replan after N successful sub-tasks: {self.replan_after_n_successful_subtasks}")
        self.logger.info(f"  Replan after N global sub-task calls: {self.replan_after_n_global_subtask_calls_for_replan}")
        self.logger.info(f"  Max overall actions performed limit: {self.max_overall_actions_performed}")

        self.current_plan: Optional[List[str]] = None
        self.current_sub_task_index: int = 0
        self.global_subtask_calls_since_last_replan: int = 0
        self.consecutive_sub_tasks_completed_count: int = 0
        self.current_plan_execution_history: List[Dict[str, Any]] = []
        self.all_executed_actions: List[Dict] = []
        self.all_worker_thoughts: List[Dict] = []
        self.all_plans_archive: List[Dict[str, Any]] = [] 

    def _reset_goal_state(self):
        self._reset_replan_counters_and_history()
        self.current_plan = None
        self.current_sub_task_index = 0
        self.all_executed_actions = []
        self.all_worker_thoughts = []
        self.all_plans_archive = [] 
        self.logger.debug("Full Coordinator state reset for new goal.")

    def _reset_replan_counters_and_history(self):
        self.global_subtask_calls_since_last_replan = 0
        self.consecutive_sub_tasks_completed_count = 0
        self.current_plan_execution_history = []
        self.logger.debug("Re-planning counters and execution history for current plan segment reset.")

    def _add_to_current_plan_execution_history(
        self, sub_task_desc: str, outcome: str, reason: Optional[str] = None,
        actions_performed_records: Optional[List[Dict]] = None):
        entry = {"sub_task": sub_task_desc, "outcome": outcome}
        if reason: entry["reason"] = reason
        if actions_performed_records: entry["actions_count"] = len(actions_performed_records)
        self.current_plan_execution_history.append(entry)
        self.logger.debug(f"Added to execution history for planner: {entry}")

    def _should_replan(self, current_sub_task_fully_failed: bool) -> bool:
        if current_sub_task_fully_failed:
            self.logger.info("Re-plan Trigger: Current sub-task failed all Coordinator attempts.")
            return True
        if self.consecutive_sub_tasks_completed_count > 0 and \
           self.consecutive_sub_tasks_completed_count >= self.replan_after_n_successful_subtasks:
            self.logger.info(f"Re-plan Trigger: {self.consecutive_sub_tasks_completed_count} successful sub-tasks completed consecutively (threshold: {self.replan_after_n_successful_subtasks}).")
            return True
        if self.global_subtask_calls_since_last_replan > 0 and \
           self.global_subtask_calls_since_last_replan >= self.replan_after_n_global_subtask_calls_for_replan:
            self.logger.info(f"Re-plan Trigger: {self.global_subtask_calls_since_last_replan} sub-task calls made for this plan iteration (threshold: {self.replan_after_n_global_subtask_calls_for_replan}).")
            return True
        return False

    def _get_screenshot_bytes_for_planner(self) -> Optional[bytes]:
        self.logger.debug("Requesting fresh screenshot from Worker for Planner.")
        pil_image = self.worker.capture_current_screenshot_pil(display_image=False)
        if pil_image:
            try:
                if hasattr(self.worker, '_pil_image_to_bytes'):
                    return self.worker._pil_image_to_bytes(pil_image)
                else: # Fallback
                    buffered = BytesIO()
                    pil_image.save(buffered, format="PNG")
                    self.logger.warning("Used fallback PIL -> bytes conversion in Coordinator.")
                    return buffered.getvalue()
            except Exception as e:
                self.logger.error(f"Coordinator: Failed to convert PIL image to bytes: {e}")
        else:
            self.logger.warning("Coordinator: Worker failed to capture screenshot for Planner.")
        return None

    def execute_overall_goal(self, overall_task_description: str) -> Dict[str, Any]:
        self.logger.info(f"Coordinator starting overall goal: '{overall_task_description}'")
        self._reset_goal_state()

        task_halted_by_max_actions = False
        overall_actions_performed_count = 0 

        # --- Initial Plan ---
        self.logger.info("Requesting initial plan from Planner...")
        initial_screenshot_bytes = self._get_screenshot_bytes_for_planner()
        plan, plan_meta = self.planner.get_plan(overall_goal=overall_task_description, current_screenshot_bytes=initial_screenshot_bytes, history_context=None)

        if not plan:
            self.logger.error("Planner failed to provide an initial plan.")
            reason = "Planner failed to provide an initial plan."
            if plan_meta and plan_meta.get("error_details"):
                reason += f" Details: {plan_meta['error_details']}"
            elif plan_meta and plan_meta.get("raw_response"):
                reason += f" Raw response hint: {str(plan_meta.get('raw_response', ''))[:200]}"
            return {
                "success": False, "reason": reason, "overall_actions_performed_count": 0,
                "max_overall_actions_performed_in_run": self.max_overall_actions_performed,
                "all_plans_archive": self.all_plans_archive 
            }

        self.current_plan = plan
        self.logger.info(f"Initial plan received with {len(self.current_plan)} sub-tasks.")

        if self.current_plan:
            self.all_plans_archive.append({
                "plan_segment_number": 1,
                "plan_steps": list(self.current_plan), 
                "activated_at_overall_step": overall_actions_performed_count + 1,
                "trigger_reason": "Initial plan generation",
                "raw_planner_response": plan_meta.get("raw_response") if plan_meta else None,
                "history_context_for_this_replan": None 
            })

        self._reset_replan_counters_and_history() 

        # --- Main Execution Loop ---
        final_success: bool = False
        final_reason: str = "Task execution started but did not reach a defined end state."

        while self.current_plan and self.current_sub_task_index < len(self.current_plan):
            self.logger.debug(f"OUTER LOOP START: overall_actions: {overall_actions_performed_count}/{self.max_overall_actions_performed}, current_sub_task_index: {self.current_sub_task_index}")
            if overall_actions_performed_count >= self.max_overall_actions_performed:
                final_reason = f"Overall task stopped: Maximum overall actions ({self.max_overall_actions_performed}) reached before starting next sub-task. Actions performed: {overall_actions_performed_count}."
                self.logger.error(final_reason)
                final_success = False
                task_halted_by_max_actions = True
                break

            current_sub_task_idx_for_record = self.current_sub_task_index
            current_sub_task_desc = self.current_plan[current_sub_task_idx_for_record]
            self.logger.info(f"Coordinator: Processing sub-task {current_sub_task_idx_for_record + 1}/{len(self.current_plan)}: '{current_sub_task_desc}'")

            sub_task_succeeded_after_retries = False
            sub_task_requested_call_user = False
            actions_performed_in_last_subtask_block: Optional[List[Dict]] = []

            for attempt_num in range(1, self.max_sub_task_attempts_by_coordinator + 1):
                self.logger.debug(f"INNER LOOP ATTEMPT {attempt_num}: overall_actions: {overall_actions_performed_count}/{self.max_overall_actions_performed}")
                if overall_actions_performed_count >= self.max_overall_actions_performed:
                    final_reason = (f"Overall task stopped during sub-task '{current_sub_task_desc}' (attempt {attempt_num}): "
                                    f"Maximum overall actions ({self.max_overall_actions_performed}) already met. "
                                    f"Actions performed so far: {overall_actions_performed_count}.")
                    self.logger.error(final_reason)
                    final_success = False
                    task_halted_by_max_actions = True
                    break

                self.logger.info(f"Attempt {attempt_num}/{self.max_sub_task_attempts_by_coordinator} for sub-task: '{current_sub_task_desc}'")
                self.global_subtask_calls_since_last_replan += 1
                self.logger.info(f"  (Overall actions performed so far: {overall_actions_performed_count}, max_actions: {self.max_overall_actions_performed})")
                self.logger.info(f"  (Global subtask calls since last replan: {self.global_subtask_calls_since_last_replan}, consecutive successes: {self.consecutive_sub_tasks_completed_count})")

                worker_result = self.worker.run_sub_task(sub_task_description=current_sub_task_desc)
                actions_performed_in_last_subtask_block = worker_result.get("actions_performed_records", [])

                initial_action_count_for_batch = overall_actions_performed_count
                if actions_performed_in_last_subtask_block:
                    for action_record in actions_performed_in_last_subtask_block:
                        if overall_actions_performed_count >= self.max_overall_actions_performed:
                            final_reason = f"Overall task stopped: Maximum overall actions ({self.max_overall_actions_performed}) reached processing action #{overall_actions_performed_count + 1} during sub-task '{current_sub_task_desc}'."
                            self.logger.error(final_reason)
                            final_success = False
                            task_halted_by_max_actions = True
                            break
                        overall_actions_performed_count += 1
                        action_record["overall_action_step"] = overall_actions_performed_count
                        action_record["sub_task_description"] = current_sub_task_desc
                        action_record["sub_task_index"] = current_sub_task_idx_for_record
                        action_record["attempt"] = attempt_num
                        self.all_executed_actions.append(action_record)
                    if task_halted_by_max_actions: break

                    actions_processed_this_call = overall_actions_performed_count - initial_action_count_for_batch
                    self.logger.info(f"Processed {actions_processed_this_call} actions from worker call. Updated overall_actions_performed_count: {overall_actions_performed_count}")

                worker_thoughts_summary = worker_result.get("thought_process_summary", [])
                if worker_thoughts_summary:
                    structured_thoughts = []
                    for thought_text in worker_thoughts_summary:
                        structured_thoughts.append({
                            "overall_action_step_context": overall_actions_performed_count,
                            "sub_task_index": current_sub_task_idx_for_record,
                            "attempt": attempt_num,
                            "thought_from_worker": thought_text
                        })
                    self.all_worker_thoughts.extend(structured_thoughts)
                    self.logger.debug(f"Added {len(structured_thoughts)} worker thoughts for sub-task {current_sub_task_idx_for_record+1}, attempt {attempt_num}.")

                if task_halted_by_max_actions: break

                if worker_result.get("signaled_finish"):
                    self.logger.info(f"Sub-task '{current_sub_task_desc}' COMPLETED by Worker (attempt {attempt_num}).")
                    self.consecutive_sub_tasks_completed_count += 1
                    self._add_to_current_plan_execution_history(current_sub_task_desc, "success", actions_performed_records=actions_performed_in_last_subtask_block)
                    sub_task_succeeded_after_retries = True
                    break
                elif worker_result.get("signaled_call_user"):
                    self.logger.warning(f"Sub-task '{current_sub_task_desc}' signaled CALL_USER (attempt {attempt_num}).")
                    self._add_to_current_plan_execution_history(current_sub_task_desc, "call_user_requested", reason="Worker requested user assistance.", actions_performed_records=actions_performed_in_last_subtask_block)
                    sub_task_requested_call_user = True
                    self.consecutive_sub_tasks_completed_count = 0
                    break
                else:
                    error_msg = worker_result.get("error", "Worker did not signal finish and gave no specific error.")
                    self.logger.warning(f"Attempt {attempt_num} for sub-task '{current_sub_task_desc}' did not complete. Worker Reason: {error_msg}")
                    self.consecutive_sub_tasks_completed_count = 0
                    if attempt_num >= self.max_sub_task_attempts_by_coordinator:
                        self.logger.error(f"Sub-task '{current_sub_task_desc}' FAILED after {self.max_sub_task_attempts_by_coordinator} Coordinator attempts.")
                        self._add_to_current_plan_execution_history(current_sub_task_desc, "failure", reason=f"Max Coordinator retries. Last worker reason: {error_msg}", actions_performed_records=actions_performed_in_last_subtask_block)

            if task_halted_by_max_actions:
                self.logger.info("Task halted by max actions during sub-task attempts. Skipping further processing.")
                break

            needs_to_replan_now = False

            if sub_task_requested_call_user:
                final_reason = f"Worker requested user intervention on sub-task: '{current_sub_task_desc}'."
                self.logger.warning(final_reason)
                final_success = False
                break

            if sub_task_succeeded_after_retries:
                self.logger.info(f"Sub-task '{current_sub_task_desc}' was successful according to Coordinator.")
                if self._should_replan(current_sub_task_fully_failed=False):
                    needs_to_replan_now = True
                else:
                    self.current_sub_task_index += 1
                    continue
            else:
                self.logger.warning(f"Sub-task '{current_sub_task_desc}' did not succeed after all Coordinator attempts.")
                if self._should_replan(current_sub_task_fully_failed=True):
                    needs_to_replan_now = True
                else:
                    final_reason = f"Sub-task '{current_sub_task_desc}' failed, and no re-plan condition was met as per _should_replan. Halting overall task."
                    self.logger.error(final_reason)
                    final_success = False
                    break

            if needs_to_replan_now:
                self.logger.info(f"Re-planning condition met. Current sub-task was '{current_sub_task_desc}'. Proceeding with re-planning attempt.")
                self.logger.debug(f"REPLAN CHECK (pre-replan action count): overall_actions: {overall_actions_performed_count}/{self.max_overall_actions_performed}")
                if overall_actions_performed_count >= self.max_overall_actions_performed:
                    final_reason = (f"Overall task stopped: Maximum overall actions ({self.max_overall_actions_performed}) "
                                    f"reached before re-planning could be initiated for '{current_sub_task_desc}'. "
                                    f"Actions performed: {overall_actions_performed_count}.")
                    self.logger.error(final_reason)
                    final_success = False
                    task_halted_by_max_actions = True
                    break

                self.logger.info("Requesting new plan from Planner...")
                screenshot_bytes_for_replan = self._get_screenshot_bytes_for_planner()

                failed_desc_for_history = None
                last_successfully_completed_idx = -1

                if sub_task_succeeded_after_retries:
                    last_successfully_completed_idx = self.current_sub_task_index
                else:
                    failed_desc_for_history = current_sub_task_desc
                    last_successfully_completed_idx = self.current_sub_task_index - 1

                history_for_planner_api = {
                    "previous_plan": list(self.current_plan) if self.current_plan else [], 
                    "completed_up_to_index": last_successfully_completed_idx,
                    "failed_sub_task_description": failed_desc_for_history,
                    "attempt_summary": list(self.current_plan_execution_history) 
                }

                new_plan, new_plan_meta = self.planner.get_plan(
                    overall_goal=overall_task_description,
                    current_screenshot_bytes=screenshot_bytes_for_replan,
                    history_context=history_for_planner_api
                )

                if new_plan:
                    self.logger.info(f"New plan received with {len(new_plan)} sub-tasks. Resetting plan-specific counters and state.")
                    self.current_plan = new_plan

                    replan_trigger_description = "Replan triggered"
                    if history_for_planner_api.get("failed_sub_task_description"):
                        replan_trigger_description += f" due to failure of sub-task: '{history_for_planner_api['failed_sub_task_description']}'."
                    elif history_for_planner_api.get("completed_up_to_index", -1) > -1 :
                        replan_trigger_description += f" after processing previous plan segment (completed up to index {history_for_planner_api['completed_up_to_index'] +1})."
                    else:
                        replan_trigger_description += " based on other replan criteria."

                    safe_history_context_for_archive = {
                        "previous_plan": list(history_for_planner_api["previous_plan"]), 
                        "completed_up_to_index": history_for_planner_api["completed_up_to_index"],
                        "failed_sub_task_description": history_for_planner_api.get("failed_sub_task_description"),
                        "attempt_summary": list(history_for_planner_api["attempt_summary"]) 
                    }

                    self.all_plans_archive.append({
                        "plan_segment_number": len(self.all_plans_archive) + 1,
                        "plan_steps": list(self.current_plan),  
                        "activated_at_overall_step": overall_actions_performed_count + 1,
                        "trigger_reason": replan_trigger_description,
                        "history_context_for_this_replan": safe_history_context_for_archive,
                        "raw_planner_response": new_plan_meta.get("raw_response") if new_plan_meta else None
                    })

                    self.current_sub_task_index = 0
                    self._reset_replan_counters_and_history()
                else:
                    planner_error_info = str(new_plan_meta.get("raw_response", ""))[:200] if new_plan_meta else "No details from planner."
                    if new_plan_meta and new_plan_meta.get("error_details"):
                        planner_error_info += f" Details: {new_plan_meta['error_details']}"
                    final_reason = f"Planner failed to provide a new plan during re-planning. Planner info: {planner_error_info}"
                    self.logger.error(final_reason)
                    final_success = False
                    break
        # --- End of Main Execution (outer while) Loop ---

        if not task_halted_by_max_actions and final_reason == "Task execution started but did not reach a defined end state.":
            if self.current_plan and self.current_sub_task_index >= len(self.current_plan):
                final_reason = "All sub-tasks in the final plan completed successfully."
                final_success = True
            elif not self.current_plan:
                final_reason = "Task ended prematurely: No plan was available or generated." 
                final_success = False
            else:
                final_reason = (f"Task ended prematurely. Processed up to sub-task {self.current_sub_task_index +1} "
                                f"of {len(self.current_plan) if self.current_plan else 0}. Actions: {overall_actions_performed_count}.")
                final_success = False

        self.logger.info(f"Coordinator finishing goal execution. Success: {final_success}. Reason: {final_reason}")
        self.logger.info(f"FINAL TALLY: Total individual actions performed: {overall_actions_performed_count} (Max allowed: {self.max_overall_actions_performed})")

        final_result_dict = {
            "success": final_success,
            "reason": final_reason,
            "total_sub_tasks_in_final_plan": len(self.current_plan) if self.current_plan else 0,
            "completed_sub_tasks_in_final_plan": self.current_sub_task_index,
            "final_plan_if_any": list(self.current_plan) if self.current_plan else None, 
            "all_plans_archive": self.all_plans_archive, 
            "executed_actions": self.all_executed_actions,
            "worker_thoughts": self.all_worker_thoughts,
            "overall_actions_performed_count": overall_actions_performed_count,
            "max_overall_actions_performed_in_run": self.max_overall_actions_performed
        }
        return final_result_dict