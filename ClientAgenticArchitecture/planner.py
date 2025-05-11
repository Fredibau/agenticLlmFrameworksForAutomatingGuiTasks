import logging
import re 
import base64 
from typing import List, Optional, Dict, Tuple, Any
import openai 

from utils.logging_utils import setup_logger

# Configuration for the planner LLM
DEFAULT_PLANNER_MODEL = "gpt-4o"
MAX_PLANNER_TOKENS = 3048
PLANNER_TEMPERATURE = 0.5

class Planner:
    """
    Interacts with a multi-modal language model to generate step-by-step plans
    for achieving an overall goal via GUI interactions, using a screenshot.
    """
    def __init__(
        self,
        api_key: str, # API key for the planning LLM (OpenAI API key)
        model_name: str = DEFAULT_PLANNER_MODEL,
        logger: Optional[logging.Logger] = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.logger = logger or setup_logger("Planner", logging.INFO)
        
        self.logger.info(
            f"Planner initialized with model '{model_name}'. "
            "This model is assumed to be multi-modal and support image inputs."
        )

        self.llm_client = openai.OpenAI(api_key=self.api_key)

    def _construct_text_prompt_parts(
        self,
        overall_goal: str,
        history_context: Optional[Dict] = None
    ) -> str:
        prompt_lines = [
            "You are a meticulous AI assistant responsible for creating high-level, step-by-step plans to achieve a user's goal by interacting with a graphical user interface (GUI).",
            "You will be given an overall goal, an image of the current screen, and optionally, the history of a previous plan attempt.",
            "Your task is to break down the overall goal into a sequence of logically coherent subplans, each composed of multiple ordered sub-steps.",
            "Each subplan should group together closely related GUI actions and present them in a way that makes the flow of interaction clear and efficient.",
            "The plan should be as efficient as possible, minimizing the number of steps while ensuring clarity and correctness.",
            "Just focus on the task at hand and do not include any disclaimers, unnecessary information or testing. Also you do not need to verify changes.",
            "Do not include any steps related to verifying, testing, evaluating, or checking if the task worked — the final subplan must be the last actionable step needed to complete the goal.",
            "Base your reasoning on the visual information from the screen and the textual goal.",
            "Output Format:",
            "Provide your plan as a numbered list of coherent subplans. Each subplan should begin with 'Follow these steps in order to 'Short description of task': followed by a list of 1) ..., 2) ..., etc.",
            "Use clear, imperative language. Each sub-step should describe a single, actionable task.",
            "Be specific about interactions, e.g., state when to use a single click, double click, type text, or drag-and-drop.",
            "After the last actionable Step, include a final step that signals completion: 'Signal finished()'.",
            "Example Plan Output (ensure each main numbered item below corresponds to a distinct subplan, and its sub-steps 1), 2) are part of its description):",
            "1. Follow these steps in order to open Notepad: 1) Click on the Windows search bar, 2) Type 'Notepad', 3) Click on the Notepad app.",
            "2. Follow these steps in order to write and save our massage: 1) Type 'Hello World', 2) Click 'Save' in the popup., 3) Signal finished()",
            "---"
        ]

        prompt_lines.append(f"Overall Goal: {overall_goal}")
        prompt_lines.append("An image of the current screen is provided for visual context.")


        if history_context:
            prompt_lines.append("\n--- Previous Plan Attempt Information ---")
            if history_context.get("previous_plan"):
                prev_plan_str = "\n".join([f"  {idx+1}. {task}" for idx, task in enumerate(history_context['previous_plan'])])
                prompt_lines.append(f"Previous Plan Being Executed:\n{prev_plan_str}")
            
            completed_idx = history_context.get("completed_up_to_index", -1)
            if completed_idx > -1 :
                prompt_lines.append(f"Status: Sub-tasks up to and including number {completed_idx + 1} of the previous plan were reported as completed.")
            
            if history_context.get("failed_sub_task_description"):
                prompt_lines.append(f"Issue: The plan execution stopped or failed while attempting: '{history_context['failed_sub_task_description']}'")

            if history_context.get("attempt_summary"):
                summary_str = "\n".join([
                    f"  - '{s.get('sub_task', 'N/A')}': {s.get('outcome', 'N/A')}" + (f" (Reason: {s.get('reason')})" if s.get('reason') else "")
                    for s in history_context["attempt_summary"]
                ])
                prompt_lines.append(f"Summary of recent sub-task outcomes for the previous plan:\n{summary_str}")
            prompt_lines.append("--- End of Previous Plan Attempt Information ---")
            prompt_lines.append("\nConsidering this history and the provided screen image, please provide an updated or revised step-by-step plan to achieve the overall goal.")
            prompt_lines.append("If you determine that the 'Overall Goal' IS ALREADY ACHIEVED, you should write exactly: 1. Follow these steps in order: Just signal finished()")

        else:
            prompt_lines.append("\nBased on the provided screen image and the goal, generate the initial step-by-step plan.")

        prompt_lines.append("---")
        prompt_lines.append("Your Plan (numbered list):")
        return "\n".join(prompt_lines)

    def _parse_plan_from_response(self, llm_response_content: str) -> Optional[List[str]]:
        """
        Parses the LLM response to extract a list of plan steps.
        Each plan step can be multi-line and should correspond to a main numbered item
        (e.g., "1. Follow these steps...") from the LLM's output.
        """
        self.logger.debug(f"Raw plan from LLM:\n{llm_response_content}")
        plan_items: List[str] = []

        markers = list(re.finditer(r"^\s*\d+\.\s*", llm_response_content, re.MULTILINE))

        if not markers:
            self.logger.error(f"Failed to find any numbered step markers (e.g., '1.', '2.') in LLM response. Content:\n{llm_response_content}")
            return None

        for i in range(len(markers)):
            marker_obj = markers[i]
            content_start_index = marker_obj.end()

            if i + 1 < len(markers):
                next_marker_start_index = markers[i+1].start()
                content_block = llm_response_content[content_start_index:next_marker_start_index]
            else:
                content_block = llm_response_content[content_start_index:]
            
            stripped_content = content_block.strip()
            if stripped_content:
                plan_items.append(stripped_content)

        if not plan_items:
            self.logger.error(f"Found markers but failed to extract valid plan steps from LLM response. Content:\n{llm_response_content}")
            return None
            
        self.logger.info(f"Parsed plan with {len(plan_items)} main steps.")
        if self.logger.level == logging.DEBUG:
            for idx, item in enumerate(plan_items):
                self.logger.debug(f"Step {idx+1}: {item}")
        return plan_items

    # Main method to get a plan from the LLM
    def get_plan(
        self,
        overall_goal: str,
        current_screenshot_bytes: Optional[bytes] = None,
        history_context: Optional[Dict] = None  
    ) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        
        # Create the text prompt for the LLM
        text_prompt_part = self._construct_text_prompt_parts(
            overall_goal, history_context
        )
        
        self.logger.info(f"Requesting plan for goal: '{overall_goal}'" + 
                         (" (re-planning)" if history_context else " (initial plan)") +
                         (" with screenshot." if current_screenshot_bytes else " without screenshot."))
        self.logger.debug(f"Planner text prompt part for {self.model_name}:\n{text_prompt_part[:1000]}...")

        messages_content_list = []
        messages_content_list.append({"type": "text", "text": text_prompt_part})

        # Prepare the screenshot for the LLM 
        if current_screenshot_bytes:
            try:
                base64_image = base64.b64encode(current_screenshot_bytes).decode('utf-8')
                messages_content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                    }
                })
                self.logger.debug("Screenshot image prepared for multi-modal LLM.")
            except Exception as e:
                self.logger.error(f"Failed to encode screenshot for LLM: {e}. Proceeding with text prompt only.")
        
        messages = [{"role": "user", "content": messages_content_list}] # type: ignore

        # Send the request to the LLM
        try:
            api_response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=MAX_PLANNER_TOKENS,
                temperature=PLANNER_TEMPERATURE,
            )
            response_content = api_response.choices[0].message.content
            if response_content:
                response_content = response_content.strip()
            else:
                self.logger.error("LLM response content was None.")
                return None, {"raw_response": None, "model_used": self.model_name, "error": "No content in response"}

            plan = self._parse_plan_from_response(response_content)
            
            usage_info = api_response.usage
            metadata = {
                "raw_response": response_content,
                "model_used": api_response.model if hasattr(api_response, 'model') else self.model_name,
                "prompt_tokens": usage_info.prompt_tokens if usage_info else None,
                "completion_tokens": usage_info.completion_tokens if usage_info else None,
                "total_tokens": usage_info.total_tokens if usage_info else None,
                "image_sent": any(isinstance(item, dict) and item.get("type") == "image_url" for item in messages_content_list)
            }
            self.logger.debug(f"Planner metadata: {metadata}")

            if not plan:
                self.logger.error("Planner API call successful, but failed to parse a valid plan from the response.")
                return None, metadata

            # Return the created plan and metadata
            return plan, metadata

        except openai.APIError as e:
            self.logger.error(f"Planner OpenAI API error ({type(e).__name__}): {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error in Planner's get_plan method: {e}")

        return None, None