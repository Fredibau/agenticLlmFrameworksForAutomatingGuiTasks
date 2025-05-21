"""
Utility functions for executing actions from parsed data.
"""
import pyautogui
import time
import logging
from typing import Dict, List, Tuple, Optional

# Assuming coordinate_utils is in the same directory or utils is in PYTHONPATH
from .coordinate_utils import constrain_to_primary_screen

def execute_action(
    parsed_action: Dict,
    raw_action: str,
    primary_width: int,
    primary_height: int,
    logger: logging.Logger
) -> Tuple[bool, Optional[Dict]]:
    """
    Execute the action based on the decision using pyautogui.

    Args:
        parsed_action: Parsed action dict from the VLM.
                       Expected keys based on action type:
                       - 'type': (str) action type (e.g., "click", "left_double", "type")
                       - 'coordinates': (list) [x, y] for point-based actions
                       - 'start_coordinates': (list) [x, y] for drag start
                       - 'end_coordinates': (list) [x, y] for drag end
                       - 'content': (str) for type action
                       - 'key': (str) for hotkey action
                       - 'direction': (str) for scroll action
                       - 'duration': (int, optional) for wait action
        raw_action: Raw action string from VLM for logging/context.
        primary_width: Width of primary screen in pixels.
        primary_height: Height of primary screen in pixels.
        logger: Logger instance for recording information.

    Returns:
        Tuple containing:
        - Boolean indicating if the script should continue running (True to continue, False to stop).
        - Action record dict to be added to history, or None if pre-execution failure.
    """
    action_type = parsed_action.get("type")

    if not action_type:
        logger.error("Action type missing in parsed_action.")
        return False, {"action": "FAIL_MISSING_ACTION_TYPE", "details": "Action type was not found in parsed data."}

    logger.info(f"Attempting to execute action: {action_type} with details: {parsed_action}")

    action_record = {"action": action_type} # Base record for history

    try:
        coordinates = parsed_action.get("coordinates") # Common for point-based actions

        if action_type == "click":
            if coordinates:
                x, y = coordinates
                x_pix, y_pix = constrain_to_primary_screen(x, y, primary_width, primary_height)
                logger.info(f"Clicking at screen coordinates ({x_pix}, {y_pix}) (original VLM: {x},{y}).")
                pyautogui.moveTo(x_pix, y_pix, duration=0.2)
                pyautogui.click()
                action_record["coordinates"] = coordinates
            else:
                logger.error(f"No coordinates found for '{action_type}' action.")
                action_record["details"] = "Missing coordinates"
                action_record["action"] = f"FAIL_ACTION_EXECUTION_{action_type.upper()}"
                return False, action_record

        elif action_type == "left_double":
            if coordinates:
                x, y = coordinates
                x_pix, y_pix = constrain_to_primary_screen(x, y, primary_width, primary_height)
                logger.info(f"Left double-clicking at screen coordinates ({x_pix}, {y_pix}) (original VLM: {x},{y}).")
                pyautogui.moveTo(x_pix, y_pix, duration=0.2)
                pyautogui.doubleClick() # pyautogui.doubleClick() is a left double click
                action_record["coordinates"] = coordinates
            else:
                logger.error(f"No coordinates found for '{action_type}' action.")
                action_record["details"] = "Missing coordinates"
                action_record["action"] = f"FAIL_ACTION_EXECUTION_{action_type.upper()}"
                return False, action_record

        elif action_type == "right_single":
            if coordinates:
                x, y = coordinates
                x_pix, y_pix = constrain_to_primary_screen(x, y, primary_width, primary_height)
                logger.info(f"Right single-clicking at screen coordinates ({x_pix}, {y_pix}) (original VLM: {x},{y}).")
                pyautogui.moveTo(x_pix, y_pix, duration=0.2)
                pyautogui.rightClick()
                action_record["coordinates"] = coordinates
            else:
                logger.error(f"No coordinates found for '{action_type}' action.")
                action_record["details"] = "Missing coordinates"
                action_record["action"] = f"FAIL_ACTION_EXECUTION_{action_type.upper()}"
                return False, action_record

        elif action_type == "type":
            content_to_type = parsed_action.get("content")
            # Allow content_to_type to be an empty string if VLM intends to type nothing (or just press enter via \n)
            if content_to_type is not None:
                logger.info(f"Typing: '{content_to_type}'")
                if '\n' in content_to_type:
                    segments = content_to_type.split('\n')
                    for i, segment in enumerate(segments):
                        if segment:
                            pyautogui.write(segment)
                        if i < len(segments) - 1: 
                            logger.info("Pressing Enter (due to '\\n')")
                            pyautogui.press('enter')
                else:
                    pyautogui.write(content_to_type)
                action_record["content"] = content_to_type
            else:
                logger.error(f"No 'content' (null) found for '{action_type}' action. If intending empty type, VLM should pass empty string.")
                action_record["details"] = "Missing content field" 
                action_record["action"] = f"FAIL_ACTION_EXECUTION_{action_type.upper()}"
                return False, action_record

        elif action_type == "hotkey":
            key_to_press = parsed_action.get("key")
            if key_to_press and isinstance(key_to_press, str): 
                logger.info(f"Pressing hotkey: {key_to_press}")
                keys_to_press_list = [k.strip() for k in key_to_press.split('+')]
                pyautogui.hotkey(*keys_to_press_list)
                action_record["key"] = key_to_press
            else:
                logger.error(f"No or invalid 'key' found for '{action_type}' action: '{key_to_press}'")
                action_record["details"] = "Missing or invalid key string"
                action_record["action"] = f"FAIL_ACTION_EXECUTION_{action_type.upper()}"
                return False, action_record

        elif action_type == "drag":
            start_coords = parsed_action.get("start_coordinates")
            end_coords = parsed_action.get("end_coordinates")
            if start_coords and end_coords:
                start_x, start_y = start_coords
                end_x, end_y = end_coords
                s_x_pix, s_y_pix = constrain_to_primary_screen(start_x, start_y, primary_width, primary_height)
                e_x_pix, e_y_pix = constrain_to_primary_screen(end_x, end_y, primary_width, primary_height)

                logger.info(f"Dragging from ({s_x_pix}, {s_y_pix}) to ({e_x_pix}, {e_y_pix}) "
                            f"(original VLM: {start_coords} to {end_coords}).")
                pyautogui.moveTo(s_x_pix, s_y_pix, duration=0.2)
                pyautogui.dragTo(e_x_pix, e_y_pix, duration=0.5)

                action_record["start_coordinates"] = start_coords
                action_record["end_coordinates"] = end_coords
                action_record["coordinates"] = start_coords # For VLM compatibility if it expects a single 'coordinates' for target
            else:
                logger.error(f"Missing 'start_coordinates' or 'end_coordinates' for '{action_type}' action.")
                action_record["details"] = "Missing start or end coordinates"
                action_record["action"] = f"FAIL_ACTION_EXECUTION_{action_type.upper()}"
                return False, action_record

        elif action_type == "scroll":
            direction = parsed_action.get("direction")
            scroll_amount = 100 # Standard scroll amount

            if coordinates: # Optional: scroll at a specific point
                x, y = coordinates
                x_pix, y_pix = constrain_to_primary_screen(x, y, primary_width, primary_height)
                pyautogui.moveTo(x_pix, y_pix, duration=0.2) # Move mouse to position before scrolling
                logger.info(f"Scrolling at ({x_pix}, {y_pix}) (original VLM: {x},{y}).")
                action_record["coordinates"] = coordinates
            else:
                logger.info("Scrolling at current mouse position.")

            if direction == "down":
                logger.info(f"Scrolling down by {scroll_amount} units.")
                pyautogui.scroll(-scroll_amount)
            elif direction == "up":
                logger.info(f"Scrolling up by {scroll_amount} units.")
                pyautogui.scroll(scroll_amount)
            elif direction == "left":
                logger.info(f"Scrolling left by {scroll_amount} units.")
                pyautogui.hscroll(-scroll_amount)
            elif direction == "right":
                logger.info(f"Scrolling right by {scroll_amount} units.")
                pyautogui.hscroll(scroll_amount)
            else:
                logger.error(f"Invalid or missing 'direction' for '{action_type}' action: '{direction}'. Must be up, down, left, or right.")
                action_record["details"] = f"Invalid scroll direction: {direction}"
                action_record["action"] = f"FAIL_ACTION_EXECUTION_{action_type.upper()}"
                return False, action_record
            action_record["direction"] = direction

        elif action_type == "wait":
            wait_duration = 5 # Default duration from your original spec
            try:
                # Allow VLM to specify duration if parser supports it
                parsed_duration = parsed_action.get("duration")
                if parsed_duration is not None:
                    wait_duration = int(parsed_duration)
                if wait_duration < 0 : wait_duration = 0 # prevent negative wait
            except (ValueError, TypeError):
                logger.warning(f"Invalid duration '{parsed_action.get('duration')}' for wait action, defaulting to {wait_duration}s.")
            
            logger.info(f"Waiting for {wait_duration} seconds.")
            time.sleep(wait_duration)
            action_record["duration"] = wait_duration

        elif action_type == "finished":
            logger.info("Action 'finished' received. Task segment considered complete by VLM.")
            # `False` for should_continue_script signals to Worker to stop processing this sub-task.
            return False, action_record

        elif action_type == "call_user": # Retained for robustness
            logger.info("Action 'call_user' received. Task requires user assistance.")
            return False, action_record # Signals stop and intervention.

        else:
            logger.warning(f"Unknown action type encountered in executor: '{action_type}'. Parsed action: {parsed_action}")
            action_record["details"] = f"Unknown action type: {action_type}"
            action_record["action"] = "FAIL_UNKNOWN_ACTION_TYPE"
            return False, action_record # Cannot execute, so script should not continue.

        # If action was successfully initiated and didn't result in an early False return
        time.sleep(0.5) # Brief pause after most actions for UI to stabilize
        return True, action_record # True to indicate script can continue with next action/step from VLM

    except pyautogui.FailSafeException:
        logger.critical("PyAutoGUI FailSafeException triggered (mouse moved to a corner). Halting action execution.")
        action_record["action"] = "FAIL_PYAUTOGUI_FAILSAFE" # Ensure action_type is set if it failed early
        action_record["details"] = "PyAutoGUI FailSafeException triggered."
        return False, action_record # Stop script execution
    except Exception as e:
        logger.exception(f"An unexpected error occurred during execution of action '{action_type}': {e}")
        action_record["action"] = f"FAIL_UNEXPECTED_ERROR_{action_type.upper() if action_type else 'UNKNOWN'}"
        action_record["details"] = str(e)
        return False, action_record # Stop script execution on unexpected error