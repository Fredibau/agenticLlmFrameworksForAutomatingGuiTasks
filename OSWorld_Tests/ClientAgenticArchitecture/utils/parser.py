# mm_agents/uitars_agent.py

import ast # Used for robust string literal parsing (e.g., quoted strings)
import base64
import logging
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
import time

from PIL import Image
# import openai # openai import is now within the class if needed by other methods
from openai import OpenAI # Specifically import OpenAI for the client

logger = logging.getLogger("desktopenv.agent.uitars_agent")

# --- Global Action Constants ---
FINISH_WORD = "finished"
WAIT_WORD = "wait"
CALL_USER = "call_user"
ENV_FAIL_WORD = "error_env" # This is used by the agent logic

# --- New Parser Functions (Provided by User) ---
# Note: The constants FINISH_WORD, WAIT_WORD, CALL_USER are defined above globally.

def parse_value_string(value_string: str, logger: logging.Logger = None) -> Any:
    """
    Attempts to parse a string representation of a parameter value into a Python type.
    Handles quoted strings, coordinates (after pre-processing), booleans, numbers, etc.
    """
    original_value_string = value_string
    value_string = value_string.strip()

    if logger: logger.debug(f"Attempting to parse value string: '{original_value_string}'")

    # 1. Try parsing as coordinate string FIRST if it's not explicitly quoted by VLM (e.g. param=(1,2) )
    if not ((value_string.startswith("'") and value_string.endswith("'")) or \
            (value_string.startswith('"') and value_string.endswith('"'))):
        coord_match = re.match(r'^\((\d+),\s*(\d+)\)$', value_string)
        if coord_match:
            try:
                x, y = int(coord_match.group(1)), int(coord_match.group(2))
                parsed_value = [x, y]
                if logger: logger.debug(f"Parsed directly as coordinate [x, y]: {parsed_value}")
                return parsed_value
            except (ValueError, TypeError) as e:
                if logger: logger.debug(f"Direct coordinate parse failed for '{value_string}': {e}. Proceeding.")
    
    # 2. Handle Quoted Strings
    if (value_string.startswith("'") and value_string.endswith("'") and len(value_string) >= 2) or \
       (value_string.startswith('"') and value_string.endswith('"') and len(value_string) >= 2):
        try:
            if re.match(r"""^(['"]).*\1$""", value_string, re.DOTALL): 
                value_after_ast_eval = ast.literal_eval(value_string)
                
                if isinstance(value_after_ast_eval, str):
                    coord_match_after_ast = re.match(r'^\((\d+),\s*(\d+)\)$', value_after_ast_eval) # Check if the string *content* is coords
                    if coord_match_after_ast:
                        try:
                            x, y = int(coord_match_after_ast.group(1)), int(coord_match_after_ast.group(2))
                            final_parsed_value = [x, y]
                            if logger: logger.debug(f"Parsed (after unquoting by ast) as coordinate [x, y]: {final_parsed_value}")
                            return final_parsed_value
                        except (ValueError, TypeError):
                            if logger: logger.debug(f"String from ast '{value_after_ast_eval}' looked like coords but failed int conversion. Returning as string.")
                    if logger: logger.debug(f"Parsed as quoted string (ast.literal_eval gave string): '{value_after_ast_eval}'")
                    return value_after_ast_eval
                else:
                    if logger: logger.debug(f"Parsed as quoted literal (ast.literal_eval non-string): {value_after_ast_eval} (type: {type(value_after_ast_eval).__name__})")
                    return value_after_ast_eval
            else: 
                if logger: logger.debug(f"Value string '{value_string}' looks quoted but not a standard literal. Stripping outer quotes.")
                return value_string[1:-1] if len(value_string) >=2 else ""
        except (ValueError, SyntaxError) as e: 
            if logger: logger.warning(f"ast.literal_eval failed on quoted string '{value_string}': {e}. Stripping outer quotes as fallback.")
            return value_string[1:-1] if len(value_string) >=2 else ""

    # 3. Handle Booleans (case-insensitive)
    if value_string.lower() == 'true':
        if logger: logger.debug("Parsed as boolean: True")
        return True
    if value_string.lower() == 'false':
        if logger: logger.debug("Parsed as boolean: False")
        return False

    # 4. Handle None
    if value_string.lower() == 'none':
        if logger: logger.debug("Parsed as NoneType: None")
        return None

    # 5. Handle Numbers (Integer or Float)
    try:
        parsed_value = int(value_string)
        if logger: logger.debug(f"Parsed as integer: {parsed_value}")
        return parsed_value
    except ValueError:
        pass

    try:
        parsed_value = float(value_string)
        if logger: logger.debug(f"Parsed as float: {parsed_value}")
        return parsed_value
    except ValueError:
        pass

    # 6. Default: Return as a cleaned-up string
    if logger: logger.debug(f"No specific type matched for '{original_value_string}', returning as string: '{value_string}'")
    return value_string


def parse_single_action(action_str: str, logger: logging.Logger = None) -> Optional[Dict]:
    """
    Helper function to parse a single action string.
    """
    if logger: logger.debug(f"Parsing single action string: '{action_str}'")
    action_str = action_str.strip()

    match = re.match(r"(\w+)\s*\((.*)\)\s*$", action_str, re.DOTALL) 
    if not match:
        if logger: logger.warning(f"Failed to match basic action_name(arguments) pattern for: '{action_str}'")
        simple_action_match = re.match(r"^\s*(\w+)\s*$", action_str)
        if simple_action_match:
            action_type_raw_simple = simple_action_match.group(1).lower()
            # Check against globally defined constants
            if action_type_raw_simple in [FINISH_WORD.lower(), WAIT_WORD.lower(), CALL_USER.lower()]:
                if logger: logger.debug(f"Matched simple action word: {action_type_raw_simple}")
                # Return type consistent with global constants' casing if applicable
                act_type_to_return = action_type_raw_simple
                if action_type_raw_simple == FINISH_WORD.lower(): act_type_to_return = FINISH_WORD
                elif action_type_raw_simple == WAIT_WORD.lower(): act_type_to_return = WAIT_WORD
                elif action_type_raw_simple == CALL_USER.lower(): act_type_to_return = CALL_USER
                return {'type': act_type_to_return, 'inputs_raw_string': action_str, 'inputs': {}}
        return None

    action_type_raw = match.group(1)
    args_str = match.group(2).strip()
    action_inputs_parsed: Dict[str, Any] = {}

    if args_str: 
        param_strings: List[str] = []
        current_param = ""
        in_quotes = False
        paren_level = 0
        bracket_level = 0
        curly_level = 0
        quote_char = None

        for i, char in enumerate(args_str):
            is_escaped = (i > 0 and args_str[i-1] == '\\')
            if char in ("'", '"') and not is_escaped:
                if not in_quotes: in_quotes = True; quote_char = char
                elif char == quote_char: in_quotes = False; quote_char = None
            elif char == '(' and not in_quotes: paren_level += 1
            elif char == ')' and not in_quotes: paren_level = max(0, paren_level - 1)
            elif char == '[' and not in_quotes: bracket_level += 1
            elif char == ']' and not in_quotes: bracket_level = max(0, bracket_level - 1)
            elif char == '{' and not in_quotes: curly_level += 1
            elif char == '}' and not in_quotes: curly_level = max(0, curly_level - 1)
            
            if char == ',' and not in_quotes and paren_level == 0 and bracket_level == 0 and curly_level == 0:
                if current_param.strip(): param_strings.append(current_param.strip())
                current_param = ""
            else:
                current_param += char
        
        if current_param.strip(): param_strings.append(current_param.strip())
        if logger: logger.debug(f"Split args string '{args_str}' into raw parameter strings: {param_strings}")

        current_pos_arg_index = 0
        for param_string in param_strings:
            key = None
            value_string_raw = param_string # Keep original for logging if needed
            
            eq_index = -1
            temp_in_quotes = False; temp_q_char = None
            temp_p_lvl, temp_b_lvl, temp_c_lvl = 0,0,0
            for i_eq, char_eq in enumerate(param_string):
                is_esc_eq = (i_eq > 0 and param_string[i_eq-1] == '\\')
                if char_eq in ("'", '"') and not is_esc_eq:
                    if not temp_in_quotes: temp_in_quotes = True; temp_q_char = char_eq
                    elif char_eq == temp_q_char: temp_in_quotes = False; temp_q_char = None
                elif char_eq == '(' and not temp_in_quotes: temp_p_lvl +=1
                elif char_eq == ')' and not temp_in_quotes: temp_p_lvl = max(0, temp_p_lvl-1)
                # (add bracket and curly level checks here too if they can contain '=')
                elif char_eq == '=' and not temp_in_quotes and temp_p_lvl == 0 and temp_b_lvl == 0 and temp_c_lvl == 0:
                    eq_index = i_eq; break
            
            if eq_index != -1:
                key = param_string[:eq_index].strip()
                value_string_for_parser = param_string[eq_index + 1:].strip()
                if not key: key = None; # Treat '=value' as positional if key is empty
            else:
                value_string_for_parser = param_string.strip()
            
            parsed_value = parse_value_string(value_string_for_parser, logger)

            if key:
                action_inputs_parsed[key] = parsed_value
                if logger: logger.debug(f"Parsed KW parameter '{key}': {parsed_value} (from value string '{value_string_for_parser}')")
            elif value_string_raw.strip(): 
                param_name = None
                act_type_lower = action_type_raw.lower()
                if act_type_lower == "type" and current_pos_arg_index == 0: param_name = "content"
                elif act_type_lower == "hotkey" and current_pos_arg_index == 0: param_name = "key"
                elif act_type_lower in ["click", "left_double", "right_single", "hover", "drag"] and \
                     current_pos_arg_index == 0 and isinstance(parsed_value, list) and len(parsed_value) == 2:
                    param_name = "start_box"
                elif act_type_lower == "drag" and current_pos_arg_index == 1 and \
                     isinstance(parsed_value, list) and len(parsed_value) == 2:
                    param_name = "end_box"
                elif act_type_lower == "scroll":
                    if current_pos_arg_index == 0:
                        if isinstance(parsed_value, list) and len(parsed_value) == 2: param_name = "start_box"
                        elif isinstance(parsed_value, str): param_name = "direction"
                    elif current_pos_arg_index == 1 and isinstance(parsed_value, str): param_name = "direction"
                
                if param_name:
                    action_inputs_parsed[param_name] = parsed_value
                    if logger: logger.debug(f"Mapped positional arg {current_pos_arg_index} to '{param_name}': {parsed_value}")
                else:
                    if logger: logger.warning(f"Ignoring unmapped positional argument '{param_string}' (parsed as {parsed_value}) for action '{action_type_raw}' at index {current_pos_arg_index}.")
                current_pos_arg_index += 1

    action_type = action_type_raw.lower()
    # Standardize special action types from constants (using .lower() for comparison)
    if action_type == FINISH_WORD.lower(): action_type_to_set = FINISH_WORD
    elif action_type == WAIT_WORD.lower(): action_type_to_set = WAIT_WORD
    elif action_type == CALL_USER.lower(): action_type_to_set = CALL_USER
    # ENV_FAIL_WORD is not in the new parser's simple list, so it needs to be action_type_raw.lower()
    # The agent logic downstream will handle its specific string value.
    else: action_type_to_set = action_type 


    parsed_action_dict: Dict[str, Any] = {
        'type': action_type_to_set, 'inputs_raw_string': action_str, 'inputs': action_inputs_parsed,
        'coordinates': None, 'content': None, 'key': None, 'direction': None,
        'start_coordinates': None, 'end_coordinates': None,
    }

    raw_start_box = action_inputs_parsed.get("start_box")
    if isinstance(raw_start_box, list) and len(raw_start_box) == 2:
        parsed_action_dict['start_coordinates'] = raw_start_box
        if action_type in ["click", "left_double", "right_single", "scroll", "hover"]: # scroll can have optional coordinates
            parsed_action_dict['coordinates'] = raw_start_box
    elif raw_start_box is not None and logger:
        logger.warning(f"Found 'start_box' but it's not a coordinate list: {raw_start_box} for action {action_type}")

    raw_end_box = action_inputs_parsed.get("end_box")
    if isinstance(raw_end_box, list) and len(raw_end_box) == 2:
        parsed_action_dict['end_coordinates'] = raw_end_box
    elif raw_end_box is not None and logger:
        logger.warning(f"Found 'end_box' but it's not a coordinate list: {raw_end_box} for action {action_type}")

    if 'content' in action_inputs_parsed: parsed_action_dict['content'] = action_inputs_parsed['content']
    if 'key' in action_inputs_parsed: parsed_action_dict['key'] = action_inputs_parsed['key']
    
    raw_direction = action_inputs_parsed.get("direction")
    if isinstance(raw_direction, str): parsed_action_dict['direction'] = raw_direction.lower()
    elif raw_direction is not None and logger:
        logger.warning(f"Found 'direction' but it's not a string: {raw_direction} for action {action_type}")

    if logger: logger.debug(f"Parsed action dict before final validation: {parsed_action_dict}")

    # Final Validation (using the 'type' from parsed_action_dict which may have been standardized)
    current_action_type_for_validation = parsed_action_dict['type']
    if current_action_type_for_validation in ["click", "left_double", "right_single", "hover"]:
        if not isinstance(parsed_action_dict['coordinates'], list) or len(parsed_action_dict['coordinates']) != 2 :
            if logger: logger.error(f"Validation FAIL: Action '{current_action_type_for_validation}' requires valid 'coordinates' list [x,y]. Got: {parsed_action_dict['coordinates']}")
            return None
    if current_action_type_for_validation == "scroll":
        if not isinstance(parsed_action_dict['direction'], str) or parsed_action_dict['direction'] not in ['up', 'down', 'left', 'right']:
            if logger: logger.error(f"Validation FAIL: Action 'scroll' requires valid 'direction'. Got: {parsed_action_dict['direction']}")
            return None
    if current_action_type_for_validation == "drag":
        if not (isinstance(parsed_action_dict['start_coordinates'], list) and len(parsed_action_dict['start_coordinates'])==2 and \
                isinstance(parsed_action_dict['end_coordinates'], list) and len(parsed_action_dict['end_coordinates'])==2):
            if logger: logger.error(f"Validation FAIL: Action 'drag' requires valid 'start_coordinates' and 'end_coordinates'. Got S: {parsed_action_dict['start_coordinates']}, E: {parsed_action_dict['end_coordinates']}")
            return None
    if current_action_type_for_validation == "type": 
        if parsed_action_dict['content'] is None: 
            parsed_action_dict['content'] = "" 
            if logger: logger.debug("Validation: 'type' action had None content, defaulted to empty string.")
    if current_action_type_for_validation == "hotkey":
        if not isinstance(parsed_action_dict['key'], str) or not parsed_action_dict['key'].strip():
            if logger: logger.error(f"Validation FAIL: Action 'hotkey' requires a non-empty 'key' string. Got: {parsed_action_dict['key']}")
            return None

    if logger: logger.debug(f"Action '{current_action_type_for_validation}' passed validation. Final dict: {parsed_action_dict}")
    return parsed_action_dict


def parse_llm_output(text: str, logger: logging.Logger = None) -> Tuple[List[Dict], Optional[str]]:
    """
    Parse model actions and thoughts from text output.
    """
    if logger: logger.debug(f"Attempting to parse raw LLM output:\n---\n{text}\n---")
    text = text.strip()
    thought = None
    action_str_part = text # Default if no specific "Action:" section found

    # 1. Attempt to extract thought using explicit prefixes
    thought_match = re.search(r"(Thought:|Reflection:)\s*(.*?)(?=\s*Action:|$)", text, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(2).strip()
        if logger: logger.debug(f"Extracted thought using prefix: {thought}")
        # Update action_str_part to be what's after the thought block
        # Find where the full thought_match group ends to get text after it
        action_str_part = text[thought_match.end():].strip() 
    else: 
        # 2. If no explicit prefix thought, check for text before the first "Action:"
        first_action_keyword_match = re.search(r"Action:", text, re.IGNORECASE)
        if first_action_keyword_match:
            potential_thought_block = text[:first_action_keyword_match.start()].strip()
            if len(potential_thought_block.split()) > 3 or re.search(r'[a-zA-Z0-9]{5,}', potential_thought_block):
                thought = potential_thought_block
                if logger: logger.debug(f"Extracted potential thought from text before 'Action:': '{thought}'")
            elif logger and potential_thought_block:
                logger.debug(f"Text before 'Action:' ('{potential_thought_block}') not substantial enough for thought.")
            action_str_part = text[first_action_keyword_match.end():].strip() # Get text after "Action:"
        elif logger:
            logger.debug("No 'Thought:'/'Reflection:' prefix and no 'Action:' keyword found to delimit thought. Action part is whole text.")


    # 3. Strip "Action:" prefix from action_str_part if it's still there
    # (Could happen if thought extraction logic didn't consume it)
    if action_str_part.lower().startswith("action:"):
        action_str_part = action_str_part[len("Action:"):].strip()
    
    # 4. Pre-process the action string part
    processed_action_str_part = re.sub(r'<\|box_start\|>\s*\((\s*\d+\s*,\s*\d+\s*)\)\s*<\|box_end\|>', r'(\1)', action_str_part)
    processed_action_str_part = re.sub(r'<\|box_start\|>\s*\((.+?)\)\s*<\|box_end\|>', r'(\1)', processed_action_str_part) # More general fallback
    if logger: logger.debug(f"Processed action part for splitting: '{processed_action_str_part}'")

    # 5. Split actions by newline characters that are NOT inside parentheses, brackets, or quotes
    raw_action_strings: List[str] = []
    current_action_string_assembler = ""
    in_quotes_split = False
    paren_level_split = 0
    bracket_level_split = 0
    curly_level_split = 0
    quote_char_split = None

    for i, char_split in enumerate(processed_action_str_part):
        is_escaped_split = (i > 0 and processed_action_str_part[i-1] == '\\')

        if char_split in ("'", '"') and not is_escaped_split:
            if not in_quotes_split: in_quotes_split = True; quote_char_split = char_split
            elif char_split == quote_char_split: in_quotes_split = False; quote_char_split = None
        elif char_split == '(' and not in_quotes_split: paren_level_split += 1
        elif char_split == ')' and not in_quotes_split: paren_level_split = max(0, paren_level_split - 1)
        elif char_split == '[' and not in_quotes_split: bracket_level_split += 1
        elif char_split == ']' and not in_quotes_split: bracket_level_split = max(0, bracket_level_split - 1)
        elif char_split == '{' and not in_quotes_split: curly_level_split += 1
        elif char_split == '}' and not in_quotes_split: curly_level_split = max(0, curly_level_split - 1)
        
        current_action_string_assembler += char_split # Append char regardless first

        # Split on newline only if not protected and assembler is not empty
        if char_split == '\n' and not in_quotes_split and \
           paren_level_split == 0 and bracket_level_split == 0 and curly_level_split == 0:
            # We've hit a newline that's a potential separator.
            # The current_action_string_assembler includes this newline.
            # We want to add the part *before* this newline.
            to_add = current_action_string_assembler[:-1].strip() # Exclude the newline itself
            if to_add:
                raw_action_strings.append(to_add)
            current_action_string_assembler = "" # Reset for the next action string

    # Add any remaining part after the last valid newline separator or if no newlines
    if current_action_string_assembler.strip():
        raw_action_strings.append(current_action_string_assembler.strip())

    if not raw_action_strings and processed_action_str_part.strip():
        if logger: logger.debug("No newlines found or complex structure, treating entire processed part as one action string.")
        raw_action_strings.append(processed_action_str_part.strip())
    if logger: logger.debug(f"Split into raw action strings: {raw_action_strings}")

    # 6. Parse each individual action string
    parsed_actions = []
    for act_str_to_parse in raw_action_strings:
        if not act_str_to_parse: continue
        parsed_action = parse_single_action(act_str_to_parse, logger)
        if parsed_action:
            parsed_actions.append(parsed_action)
        elif logger:
            logger.warning(f"Failed to parse single action string: '{act_str_to_parse}' (parse_single_action returned None).")

    if not parsed_actions and action_str_part and thought and not raw_action_strings:
         if logger: logger.info("No actions parsed via split, but 'Action:' was present. The content after 'Action:' might be part of thought or malformed.")

    return parsed_actions, thought


def extract_coordinates_from_action(action_str: str, logger: logging.Logger = None) -> Optional[List[int]]:
    """
    (Legacy-style, direct extraction) Extract *any* (x,y) coordinate tuple from a string.
    """
    try:
        if logger: logger.debug(f"(Legacy extract_coordinates) Attempting from: {action_str}")
        coords_match = re.search(r"\((\s*\d+\s*,\s*\d+\s*)\)", action_str) 
        if coords_match:
            coord_pair_str = coords_match.group(1).split(',')
            x, y = int(coord_pair_str[0].strip()), int(coord_pair_str[1].strip())
            if logger: logger.debug(f"(Legacy extract_coordinates) Found: [{x}, {y}]")
            return [x, y]
        if logger: logger.warning(f"(Legacy extract_coordinates) No (x,y) pattern found in: {action_str}")
        return None
    except Exception as e:
        if logger: logger.error(f"(Legacy extract_coordinates) Failed: {e}")
        return None

# --- Existing Helper Functions ---
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

def escape_for_pyautogui_write(text: Optional[str]) -> str:
    if text is None: return ""
    # More comprehensive escape for pyautogui.write, especially for single quotes within the text
    return str(text).replace("\\", "\\\\").replace("'", "\\'") # Escape backslashes then single quotes

def parsing_response_to_pyautogui_code_v2( # Name kept for now, can be renamed if desired
    parsed_action: Dict, image_height: int, image_width: int
) -> str:
    py_cmds = []
    action_type = parsed_action.get("type") # This should be the standardized type

    # Use standardized keys from the new parser's output
    # e.g., parsed_action.get('coordinates'), parsed_action.get('content'), etc.

    if action_type == "hotkey":
        key_val = parsed_action.get("key")
        if key_val and isinstance(key_val, str):
            keys = [repr(k.strip().lower()) for k in key_val.split('+') if k.strip()]
            if keys: py_cmds.append(f"pyautogui.hotkey({', '.join(keys)})")
        else: py_cmds.append(f"# CodeGenErr: Invalid hotkey value: {key_val}")
    elif action_type == "type":
        content_to_type = parsed_action.get("content", "") 
        # The new parser should handle unquoting, escape_for_pyautogui_write handles escaping for the script
        escaped_content = escape_for_pyautogui_write(str(content_to_type)) # Ensure it's a string
        py_cmds.append(f"pyautogui.write('{escaped_content}', interval=0.02)")
        # The new parser does not add "\n" for enter, it should be explicit in 'content'
        # However, the prompt template says: #If you want to submit your input, use "\\n" at the end of `content`.
        # So, if "\\n" (literal backslash n) is in content, it will be typed. If actual newline char, pyautogui.write handles it.
        # Let's assume if LLM intends an Enter, it puts a newline character or literal "\\n" in the content.
        # Pyautogui `write` types characters including newline for enter.
        # If literal "\\n" is desired for 'enter' key press:
        if isinstance(content_to_type, str) and content_to_type.endswith("\\n"):
             # py_cmds.append("pyautogui.press('enter')") # If \\n means press enter
             pass # pyautogui.write already handles \n character. If it's literal \\n, it types those.
                  # The previous code for type() had a specific check for '\n' in content to press enter.
                  # With robust parsing, if content is "abc\n", pyautogui.write types abc then enter.
                  # This is probably sufficient.

    elif action_type in ["drag", "select"]: # 'select' is not in new parser, but kept for compatibility if used by LLM
        s_coords_1k = parsed_action.get("start_coordinates")
        e_coords_1k = parsed_action.get("end_coordinates")
        if isinstance(s_coords_1k, list) and len(s_coords_1k) == 2 and \
           isinstance(e_coords_1k, list) and len(e_coords_1k) == 2 :
            # Assuming coordinates are already 0-1000 range as per previous logic
            sx, sy = round(s_coords_1k[0]/1000.0*image_width), round(s_coords_1k[1]/1000.0*image_height)
            ex, ey = round(e_coords_1k[0]/1000.0*image_width), round(e_coords_1k[1]/1000.0*image_height)
            sx,sy = max(0,min(sx,image_width-1)), max(0,min(sy,image_height-1))
            ex,ey = max(0,min(ex,image_width-1)), max(0,min(ey,image_height-1))
            py_cmds.extend([f"pyautogui.moveTo({sx}, {sy}, duration=0.2)",
                            f"pyautogui.dragTo({ex}, {ey}, duration=0.5, button='left')"])
        else: py_cmds.append(f"# CodeGenErr: Drag/Select missing/invalid coords S:{s_coords_1k} E:{e_coords_1k}")
    elif action_type == "scroll":
        direction = parsed_action.get("direction", "").lower() # Already lowercased by parser
        coords_1k = parsed_action.get("coordinates") # Optional
        scroll_args_list = []
        if isinstance(coords_1k, list) and len(coords_1k) == 2: # If coordinates are provided for scroll
            x,y = round(coords_1k[0]/1000.0*image_width), round(coords_1k[1]/1000.0*image_height)
            x,y = max(0,min(x,image_width-1)), max(0,min(y,image_height-1))
            py_cmds.append(f"pyautogui.moveTo({x}, {y}, duration=0.2)") # Move to coords before scroll
            scroll_args_list.extend([f"x={x}", f"y={y}"])
        
        scroll_amount = 200 # Default scroll amount
        scroll_args_str = (", " + ", ".join(scroll_args_list)) if scroll_args_list else ""
        if "up" in direction: py_cmds.append(f"pyautogui.scroll({scroll_amount}{scroll_args_str})")
        elif "down" in direction: py_cmds.append(f"pyautogui.scroll({-scroll_amount}{scroll_args_str})")
        elif "left" in direction: py_cmds.append(f"pyautogui.hscroll({scroll_amount}{scroll_args_str})") # PyAutoGUI uses positive for left with hscroll
        elif "right" in direction: py_cmds.append(f"pyautogui.hscroll({-scroll_amount}{scroll_args_str})")# PyAutoGUI uses negative for right with hscroll
        else: py_cmds.append(f"# CodeGenErr: Scroll invalid direction: {direction}")
    elif action_type in ["click", "left_double", "right_single", "hover"]:
        coords_1k = parsed_action.get("coordinates")
        if isinstance(coords_1k, list) and len(coords_1k) == 2:
            x,y = round(coords_1k[0]/1000.0*image_width), round(coords_1k[1]/1000.0*image_height)
            x,y = max(0,min(x,image_width-1)), max(0,min(y,image_height-1))
            py_cmds.append(f"pyautogui.moveTo({x}, {y}, duration=0.2)")
            if action_type == "click" or action_type == "left_single": py_cmds.append(f"pyautogui.click(button='left')")
            elif action_type == "left_double": py_cmds.append(f"pyautogui.doubleClick(button='left')")
            elif action_type == "right_single": py_cmds.append(f"pyautogui.click(button='right')")
            # For "hover", moveTo is enough.
        else: py_cmds.append(f"# CodeGenErr: {action_type} missing/invalid coords: {coords_1k}")
    
    # Handle special action types that are just signals
    elif action_type == WAIT_WORD: py_cmds.append(f"time.sleep(5) # Action: {WAIT_WORD}")
    elif action_type == FINISH_WORD: py_cmds.append(f"# Signal: {FINISH_WORD}")
    elif action_type == CALL_USER: py_cmds.append(f"# Signal: {CALL_USER}")
    elif action_type == ENV_FAIL_WORD: py_cmds.append(f"# Signal: {ENV_FAIL_WORD}") # Agent logic handles this
    else: py_cmds.append(f"# Unrecognized action type in pyautogui codegen: {action_type}")
    
    return "\n".join(py_cmds) if py_cmds else f"# No command generated for {action_type}"


def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

class UITARSAgent:
    def __init__(
        self,
        model: str,
        max_tokens: int,
        top_p: float,
        temperature: float,
        observation_type: str,
        max_trajectory_length: int,
        action_space: str, 
        screen_width: int = 1920,
        screen_height: int = 1080,
        vlm_base_url: str = "http://127.0.0.1:4000/v1",
        vlm_api_key: str = "EMPTY",
        language: str = "English",
        history_n: int = 3, 
        max_agent_steps: int = 30,
    ):
        self.max_tokens_vlm_resp = max_tokens
        self.top_p_vlm = top_p
        self.temperature_vlm = temperature

        if observation_type != "screenshot":
            logger.warning(f"Agent received observation_type='{observation_type}', "
                           f"but is configured for 'screenshot' only. Proceeding in screenshot-only mode.")
        self.internal_observation_type = "screenshot"

        self.max_trajectory_len_agent = max_trajectory_length
        self.action_space = action_space

        self.vlm_model_name = model
        self.vlm_base_url = vlm_base_url
        self.vlm_api_key = vlm_api_key

        self.language = language
        self.history_n = history_n
        self.max_steps_per_task = max_agent_steps
        self.screen_height = screen_height
        self.screen_width = screen_width

        try:
            self.vlm = OpenAI(
                base_url=self.vlm_base_url,
                api_key=self.vlm_api_key,
            )
            logger.info(f"UITARSAgent VLM client targeting: {self.vlm_base_url} with model {self.vlm_model_name}")
        except Exception as e:
            logger.critical(f"Failed to initialize OpenAI client: {e}", exc_info=True); raise

        self.thoughts: List[str] = []
        self.actions: List[List[str]] = [] # Stores lists of pyautogui command strings per step
        self.observations: List[Dict[str, Optional[str]]] = [] 
        
        self.history_images: List[bytes] = []
        self.history_responses: List[str] = []

        self.prompt_action_space_text = WORKER_ACTION_SPACE
        self.prompt_template_str = WORKER_USER_PROMPT_TEMPLATE
        
        logger.info(f"UITARSAgent initialized. Effective ObservationMode: {self.internal_observation_type}, ActionSpace for Env: {self.action_space}, Screen: {self.screen_width}x{self.screen_height}, VLM History Turns: {self.history_n}")

    def predict(self, instruction: str, obs_dict: Dict) -> Tuple[str, List[str]]:
        logger.debug(f"Predict called. Instruction: {instruction[:100]}...")
        logger.debug(f"Obs keys: {obs_dict.keys() if obs_dict else 'None'}")
        
        current_screenshot_bytes: Optional[bytes] = None

        if "screenshot" in obs_dict and obs_dict["screenshot"]:
            screenshot_data_from_obs = obs_dict["screenshot"]
            if isinstance(screenshot_data_from_obs, str):
                try:
                    current_screenshot_bytes = base64.b64decode(screenshot_data_from_obs)
                except Exception as decode_e:
                    logger.error(f"Error decoding base64 screenshot: {decode_e}. String start: {screenshot_data_from_obs[:100]}")
                    return f"ERR_SCREENSHOT_B64_DECODE:{decode_e}", ["# ERR_SCREENSHOT_B64_DECODE"]
            elif isinstance(screenshot_data_from_obs, bytes):
                current_screenshot_bytes = screenshot_data_from_obs
            else:
                logger.error(f"Screenshot is of unexpected type: {type(screenshot_data_from_obs)}.")
                return "ERR_UNEXPECTED_SCREENSHOT_TYPE", ["# ERR_UNEXPECTED_SCREENSHOT_TYPE"]

            if current_screenshot_bytes:
                try:
                    img_verify = Image.open(BytesIO(current_screenshot_bytes))
                    img_verify.verify()
                    logger.debug(f"Current screenshot verified. Format: {img_verify.format}, Size: {img_verify.size}")
                except Exception as verify_e:
                    logger.error(f"CLIENT-SIDE VERIFICATION FAILED: Screenshot not valid. Error: {verify_e}.")
                    log_data = screenshot_data_from_obs[:100] if isinstance(screenshot_data_from_obs, str) else current_screenshot_bytes[:100]
                    logger.error(f"Data that failed verification (first 100 units): {log_data}")
                    return f"ERR_INVALID_SCREENSHOT_CLIENT_SIDE:{verify_e}", ["# ERR_INVALID_SCREENSHOT_CLIENT_SIDE"]
            else:
                 return "ERR_SCREENSHOT_PROCESSING_FAILED", ["# ERR_SCREENSHOT_PROCESSING_FAILED"]
        
        if current_screenshot_bytes:
            self.history_images.append(current_screenshot_bytes)
        elif not self.history_images: 
            logger.error("No screenshot available (current missing/invalid and no historical).")
            return "ERR_NO_IMAGES_FOR_VLM", ["# ERR_NO_IMAGES_FOR_VLM"]
        else: 
            logger.warning("Current screenshot missing or invalid. Proceeding with historical images.")

        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]
        
        base64_image_for_agents_log = None
        if current_screenshot_bytes:
            base64_image_for_agents_log = base64.b64encode(current_screenshot_bytes).decode('utf-8')
        elif obs_dict.get("screenshot") and isinstance(obs_dict.get("screenshot"), str):
            base64_image_for_agents_log = obs_dict.get("screenshot")

        self.observations.append({"screenshot": base64_image_for_agents_log, "accessibility_tree": None})
        if len(self.observations) > self.max_trajectory_len_agent > 0:
            self.observations = self.observations[-self.max_trajectory_len_agent:]

        api_images_to_send = list(self.history_images)
        api_responses_to_interleave = list(self.history_responses)
        expected_prev_responses = max(0, len(api_images_to_send) - 1)
        if len(api_responses_to_interleave) > expected_prev_responses:
             api_responses_to_interleave = api_responses_to_interleave[-expected_prev_responses:]
        
        full_user_prompt_text = self.prompt_template_str.format(
            action_space=self.prompt_action_space_text, language=self.language,
            sub_task_instruction=instruction
        )
        api_messages: List[Dict[str, Any]] = [{"role": "system", "content": [{"type": "text", "text": "You are a precise GUI agent."}]}]
        first_user_turn_content = [{"type": "text", "text": full_user_prompt_text}]
        if api_images_to_send:
            b64_img = pil_to_base64(Image.open(BytesIO(api_images_to_send[0])))
            first_user_turn_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}})
        api_messages.append({"role": "user", "content": first_user_turn_content})

        for i in range(len(api_responses_to_interleave)):
            api_messages.append({"role": "assistant", "content": [{"type": "text", "text": api_responses_to_interleave[i]}]})
            if (i + 1) < len(api_images_to_send): 
                b64_img = pil_to_base64(Image.open(BytesIO(api_images_to_send[i+1])))
                api_messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}]})
        
        raw_vlm_prediction = None; api_err = None
        for attempt in range(3):
            try:
                logger.info(f"VLM Call (Attempt {attempt+1}). Model: {self.vlm_model_name}")
                response = self.vlm.chat.completions.create(
                    model=self.vlm_model_name, messages=api_messages,
                    temperature=self.temperature_vlm, max_tokens=self.max_tokens_vlm_resp, top_p=self.top_p_vlm
                )
                raw_vlm_prediction = response.choices[0].message.content.strip(); api_err = None; break
            except Exception as e:
                api_err = f"VLM API Error (Attempt {attempt+1}): {e}"
                logger.error(api_err, exc_info=(attempt==2))
                if attempt < 2: time.sleep(2 * (attempt + 1))
        
        if api_err or not raw_vlm_prediction:
            self.thoughts.append(api_err or "VLM NO RESPONSE"); self.actions.append(["#VLM_CALL_FAILED"])
            return api_err or "VLM_NO_RESPONSE", ["#VLM_CALL_FAILED"]
        
        self.history_responses.append(raw_vlm_prediction)
        if len(self.history_responses) > self.history_n :
             self.history_responses = self.history_responses[-self.history_n:]
             
        logger.info(f"VLM Raw Prediction: {raw_vlm_prediction[:200]}...")
        
        # UPDATED Call to new parser function
        parsed_action_dicts, extracted_thought = parse_llm_output(raw_vlm_prediction, logger)
        
        self.thoughts.append(extracted_thought or "No thought parsed.")
        if len(self.thoughts) > self.max_trajectory_len_agent > 0:
            self.thoughts = self.thoughts[-self.max_trajectory_len_agent:]
        
        # Error handling for parsing can be more specific if parse_llm_output raises exceptions
        # For now, assuming it returns empty list on failure or if no actions.
        if not parsed_action_dicts and raw_vlm_prediction: # If parsing gave nothing but there was a prediction
             logger.warning("parse_llm_output did not yield any structured actions. Checking raw prediction for signals.")
             # Fallback logic based on raw_vlm_prediction will run if parsed_action_dicts is empty

        step_pyautogui_commands = []; return_signal_list = None
        if not parsed_action_dicts: # No structured actions parsed
            # Default to WAIT or check raw prediction for keywords
            if FINISH_WORD in raw_vlm_prediction.lower(): return_signal_list = ["DONE"]
            elif WAIT_WORD in raw_vlm_prediction.lower(): return_signal_list = ["WAIT"]
            elif CALL_USER in raw_vlm_prediction.lower(): return_signal_list = ["FAIL_CALL_USER"] 
            else: 
                logger.info("No specific signals in raw prediction, defaulting to WAIT.")
                return_signal_list = ["WAIT"]
                step_pyautogui_commands.append("time.sleep(5) # Default wait due to no parsed actions/signals")
        
        for pa_dict in parsed_action_dicts:
            action_type = pa_dict.get("type") # This is the crucial 'type' from the new parser
            cmd_str = parsing_response_to_pyautogui_code_v2(pa_dict, self.screen_height, self.screen_width)
            
            # Check against global constants for signals
            if action_type == FINISH_WORD: return_signal_list = ["DONE"]; step_pyautogui_commands.append(cmd_str); break
            if action_type == WAIT_WORD: return_signal_list = ["WAIT"]; step_pyautogui_commands.append(cmd_str); break 
            if action_type == CALL_USER: return_signal_list = ["FAIL_CALL_USER"]; step_pyautogui_commands.append(cmd_str); break 
            if action_type == ENV_FAIL_WORD: return_signal_list = ["FAIL"]; step_pyautogui_commands.append(cmd_str); break
            
            if cmd_str and not cmd_str.startswith("# Signal:") and not cmd_str.startswith("# No command generated for"):
                step_pyautogui_commands.append(cmd_str)
            elif cmd_str: 
                logger.info(f"Generated non-command code or signal for {action_type}: {cmd_str}")


        full_script_str = ""
        if step_pyautogui_commands:
            imports = "import pyautogui\nimport time\n"
            thought_comment_str = self.thoughts[-1] if self.thoughts else 'N/A (no thought extracted or recorded)'
            thought_comment = f"\n'''\n--- Agent Thought ---\n{thought_comment_str}\n'''\n"
            full_script_str = imports + thought_comment + "\n".join(step_pyautogui_commands)
            if any(cmd for cmd in step_pyautogui_commands if not cmd.startswith("#") and "time.sleep(5)" not in cmd):
                full_script_str += "\ntime.sleep(0.5) # UI stabilization"
        elif not return_signal_list: 
            thought_comment_str = self.thoughts[-1] if self.thoughts else 'N/A'
            full_script_str = f"# No executable actions. Thought: {thought_comment_str}"
            return_signal_list = ["# NO_ACTION_SCRIPT"] 

        self.actions.append(step_pyautogui_commands if step_pyautogui_commands else (return_signal_list or ["#NO_CMDS"]))
        if len(self.actions) > self.max_trajectory_len_agent > 0:
            self.actions = self.actions[-self.max_trajectory_len_agent:]

        if len(self.history_responses) >= self.max_steps_per_task :
            logger.info(f"Max steps ({self.max_steps_per_task}) reached.")
            return raw_vlm_prediction, ["FAIL_MAX_STEPS"]
        
        if return_signal_list: return raw_vlm_prediction, return_signal_list
        return raw_vlm_prediction, [full_script_str] if full_script_str else ["# NO_COMMANDS_FINAL"]

    def reset(self, runtime_logger: Optional[logging.Logger] = None):
        self.thoughts = []
        self.actions = []
        self.observations = []
        
        self.history_images = []
        self.history_responses = []
        log_to_use = runtime_logger or logger
        log_to_use.info(f"UITARSAgent ({self.vlm_model_name}) has been reset.")