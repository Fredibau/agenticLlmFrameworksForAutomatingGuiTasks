import ast 
import base64
import logging
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
import time

from PIL import Image
from openai import OpenAI 

logger = logging.getLogger("desktopenv.agent.uitars_agent")

# --- Global Action Constants ---
FINISH_WORD = "finished"
WAIT_WORD = "wait"
CALL_USER = "call_user"
ENV_FAIL_WORD = "error_env" 


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
