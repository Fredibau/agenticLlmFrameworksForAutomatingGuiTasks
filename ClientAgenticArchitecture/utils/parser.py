"""
Utility functions for parsing action strings and LLM outputs.
Improved robustness for parsing action parameters and values.
"""
import logging
import re
import ast # Used for robust string literal parsing (e.g., quoted strings)
from typing import List, Optional, Dict, Tuple, Any


# Define constants for special actions (must match client and executor)
FINISH_WORD = "finished"
WAIT_WORD = "wait"
CALL_USER = "call_user"


def parse_value_string(value_string: str, logger: logging.Logger = None) -> Any:
    """
    Attempts to parse a string representation of a parameter value into a Python type.
    Handles quoted strings, coordinates (after pre-processing), booleans, numbers, etc.

    Args:
        value_string: The raw string value extracted from the action arguments (e.g., '"hello"', '(123,456)', 'True', '5').
        logger: Optional logger.

    Returns:
        The parsed Python value (str, list, bool, int, float, etc.) or the original string if parsing fails.
    """
    original_value_string = value_string
    value_string = value_string.strip()

    if logger: logger.debug(f"Attempting to parse value string: '{original_value_string}'")

    # 1. Handle Coordinate Strings (after <|box_start|> pre-processing, should be '(x,y)')
    # Look for the exact (int, int) pattern with optional whitespace.
    # **DO THIS CHECK BEFORE QUOTED STRINGS**
    coord_match = re.match(r'^\((\d+),\s*(\d+)\)$', value_string)
    if coord_match:
        try:
            x, y = int(coord_match.group(1)), int(coord_match.group(2))
            parsed_value = [x, y] # Store as a list of integers [x, y]
            if logger: logger.debug(f"Parsed as coordinate [x, y]: {parsed_value}")
            return parsed_value
        except (ValueError, TypeError) as e:
            if logger: logger.error(f"Failed to parse coordinate tuple '{value_string}': {e}. Treating as raw string.")
            return original_value_string # Parsing failed, return original string

    # 2. Handle Quoted Strings (most common for content)
    # Use ast.literal_eval for safe parsing of string literals including escape sequences
    # Check for both single and double quotes
    if (value_string.startswith("'") and value_string.endswith("'") and len(value_string) >= 2) or \
       (value_string.startswith('"') and value_string.endswith('"') and len(value_string) >= 2):
        try:
            # ast.literal_eval is safer than eval()
            # Ensure the string looks like a valid literal before trying
            if re.match(r'^[\'"].*[\'"]$', value_string):
                parsed_value = ast.literal_eval(value_string)
                if logger: logger.debug(f"Parsed as quoted string using ast.literal_eval: {parsed_value}")
                return parsed_value
            else:
                 if logger: logger.debug(f"Value string looks quoted but not a valid literal: '{value_string}'")
                 # Fall through to raw string handling
        except (ValueError, SyntaxError) as e:
            if logger: logger.warning(f"ast.literal_eval failed on quoted string '{value_string}': {e}. Treating as raw string after stripping quotes.")
            # Fallback: just strip quotes if literal_eval fails or input format is weirdly quoted
            return value_string.strip("'\"")

    # 3. Handle Booleans (case-insensitive)
    if value_string.lower() == 'true':
        if logger: logger.debug("Parsed as boolean: True")
        return True
    if value_string.lower() == 'false':
        if logger: logger.debug("Parsed as boolean: False")
        return False

    # 4. Handle None (if applicable in prompt)
    if value_string.lower() == 'none':
        if logger: logger.debug("Parsed as None: None")
        return None

    # 5. Handle Numbers (Integer or Float) - check after booleans/none
    try:
        # Try integer first
        parsed_value = int(value_string)
        if logger: logger.debug(f"Parsed as integer: {parsed_value}")
        return parsed_value
    except ValueError:
        pass # Not an integer

    try:
        # Try float if not integer
        parsed_value = float(value_string)
        if logger: logger.debug(f"Parsed as float: {parsed_value}")
        return parsed_value
    except ValueError:
        pass # Not a float


    # 6. Default: Return as a cleaned-up string if no specific type matched
    if logger: logger.debug(f"No specific type matched, returning as string: '{value_string}'")
    return value_string # Return the cleaned-up string


# Replace the existing parse_llm_output function in utils/parser.py with this:

def parse_llm_output(text: str, logger: logging.Logger = None) -> Tuple[List[Dict], Optional[str]]:
    """
    Parse model actions and thoughts from text output.

    Args:
        text: The text containing actions and thought (raw LLM output).
        logger: Optional logger for debugging.

    Returns:
        Tuple containing:
        - List of parsed action dictionaries (can be empty if no actions found).
        - Optional string containing the extracted thought/reflection.
    """
    if logger: logger.debug(f"Attempting to parse raw LLM output:\n---\n{text}\n---")
    text = text.strip()
    thought = None # Initialize thought to None

    # 1. Attempt to extract thought using explicit prefixes (Thought:, Reflection:)
    thought_match = re.search(r"(Thought:|Reflection:)\s*(.*?)(?=\s*Action:|$)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(2).strip()
        if logger: logger.debug(f"Extracted thought using prefix: {thought}")

    # 2. If no explicit prefix thought found, attempt to extract text before the first Action: line
    if thought is None:
        first_action_index = text.lower().find("action:") # Find the index of the first 'action:' (case-insensitive)
        if first_action_index != -1:
            # Extract the text from the start up to the first 'Action:' line
            potential_thought_block = text[:first_action_index].strip()

            # Check if this block of text is substantial enough to be a thought
            # (Basic heuristic: is it more than just a few words or punctuation?)
            if len(potential_thought_block.split()) > 5 or re.search(r'[a-zA-Z0-9]', potential_thought_block): # Has more than 5 words or contains alphanumeric chars
                 thought = potential_thought_block
                 if logger: logger.debug(f"Extracted potential thought from text before Action: {thought}")
            elif logger:
                logger.debug(f"Text before Action: prefix is insubstantial, not treating as thought: '{potential_thought_block}'")

        elif logger:
             # Only log this if NO 'Action:' was found at all (handled later, but good context)
             # Or if thought is None and no text before action.
             logger.debug("No 'Thought:'/'Reflection:' prefix and no text before 'Action:'.")


    # 3. Extract action string part after the last "Action:"
    action_start_index = text.rfind("Action:") # Find the index of the last 'Action:'
    if action_start_index == -1:
         action_start_index = text.rfind("action:") # Check for lowercase 'action:'
         if action_start_index == -1:
            if logger: logger.warning("No 'Action:' or 'action:' found in LLM output at all.")
            # Attempt to parse any single action-like string directly if no prefix - less reliable
            # This is a fallback for very unconventional outputs.
            single_action_match = re.match(r"^\s*(\w+)\(.*?\)\s*$", text, re.DOTALL)
            if single_action_match:
                 action_str_potential = single_action_match.group(0)
                 if logger: logger.debug(f"Attempting to parse as single action without prefix: {action_str_potential}")
                 # Pre-process special box tokens before parsing single action
                 processed_action_str = re.sub(r'<\|box_start\|>\s*\((\d+),\s*(\d+)\)\s*<\|box_end\|>', r'(\1,\2)', action_str_potential)
                 processed_action_str = re.sub(r'<\|box_start\|>\s*\((.+?)\)\s*<\|box_end\|>', r'(\1)', processed_action_str)
                 parsed_action = parse_single_action(processed_action_str, logger)
                 if parsed_action:
                     return [parsed_action], thought # Return list with one action and the extracted thought
            return [], thought # Return empty list if no actions found, but return the extracted thought


    # Extract the part of the string that contains actions
    action_str_part = text[action_start_index + len("Action:"):].strip() # Assumes "Action:" is 7 chars

    # 4. Pre-process the action string part: Replace <|box_start|>(x,y)<|box_end|> with just (x,y)
    processed_action_str_part = re.sub(r'<\|box_start\|>\s*\((\d+),\s*(\d+)\)\s*<\|box_end\|>', r'(\1,\2)', action_str_part)
    # Also handle potential variations where coordinates might be inside different delimiters or just text
    processed_action_str_part = re.sub(r'<\|box_start\|>\s*\((.+?)\)\s*<\|box_end\|>', r'(\1)', processed_action_str_part)

    if logger: logger.debug(f"Processed action part (special tokens removed): {processed_action_str_part}")


    # 5. Split actions by newline characters that are NOT inside parentheses, brackets, or quotes
    # Use a state-based split for robustness over complex regex.
    raw_action_strings: List[str] = []
    current_action_string = ""
    in_quotes = False
    in_parens = 0
    in_brackets = 0
    quote_char = None # Track which quote char (' or ") started the quote

    for i, char in enumerate(processed_action_str_part):
        is_escaped = (i > 0 and processed_action_str_part[i-1] == '\\')

        if char in ("'", '"') and not is_escaped:
            if not in_quotes:
                in_quotes = True
                quote_char = char # Remember which quote char started it
            elif char == quote_char: # Only close if it's the matching quote
                in_quotes = False
                quote_char = None
            current_action_string += char
        elif char == '(' and not in_quotes:
             in_parens += 1
             current_action_string += char
        elif char == ')' and not in_quotes:
             in_parens -= 1
             current_action_string += char
        elif char == '[' and not in_quotes:
             in_brackets += 1
             current_action_string += char
        elif char == ']' and not in_quotes:
             in_brackets -= 1
             current_action_string += char
        elif char == '\n' and not in_quotes and in_parens == 0 and in_brackets == 0:
            # Found a newline outside quotes/parens/brackets - it's an action separator
            if current_action_string.strip(): # Only add if it's not just whitespace
                raw_action_strings.append(current_action_string.strip())
            current_action_string = "" # Reset for the next action
        else:
            current_action_string += char

    # Add the last action string after the loop finishes
    if current_action_string.strip():
        raw_action_strings.append(current_action_string.strip())

    if logger: logger.debug(f"Split into raw action strings: {raw_action_strings}")


    # 6. Parse each individual action string
    parsed_actions = []
    for action_str in raw_action_strings:
        parsed_action = parse_single_action(action_str, logger) # This calls the fixed parse_single_action
        if parsed_action:
            parsed_actions.append(parsed_action)
        elif logger:
             logger.warning(f"Failed to parse single action string: '{action_str}'")

    return parsed_actions, thought # Return the extracted thought here


def parse_single_action(action_str: str, logger: logging.Logger = None) -> Optional[Dict]:
    """
    Helper function to parse a single action string (e.g., 'click(start_box='(123,456)')' or 'type(content="...")')
    after pre-processing special box tokens.

    Args:
        action_str: The action string to parse (should have tokens like <|box_start|> already removed).
                    Expected format: action_type(arg1=val1, arg2=val2, ...) or action_type(val1, val2, ...)
                    Coordinates, after pre-processing, should be in (x,y) string format within quotes or raw.
        logger: Optional logger for debugging.

    Returns:
        Parsed action dictionary or None if parsing fails critically.
        The dictionary includes keys like 'type', 'inputs' (raw parsed inputs), and specific keys
        ('coordinates', 'content', 'key', etc.) mapped for the action executor.
    """
    if logger: logger.debug(f"Parsing single action string: '{action_str}'")
    action_str = action_str.strip() # Ensure no leading/trailing whitespace

    # Regex to extract the action name and the full argument string inside the outermost parentheses
    match = re.match(r"(\w+)\((.*)\)", action_str)
    if not match:
        if logger: logger.warning(f"Failed to match basic action(args) pattern: '{action_str}'")
        return None

    action_type_raw = match.group(1)
    args_str = match.group(2).strip() # Get the string inside parentheses

    action_inputs_parsed: Dict[str, Any] = {} # Store parsed values

    if args_str: # Only process arguments if the string is not empty
        # Split argument string by comma, but only if the comma is NOT inside parentheses, brackets, or quotes.
        # Use a state-based split for robustness.
        param_strings: List[str] = []
        current_param = ""
        in_quotes = False
        in_parens = 0
        in_brackets = 0
        quote_char = None

        for i, char in enumerate(args_str):
            # Check for escaped quotes - look back one character. Be careful at index 0.
            is_escaped = (i > 0 and args_str[i-1] == '\\')

            if char in ("'", '"') and not is_escaped:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char # Remember which quote char started it
                elif char == quote_char: # Only close if it's the matching quote
                    in_quotes = False
                    quote_char = None
                current_param += char
            elif char == '(' and not in_quotes:
                 in_parens += 1
                 current_param += char
            elif char == ')' and not in_quotes:
                 in_parens -= 1
                 current_param += char
            elif char == '[' and not in_quotes:
                 in_brackets += 1
                 current_param += char
            elif char == ']' and not in_quotes:
                 in_brackets -= 1
                 current_param += char
            elif char == ',' and not in_quotes and in_parens == 0 and in_brackets == 0:
                # Found a comma outside quotes/parens/brackets - it's a separator
                if current_param.strip(): # Only add if it's not just whitespace
                    param_strings.append(current_param.strip())
                current_param = "" # Reset for the next parameter
            else:
                current_param += char

        # Add the last parameter after the loop finishes
        if current_param.strip():
            param_strings.append(current_param.strip())

        if logger: logger.debug(f"Split args string into raw parameter strings: {param_strings}")


        # Now parse each parameter string (key=value or value)
        current_pos_arg_index = 0
        for param_string in param_strings:
            key = None
            value_string_raw = param_string # Store the raw parameter string

            # Check if it's a key=value pair
            # Find the first '=' that is NOT inside quotes or parentheses/brackets
            eq_index = -1
            in_quotes_eq = False
            in_parens_eq = 0
            in_brackets_eq = 0
            quote_char_eq = None
            for i, char in enumerate(param_string):
                 is_escaped = (i > 0 and param_string[i-1] == '\\')
                 if char in ("'", '"') and not is_escaped:
                     if not in_quotes_eq:
                         in_quotes_eq = True
                         quote_char_eq = char
                     elif char == quote_char_eq:
                         in_quotes_eq = False
                         quote_char_eq = None
                 elif char == '(' and not in_quotes_eq:
                     in_parens_eq += 1
                 elif char == ')' and not in_quotes_eq:
                     in_parens_eq -= 1
                 elif char == '[' and not in_quotes_eq:
                     in_brackets_eq += 1
                 elif char == ']' and not in_quotes_eq:
                     in_brackets_eq -= 1
                 elif char == '=' and not in_quotes_eq and in_parens_eq == 0 and in_brackets_eq == 0:
                      eq_index = i
                      break

            if eq_index != -1:
                 key = param_string[:eq_index].strip()
                 value_string = param_string[eq_index + 1:].strip()
                 if not key: # '=' was at the start or preceded only by whitespace
                      if logger: logger.warning(f"Found '=' but no key in parameter string: '{param_string}'")
                      key = None # Not a valid key=value pair
            elif logger:
                 logger.debug(f"No '=' found in parameter string: '{param_string}' - treating as positional.")
                 value_string = param_string.strip() # Use the whole string as the value if no key


            # --- Strip outer quotes from the value_string if it's a quoted literal ---
            # This happens BEFORE calling parse_value_string
            value_string_stripped = value_string
            if ((value_string_stripped.startswith("'") and value_string_stripped.endswith("'")) or
                (value_string_stripped.startswith('"') and value_string_stripped.endswith('"'))) and \
               len(value_string_stripped) >= 2: # Ensure it's not just a single quote
                # Check if the quotes are balanced and the first/last non-whitespace chars are quotes
                stripped_check = value_string_stripped.strip()
                if ((stripped_check.startswith("'") and stripped_check.endswith("'")) or
                    (stripped_check.startswith('"') and stripped_check.endswith('"'))) and \
                    len(stripped_check) >= 2:
                     value_string_stripped = value_string_stripped[1:-1] # Strip the outer quotes


            # --- Parse the (potentially stripped) value string using the helper ---
            parsed_value = parse_value_string(value_string_stripped, logger)

            # Assign the parsed value to action_inputs_parsed
            if key:
                action_inputs_parsed[key] = parsed_value
                if logger: logger.debug(f"Parsed parameter '{key}': {parsed_value} (from string '{value_string}')")
            elif value_string_raw.strip(): # Use raw string to check if *something* was there positionally
                # It's a positional argument - try to map it (heuristic)
                param_name = None
                # Common positional arguments (map based on action type and index)
                # NOTE: If VLM uses both positional and keyword, this needs more complex handling.
                # Assuming consistent style for a given action type.
                if action_type_raw.lower() in ["type"] and current_pos_arg_index == 0: param_name = "content"
                elif action_type_raw.lower() in ["hotkey"] and current_pos_arg_index == 0: param_name = "key"
                # For coordinate actions, the first arg is often the coordinate
                # Assuming raw (x,y) string was already converted to [x,y] by parse_value_string IF it matched.
                # So check if the parsed_value is a list [x,y].
                elif action_type_raw.lower() in ["click", "left_double", "right_single", "scroll", "hover", "drag"] and current_pos_arg_index == 0 and isinstance(parsed_value, list) and len(parsed_value) == 2:
                    param_name = "start_box" # Map first coordinate-like arg

                # For drag, the second arg is often the end coordinate
                elif action_type_raw.lower() in ["drag"] and current_pos_arg_index == 1 and isinstance(parsed_value, list) and len(parsed_value) == 2:
                    param_name = "end_box" # Map second coordinate-like arg

                # For scroll, the second arg is often the direction string
                elif action_type_raw.lower() in ["scroll"] and current_pos_arg_index == 1 and isinstance(parsed_value, str):
                     param_name = "direction" # Map second arg for scroll

                if param_name:
                    action_inputs_parsed[param_name] = parsed_value # Map the positional value
                    current_pos_arg_index += 1
                    if logger: logger.debug(f"Parsed positional parameter at index {current_pos_arg_index-1} mapped to '{param_name}': {parsed_value} (from string '{value_string}')")
                else:
                    if logger: logger.warning(f"Ignoring unassigned positional argument '{param_string}' (parsed as {parsed_value}) for action '{action_type_raw}' at index {current_pos_arg_index}.")
                    current_pos_arg_index += 1 # Still increment to track position


    # Convert action type to lowercase for consistent matching
    action_type = action_type_raw.lower()

    parsed_action_dict: Dict[str, Any] = {
        'type': action_type, # Store lowercase type
        'inputs_raw_string': action_str, # Store the original action string for debugging
        'inputs': action_inputs_parsed, # Store the dictionary of all parsed inputs
        # Initialize keys expected by execute_action to None
        'coordinates': None,
        'content': None,
        'key': None,
        'direction': None,
        'start_coordinates': None,
        'end_coordinates': None,
    }

    # Map specific parsed parameters to the output dictionary structure
    # Coordinates mapping: Take the [x, y] list stored under 'start_box' or 'end_box' keys
    if 'start_box' in action_inputs_parsed:
         coords = action_inputs_parsed['start_box']
         if isinstance(coords, list) and len(coords) == 2:
              parsed_action_dict['start_coordinates'] = coords
              # For actions expecting a single coordinate, use the start_box point
              # Note: This mapping might be redundant if execute_action always uses start_coordinates for point actions.
              # Check execute_action implementation. It seems to use 'coordinates' key primarily for single points.
              if action_type in ["click", "left_double", "right_single", "scroll", "hover"]:
                   parsed_action_dict['coordinates'] = coords # Assign [x, y] list
         elif coords is not None and logger: # Log if start_box was found but wasn't a valid [x,y] after parsing
              logger.warning(f"Parsed 'start_box' value is not a valid [x, y] list for action '{action_type}': {coords} (type: {type(coords).__name__}). Action may fail.") # Reduced severity from critical failure

    if 'end_box' in action_inputs_parsed:
         coords = action_inputs_parsed['end_box']
         if isinstance(coords, list) and len(coords) == 2:
              parsed_action_dict['end_coordinates'] = coords # Assign [x, y] list
         elif coords is not None and logger: # Log if end_box was found but wasn't a valid [x,y]
              logger.warning(f"Parsed 'end_box' value is not a valid [x, y] list for action '{action_type}': {coords} (type: {type(coords).__name__}). Action may fail.")


    # Map other parameters
    if 'content' in action_inputs_parsed:
         parsed_action_dict['content'] = action_inputs_parsed['content']

    if 'key' in action_inputs_parsed:
         parsed_action_dict['key'] = action_inputs_parsed['key']

    if 'direction' in action_inputs_parsed:
         # Ensure direction is a string and lowercase it
         dir_val = action_inputs_parsed['direction']
         if isinstance(dir_val, str):
             parsed_action_dict['direction'] = dir_val.lower() # Store lowercase direction
         elif dir_val is not None and logger:
             logger.warning(f"Parsed 'direction' value is not a string for action '{action_type}': {dir_val} (type: {type(dir_val).__name__}). Action may fail.")


    # Handle special action types (case-insensitive matching)
    if action_type == FINISH_WORD.lower():
         parsed_action_dict['type'] = FINISH_WORD # Store as the standard constant
    elif action_type == WAIT_WORD.lower():
         parsed_action_dict['type'] = WAIT_WORD # Store as the standard constant
    elif action_type == CALL_USER.lower():
         parsed_action_dict['type'] = CALL_USER # Store as the standard constant
    # No need for else, the type is already set to the lowercase raw type


    if logger: logger.debug(f"Final parsed action dictionary before validation: {parsed_action_dict}")

    # --- Final validation based on action type requirements ---
    # If any critical requirement is missing *after* parsing and mapping, return None
    # Note: The checks below now rely on the *mapped* values in parsed_action_dict

    if parsed_action_dict['type'] in ["click", "left_double", "right_single", "hover"]: # Actions needing a single point
         # These actions require 'coordinates' (which should be mapped from start_box)
         if parsed_action_dict['coordinates'] is None:
              if logger: logger.warning(f"Validation failed: Action '{parsed_action_dict['type']}' requires a single coordinate (from start_box).")
              return None # Treat as unparseable if critical info is missing

    if parsed_action_dict['type'] == "scroll": # Scroll needs direction, coordinates are optional for executor
         # Scroll requires 'direction'
         if parsed_action_dict['direction'] is None:
             if logger: logger.warning(f"Validation failed: Action 'scroll' requires a direction ('up', 'down', 'left', or 'right').")
             return None # Scroll without direction is invalid

    if parsed_action_dict['type'] == "drag": # Drag needs start and end coordinates
         # Drag requires 'start_coordinates' and 'end_coordinates' (mapped from start_box and end_box)
         if parsed_action_dict['start_coordinates'] is None or parsed_action_dict['end_coordinates'] is None:
              if logger: logger.warning(f"Validation failed: Action 'drag' requires valid start and end coordinates (from start_box and end_box).")
              return None # Treat as unparseable if critical info is missing

    if parsed_action_dict['type'] == "type": # Type needs content (can be None or empty string, execute_action handles it)
         # No strict validation failure here for missing content, execute_action can handle type().
         pass

    if parsed_action_dict['type'] == "hotkey": # Hotkey needs a key
         # Hotkey requires 'key'
         if parsed_action_dict['key'] is None or parsed_action_dict['key'] == '':
              if logger: logger.warning(f"Validation failed: Action 'hotkey' requires a key.")
              return None # Hotkey without a key is invalid

    # Actions like 'wait', 'finished', 'call_user' don't require specific parameters to be valid.

    if logger: logger.debug(f"Action '{parsed_action_dict['type']}' passed validation.")
    return parsed_action_dict

# Kept for potential legacy use or fallback, but the main logic now uses parse_llm_output
# It is unlikely this function will be called by the client anymore.
def extract_coordinates_from_action(action_str: str, logger: logging.Logger = None) -> Optional[List[int]]:
     """
     (Legacy) Extract *any* (x,y) coordinate tuple from a string. Less specific.
     """
     try:
         if logger:
             logger.info(f"(Legacy) Attempting to extract coordinates from: {action_str}")

         # Look for (x,y) pattern anywhere in the string
         # This legacy function doesn't handle the special tokens, relies on raw string.
         coords_match = re.search(r"\((\d+),\s*(\d+)\)", action_str)
         if coords_match:
             x, y = int(coords_match.group(1)), int(coords_match.group(2))
             if logger:
                 logger.info(f"(Legacy) Found coordinates: [{x}, {y}]")
             return [x, y]

         if logger:
             logger.warning(f"(Legacy) No (x,y) coordinates pattern found in: {action_str}")
         return None
     except Exception as e:
         if logger:
             logger.error(f"(Legacy) Failed to extract coordinates: {e}")
         return None