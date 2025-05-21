import ast
import base64
import logging
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
import time

from PIL import Image
import openai
from openai import OpenAI

logger = logging.getLogger("desktopenv.agent.uitars_agent")

# --- PROMPT and ACTION SPACE Constants (WORKER_ACTION_SPACE, WORKER_USER_PROMPT_TEMPLATE) ---
# --- Other Constants (FINISH_WORD, etc.) ---
# --- Parser Functions (parse_value_string_v2, parse_single_action_v2, parse_llm_output_v2) ---
# --- Utility Functions (escape_for_pyautogui_write, parsing_response_to_pyautogui_code_v2, pil_to_base64) ---
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
FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

def parse_value_string_v2(value_string: str, log: Optional[logging.Logger] = None) -> Any:
    current_logger = log or logger; original_value_string = value_string; value_string = value_string.strip()
    current_logger.debug(f"Parsing value string: '{original_value_string}'")
    if not ((value_string.startswith("'") and value_string.endswith("'")) or (value_string.startswith('"') and value_string.endswith('"'))):
        coord_match = re.match(r'^\((\s*\d+\s*),(\s*\d+\s*)\)$', value_string)
        if coord_match:
            try: x, y = int(coord_match.group(1).strip()), int(coord_match.group(2).strip()); return [x, y]
            except (ValueError, TypeError): current_logger.debug(f"Direct coordinate parse failed for '{value_string}'. Proceeding.")
    if (value_string.startswith("'") and value_string.endswith("'") and len(value_string) >= 2) or \
       (value_string.startswith('"') and value_string.endswith('"') and len(value_string) >= 2):
        try:
            if re.match(r"""^(['"]).*\1$""", value_string, re.DOTALL):
                val_ast = ast.literal_eval(value_string)
                if isinstance(val_ast, str):
                    coord_match_ast = re.match(r'^\((\s*\d+\s*),(\s*\d+\s*)\)$', val_ast)
                    if coord_match_ast:
                        try: x, y = int(coord_match_ast.group(1).strip()), int(coord_match_ast.group(2).strip()); return [x, y]
                        except (ValueError, TypeError): pass # Fall through to return string
                    return val_ast
                return val_ast
            return value_string[1:-1] if len(value_string) >=2 else ""
        except (ValueError, SyntaxError): return value_string[1:-1] if len(value_string) >=2 else ""
    if value_string.lower() == 'true': return True
    if value_string.lower() == 'false': return False
    if value_string.lower() == 'none': return None
    try: return int(value_string)
    except ValueError: pass
    try: return float(value_string)
    except ValueError: pass
    return value_string

def parse_single_action_v2(action_str: str, log: Optional[logging.Logger] = None) -> Optional[Dict]:
    current_logger = log or logger; action_str = action_str.strip()
    match = re.match(r"(\w+)\s*\((.*)\)\s*$", action_str, re.DOTALL)
    if not match:
        simple_match = re.match(r"^\s*(\w+)\s*$", action_str)
        if simple_match:
            act_type = simple_match.group(1).lower()
            if act_type not in [FINISH_WORD, WAIT_WORD, CALL_USER, ENV_FAIL_WORD] and \
               act_type not in ["click", "type", "hotkey", "scroll", "drag", "select", "hover", "left_single", "left_double", "right_single"]: # Known actions that take args
                current_logger.warning(f"Unrecognized simple action: '{act_type}'"); return None
            return {'type': act_type, 'inputs_raw_string': action_str, 'inputs': {}, 'coordinates': None, 'content': None, 'key': None, 'direction': None, 'start_coordinates': None, 'end_coordinates': None, 'reflection': None, 'thought': None, 'action_type': act_type, 'action_inputs': {}}
        current_logger.warning(f"No action pattern match: '{action_str}'"); return None
    act_type_raw, args_str = match.group(1), match.group(2).strip(); inputs_parsed = {}
    if args_str:
        params = re.findall(r"""(?:[^,"']|"[^"]*"|'[^']*'|\((?:[^()']|'[^']*'|\((?:[^()']|'[^']*')*\))*\))+""", args_str)
        if not params and args_str: params = [s.strip() for s in args_str.split(',')]
        pos_idx = 0
        for p_str in params:
            p_str = p_str.strip();
            if not p_str: continue
            key = None; val_str = p_str
            eq_m = re.match(r"(\w+)\s*=\s*(.*)", p_str, re.DOTALL)
            if eq_m: key, val_str = eq_m.group(1).strip(), eq_m.group(2).strip()
            parsed_val = parse_value_string_v2(val_str, current_logger)
            if key: inputs_parsed[key] = parsed_val
            else:
                p_map = {"type": {0:"content"}, "hotkey":{0:"key"}, "click":{0:"start_box"}, "left_single":{0:"start_box"}, "left_double":{0:"start_box"}, "right_single":{0:"start_box"}, "hover":{0:"start_box"}, "drag":{0:"start_box",1:"end_box"}, "select":{0:"start_box",1:"end_box"}, "scroll":{0:"start_box_or_direction",1:"direction_if_start_box_first"}}
                p_name = f"pos_{pos_idx}"; act_low = act_type_raw.lower()
                if act_low in p_map:
                    if act_low=="scroll": p_name="start_box" if isinstance(parsed_val,list) else "direction" if pos_idx==0 else "direction" if pos_idx==1 and ("start_box" in inputs_parsed or (inputs_parsed.get("start_box_or_direction") and isinstance(inputs_parsed.get("start_box_or_direction"),list))) else p_name
                    elif pos_idx in p_map[act_low]: p_name=p_map[act_low][pos_idx]
                inputs_parsed[p_name]=parsed_val; pos_idx+=1
    act_type = act_type_raw.lower()
    if act_type == FINISH_WORD.lower(): act_type = FINISH_WORD
    elif act_type == WAIT_WORD.lower(): act_type = WAIT_WORD
    elif act_type == CALL_USER.lower(): act_type = CALL_USER
    elif act_type == ENV_FAIL_WORD.lower(): act_type = ENV_FAIL_WORD

    res={'type':act_type, 'inputs_raw_string':action_str, 'inputs':inputs_parsed, 'action_type':act_type, 'action_inputs':inputs_parsed}
    s_box = inputs_parsed.get("start_box") or inputs_parsed.get("start_coordinates") or (inputs_parsed.get("start_box_or_direction") if isinstance(inputs_parsed.get("start_box_or_direction"), list) else None)
    if isinstance(s_box,list) and len(s_box)==2: res['start_coordinates']=s_box; res['coordinates']=s_box if act_type in ["click","left_single","left_double","right_single","scroll","hover"] else None
    e_box = inputs_parsed.get("end_box") or inputs_parsed.get("end_coordinates")
    if isinstance(e_box,list) and len(e_box)==2: res['end_coordinates']=e_box
    res['content']=inputs_parsed.get('content'); res['key']=inputs_parsed.get('key')
    direction = inputs_parsed.get("direction") or (inputs_parsed.get("start_box_or_direction") if not res.get('coordinates') and isinstance(inputs_parsed.get("start_box_or_direction"),str) else None) or (inputs_parsed.get("direction_if_start_box_first") if res.get('coordinates') else None)
    if isinstance(direction,str): res['direction']=direction.lower()
    if act_type in ["click","left_single","left_double","right_single","hover"] and (not isinstance(res.get('coordinates'),list) or len(res['coordinates'])!=2): return None
    if act_type=="scroll" and not (isinstance(res.get('direction'),str) and res['direction'] in ['up','down','left','right']): return None
    if act_type=="type" and res.get('content') is None: res['content'] = ""
    if act_type=="hotkey" and (not isinstance(res.get('key'),str) or not res['key'].strip()): return None
    if act_type=="drag" and not (isinstance(res.get('start_coordinates'),list) and len(res['start_coordinates'])==2 and isinstance(res.get('end_coordinates'),list) and len(res['end_coordinates'])==2): return None
    return res

def parse_llm_output_v2(text: str, log: Optional[logging.Logger] = None) -> Tuple[List[Dict], Optional[str]]:
    current_logger = log or logger; text = text.strip(); thought = None; actions = []
    action_part = text
    m = re.search(r"Thought:\s*(.*?)(?=\s*Action:|$)", text, re.DOTALL | re.IGNORECASE)
    if m: thought, action_part = m.group(1).strip(), text[m.end():].strip()
    if action_part.lower().startswith("action:"): action_part = action_part[len("action:"):]
    elif not thought: 
        m_act = re.search(r"Action:", text, re.IGNORECASE)
        if m_act:
            pot_thought = text[:m_act.start()].strip();
            if pot_thought: thought = pot_thought
            action_part = text[m_act.end():].strip()
    
    proc_action_part = re.sub(r'<\|box_start\|>\s*\((\s*[^)]+\s*)\)\s*<\|box_end\|>', r'(\1)', action_part)
    raw_strs = [s.strip() for s in re.split(r'\n\n|\n', proc_action_part) if s.strip()]
    if not raw_strs and proc_action_part: raw_strs.append(proc_action_part)
    for s_act in raw_strs:
        p_act = parse_single_action_v2(s_act, current_logger)
        if p_act: p_act['thought'] = thought; actions.append(p_act)
    return actions, thought

def escape_for_pyautogui_write(text: Optional[str]) -> str:
    if text is None: return ""
    return str(text).replace("\\", "\\\\").replace("'", "\\'")

def parsing_response_to_pyautogui_code_v2(
    parsed_action: Dict, image_height: int, image_width: int
) -> str:
    py_cmds = []
    action_type = parsed_action.get("type")

    if action_type == "hotkey":
        key_val = parsed_action.get("key")
        if key_val and isinstance(key_val, str):
            keys = [repr(k.strip().lower()) for k in key_val.split('+') if k.strip()]
            if keys: py_cmds.append(f"pyautogui.hotkey({', '.join(keys)})")
        else: py_cmds.append(f"# CodeGenErr: Invalid hotkey: {key_val}")
    elif action_type == "type":
        content_to_type = parsed_action.get("content", "") 
        escaped_content = escape_for_pyautogui_write(content_to_type)
        py_cmds.append(f"pyautogui.write('{escaped_content}', interval=0.02)")
        if isinstance(content_to_type, str) and '\n' in content_to_type: 
            py_cmds.append("pyautogui.press('enter')")
    elif action_type in ["drag", "select"]:
        s_coords_1k = parsed_action.get("start_coordinates"); e_coords_1k = parsed_action.get("end_coordinates")
        if isinstance(s_coords_1k, list) and len(s_coords_1k) == 2 and \
           isinstance(e_coords_1k, list) and len(e_coords_1k) == 2 :
            sx, sy = round(s_coords_1k[0]/1000.0*image_width), round(s_coords_1k[1]/1000.0*image_height)
            ex, ey = round(e_coords_1k[0]/1000.0*image_width), round(e_coords_1k[1]/1000.0*image_height)
            sx,sy = max(0,min(sx,image_width-1)), max(0,min(sy,image_height-1))
            ex,ey = max(0,min(ex,image_width-1)), max(0,min(ey,image_height-1))
            py_cmds.extend([f"pyautogui.moveTo({sx}, {sy}, duration=0.2)",
                            f"pyautogui.dragTo({ex}, {ey}, duration=0.5, button='left')"])
        else: py_cmds.append(f"# CodeGenErr: Drag missing/invalid coords S:{s_coords_1k} E:{e_coords_1k}")
    elif action_type == "scroll":
        direction = parsed_action.get("direction", "").lower(); coords_1k = parsed_action.get("coordinates")
        scroll_args_list = []
        if isinstance(coords_1k, list) and len(coords_1k) == 2:
            x,y = round(coords_1k[0]/1000.0*image_width), round(coords_1k[1]/1000.0*image_height)
            x,y = max(0,min(x,image_width-1)), max(0,min(y,image_height-1))
            py_cmds.append(f"pyautogui.moveTo({x}, {y}, duration=0.2)")
            scroll_args_list.extend([f"x={x}", f"y={y}"])
        scroll_amount = 200 
        scroll_args_str = (", " + ", ".join(scroll_args_list)) if scroll_args_list else ""
        if "up" in direction: py_cmds.append(f"pyautogui.scroll({scroll_amount}{scroll_args_str})")
        elif "down" in direction: py_cmds.append(f"pyautogui.scroll({-scroll_amount}{scroll_args_str})")
        elif "left" in direction: py_cmds.append(f"pyautogui.hscroll({scroll_amount}{scroll_args_str})")
        elif "right" in direction: py_cmds.append(f"pyautogui.hscroll({-scroll_amount}{scroll_args_str})")
        else: py_cmds.append(f"# CodeGenErr: Scroll invalid direction: {direction}")
    elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
        coords_1k = parsed_action.get("coordinates")
        if isinstance(coords_1k, list) and len(coords_1k) == 2:
            x,y = round(coords_1k[0]/1000.0*image_width), round(coords_1k[1]/1000.0*image_height)
            x,y = max(0,min(x,image_width-1)), max(0,min(y,image_height-1))
            py_cmds.append(f"pyautogui.moveTo({x}, {y}, duration=0.2)")
            if action_type == "click" or action_type == "left_single": py_cmds.append(f"pyautogui.click(button='left')")
            elif action_type == "left_double": py_cmds.append(f"pyautogui.doubleClick(button='left')")
            elif action_type == "right_single": py_cmds.append(f"pyautogui.click(button='right')")
        else: py_cmds.append(f"# CodeGenErr: {action_type} missing/invalid coords: {coords_1k}")
    elif action_type == WAIT_WORD: py_cmds.append(f"time.sleep(5) # Action: {WAIT_WORD}")
    elif action_type in [FINISH_WORD, CALL_USER, ENV_FAIL_WORD]: py_cmds.append(f"# Signal: {action_type}")
    else: py_cmds.append(f"# Unrecognized action: {action_type}")
    return "\n".join(py_cmds) if py_cmds else f"# No command for {action_type}"

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
        history_n: int = 5, 
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

        # Agent's main trajectory history lists
        self.thoughts = []
        self.actions = []
        self.observations = [] 
        
        # History for VLM API calls (images as bytes, responses as text)
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
                logger.debug("Screenshot from obs is a string, attempting base64 decode.")
                try:
                    current_screenshot_bytes = base64.b64decode(screenshot_data_from_obs)
                except Exception as decode_e:
                    logger.error(f"Error decoding base64 screenshot string: {decode_e}. String start: {screenshot_data_from_obs[:100]}")
                    return f"ERR_SCREENSHOT_B64_DECODE:{decode_e}", ["# ERR_SCREENSHOT_B64_DECODE"]
            elif isinstance(screenshot_data_from_obs, bytes):
                logger.debug("Screenshot from obs is already bytes. Using directly.")
                current_screenshot_bytes = screenshot_data_from_obs
            else:
                logger.error(f"Screenshot from obs is of unexpected type: {type(screenshot_data_from_obs)}. Expected str or bytes.")
                return "ERR_UNEXPECTED_SCREENSHOT_TYPE", ["# ERR_UNEXPECTED_SCREENSHOT_TYPE"]

            # Client-side verification of the current screenshot
            if current_screenshot_bytes:
                try:
                    img_verify = Image.open(BytesIO(current_screenshot_bytes))
                    img_verify.verify()  # Basic integrity check
                    logger.debug(f"Current screenshot (bytes) verified. Format: {img_verify.format}, Size: {img_verify.size}")
                except Exception as verify_e:
                    logger.error(f"CLIENT-SIDE VERIFICATION FAILED: Current screenshot data is not a valid image. Error: {verify_e}.")
                    log_data = screenshot_data_from_obs[:100] if isinstance(screenshot_data_from_obs, str) else current_screenshot_bytes[:100]
                    logger.error(f"Data that failed verification (first 100 units): {log_data}")
                    return f"ERR_INVALID_SCREENSHOT_CLIENT_SIDE:{verify_e}", ["# ERR_INVALID_SCREENSHOT_CLIENT_SIDE"]
            else: # current_screenshot_bytes became None after processing (e.g. decode error)
                 return "ERR_SCREENSHOT_PROCESSING_FAILED", ["# ERR_SCREENSHOT_PROCESSING_FAILED"]
        
        # --- Manage Persistent History (like working script) ---
        if current_screenshot_bytes: # Only append if current screenshot is valid and available
            self.history_images.append(current_screenshot_bytes)
        elif not self.history_images: # Current is bad/missing AND history is empty
            logger.error("No screenshot available for VLM call (current is missing/invalid and no historical).")
            return "ERR_NO_IMAGES_FOR_VLM", ["# ERR_NO_IMAGES_FOR_VLM"]
        else: # Current is bad/missing, but history exists
            logger.warning("Current screenshot is missing or invalid. Proceeding with historical images.")

        # Trim image history
        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]
        
        # --- Agent's Internal Observation Logging ---
        base64_image_for_agents_log = None
        if current_screenshot_bytes:
            base64_image_for_agents_log = base64.b64encode(current_screenshot_bytes).decode('utf-8')
        elif obs_dict.get("screenshot") and isinstance(obs_dict.get("screenshot"), str):
            base64_image_for_agents_log = obs_dict.get("screenshot")

        self.observations.append({"screenshot": base64_image_for_agents_log, "accessibility_tree": None})
        if len(self.observations) > self.max_trajectory_len_agent > 0:
            self.observations = self.observations[-self.max_trajectory_len_agent:]

        # --- VLM API Message Construction ---

        # If images_for_api has N images, there should be N-1 past responses from the assistant.
        # The latest image in images_for_api does not yet have an assistant response.
        api_images_to_send = list(self.history_images)
        api_responses_to_interleave = list(self.history_responses)

        expected_prev_responses = max(0, len(api_images_to_send) - 1)
        if len(api_responses_to_interleave) > expected_prev_responses:
             api_responses_to_interleave = api_responses_to_interleave[-expected_prev_responses:]
        
        sub_task_instruction_for_vlm = instruction
        full_user_prompt_text = self.prompt_template_str.format(
            action_space=self.prompt_action_space_text, language=self.language,
            sub_task_instruction=sub_task_instruction_for_vlm
        )

        api_messages: List[Dict[str, Any]] = [{"role": "system", "content": [{"type": "text", "text": "You are a precise GUI agent."}]}]
        
        first_user_turn_content = [{"type": "text", "text": full_user_prompt_text}]
        if api_images_to_send: # If there are any images at all
            b64_img = pil_to_base64(Image.open(BytesIO(api_images_to_send[0])))
            first_user_turn_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}})
        api_messages.append({"role": "user", "content": first_user_turn_content})

        # Interleave past assistant responses and subsequent user images
        for i in range(len(api_responses_to_interleave)): # Iterate through historical assistant responses
            api_messages.append({"role": "assistant", "content": [{"type": "text", "text": api_responses_to_interleave[i]}]})
            # The image that followed this assistant response (and is now the (i+1)th image in history)
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
        
        # Update persistent history_responses
        self.history_responses.append(raw_vlm_prediction)
        if len(self.history_responses) > self.history_n : # Trim responses
             self.history_responses = self.history_responses[-self.history_n:]
             
        logger.info(f"VLM Raw Prediction: {raw_vlm_prediction[:200]}...")
        
        try:
            parsed_action_dicts, extracted_thought = parse_llm_output_v2(raw_vlm_prediction, logger)
            self.thoughts.append(extracted_thought or "No thought parsed.")
            if len(self.thoughts) > self.max_trajectory_len_agent > 0:
                self.thoughts = self.thoughts[-self.max_trajectory_len_agent:]
        except Exception as e:
            logger.error(f"Parse Error: {e}. Raw: {raw_vlm_prediction[:200]}", exc_info=True)
            self.thoughts.append(f"Parse Error: {e}"); self.actions.append(["#PARSE_ERROR"])
            return f"ERR_PARSE:{e}", ["#PARSE_ERROR"]

        step_pyautogui_commands = []; return_signal_list = None
        if not parsed_action_dicts:
            logger.warning("No actions parsed. Checking raw for keywords or defaulting to WAIT.")
            if FINISH_WORD in raw_vlm_prediction.lower(): return_signal_list = ["DONE"]
            elif WAIT_WORD in raw_vlm_prediction.lower(): return_signal_list = ["WAIT"]
            elif CALL_USER in raw_vlm_prediction.lower(): return_signal_list = ["FAIL_CALL_USER"] 
            else: return_signal_list = ["WAIT"]; step_pyautogui_commands.append("time.sleep(5) # Default wait")
        
        for pa_dict in parsed_action_dicts:
            action_type = pa_dict.get("type")
            cmd_str = parsing_response_to_pyautogui_code_v2(pa_dict, self.screen_height, self.screen_width)
            if action_type == FINISH_WORD: return_signal_list = ["DONE"]; step_pyautogui_commands.append(cmd_str); break
            if action_type == WAIT_WORD: return_signal_list = ["WAIT"]; step_pyautogui_commands.append(cmd_str); break 
            if action_type == CALL_USER: return_signal_list = ["FAIL_CALL_USER"]; step_pyautogui_commands.append(cmd_str); break 
            if action_type == ENV_FAIL_WORD: return_signal_list = ["FAIL"]; step_pyautogui_commands.append(cmd_str); break
            if cmd_str and not cmd_str.startswith("# Signal:") and not cmd_str.startswith("# No command for"):
                step_pyautogui_commands.append(cmd_str)
            elif cmd_str: logger.info(f"Generated non-command code for {action_type}: {cmd_str}")

        full_script_str = ""
        if step_pyautogui_commands:
            imports = "import pyautogui\nimport time\n"
            thought_comment = f"\n'''\n--- Agent Thought ---\n{self.thoughts[-1] if self.thoughts else 'N/A'}\n'''\n"
            full_script_str = imports + thought_comment + "\n".join(step_pyautogui_commands)
            if any(cmd for cmd in step_pyautogui_commands if not cmd.startswith("#") and "time.sleep(5)" not in cmd):
                full_script_str += "\ntime.sleep(0.5) # UI stabilization"
        elif not return_signal_list: 
            full_script_str = f"# No executable actions. Thought: {self.thoughts[-1] if self.thoughts else 'N/A'}"
            return_signal_list = ["# NO_ACTION_SCRIPT"] 

        self.actions.append(step_pyautogui_commands if step_pyautogui_commands else (return_signal_list or ["#NO_CMDS"]))
        if len(self.actions) > self.max_trajectory_len_agent > 0:
            self.actions = self.actions[-self.max_trajectory_len_agent:]

        # Check against self.history_responses length for max steps, as it's incremented each VLM call
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