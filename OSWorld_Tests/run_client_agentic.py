# agentictest/run_client_agentic.py
"""
Script to run end-to-end evaluation on the benchmark using ClientAgenticArchitecture.
Based on typical DesktopEnv framework runner.
"""
import argparse
import datetime
import json
import logging
import os
import sys
from dotenv import load_dotenv 
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), 'DesktopEnvFramework'))

try:
    import lib_run_single 
except ImportError:
    print("ERROR: lib_run_single.py not found. Ensure it's in the PYTHONPATH or accessible.")
    print("You might need to copy it from DesktopEnvFramework/ or adjust sys.path.")
    sys.exit(1)

try:
    from desktop_env.desktop_env import DesktopEnv
except ImportError:
    print("ERROR: desktop_env package not found. Ensure DesktopEnvFramework is in PYTHONPATH.")
    sys.exit(1)


# --- Import your ClientAgenticAdapter ---
try:
    from ClientAgenticArchitectureAdapter.adapter import ClientAgenticAdapter
except ImportError as e:
    print(f"ERROR: Could not import ClientAgenticAdapter: {e}")
    print("Ensure ClientAgenticArchitectureAdapter directory is in 'agentictest/' and contains adapter.py,")
    print("and ClientAgenticArchitecture is also in 'agentictest/' and its internal imports are correct.")
    sys.exit(1)

# --- Load .env variables ---
if load_dotenv():
    logging.info("Loaded environment variables from .env file.")
else:
    logging.warning(".env file not found or python-dotenv not installed. Relying on externally set environment variables.")
# --- End of .env loading ---


# --- Logger Configs (Standard from DesktopEnv framework) ---
logger = logging.getLogger() # Get root logger
logger.setLevel(logging.DEBUG) # Set root logger level

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
log_dir = os.path.join(os.path.dirname(__file__), "logs_client_agentic") # Store logs in a specific folder
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join(log_dir, "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join(log_dir, "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler( # Specific debug for desktopenv internals
    os.path.join(log_dir, "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG) # Capture detailed logs from environment/agent

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
for handler in [file_handler, debug_handler, stdout_handler, sdebug_handler]:
    handler.setFormatter(formatter)
    logger.addHandler(handler) # Add all handlers to the root logger



logger_experiment = logging.getLogger("desktopenv.experiment") # Specific logger for experiment messages
# --- End of Logger Configs ---


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation using ClientAgenticAdapter"
    )

    # --- Environment Config ---
    parser.add_argument("--path_to_vm", type=str, required=True, help="Path to .vmx or VM identifier")
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument("--action_space", type=str, default="pyautogui", choices=["pyautogui"], help="Action type for DesktopEnv (must be pyautogui for this adapter)")
    parser.add_argument("--observation_type", choices=["screenshot", "a11y_tree", "screenshot_a11y_tree"], default="screenshot_a11y_tree", help="Observation type for DesktopEnv")
    parser.add_argument("--screen_width", type=int, default=1920, help="VM screen width")
    parser.add_argument("--screen_height", type=int, default=1080, help="VM screen height")
    parser.add_argument("--sleep_after_execution", type=float, default=2.0, help="Sleep duration after each action execution in DesktopEnv")
    parser.add_argument("--max_steps", type=int, default=15, help="Max environment steps per task") # Aligned with Coordinator's DEFAULT_MAX_OVERALL_ACTIONS_PERFORMED

    # --- Agent Config (passed to Adapter) ---
    parser.add_argument("--adapter_model_config_name", type=str, default="client_agentic_default", help="Informational name for this adapter's configuration (not directly used to select OpenAI/UI-TARS models here)")
    parser.add_argument("--adapter_max_tokens", type=int, default=10024, help="Max tokens for VLM calls (if adapter uses this)")
    parser.add_argument("--adapter_temperature", type=float, default=1.0, help="Temperature for VLM calls (if adapter uses this)")
    parser.add_argument("--adapter_top_p", type=float, default=0.9, help="Top_p for VLM calls (if adapter uses this)")
    parser.add_argument("--adapter_max_trajectory_length", type=int, default=5, help="Max history turns for adapter (if it manages its own trajectory differently from coordinator)")


    # --- Task/Example Config ---
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples", help="Base directory for task configuration JSONs")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_all.json", help="Path to JSON file listing all test examples")
    parser.add_argument("--domain", type=str, default="all", help="Specify domain to test (e.g., 'application'), or 'all'")
    parser.add_argument("--instruction", type=str, default=None, help="Run a single custom instruction instead of tasks from meta file.")


    # --- Logging & Results ---
    parser.add_argument("--result_dir", type=str, default="./client_agentic_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    if args.action_space != "pyautogui":
        logger_experiment.warning(f"Action space is set to '{args.action_space}', but ClientAgenticAdapter generates 'pyautogui' code. Forcing action_space to 'pyautogui' for DesktopEnv.")
        args.action_space = "pyautogui"
        
    return args


def test(args: argparse.Namespace, test_file_list: dict) -> None:
    scores = []
    max_env_steps = args.max_steps # DesktopEnv steps

    logger_experiment.info(f"Starting test with ClientAgenticAdapter. Args: {args}")

    # --- Instantiate your ClientAgenticAdapter ---

    try:
        agent = ClientAgenticAdapter(
            model_name_from_config=args.adapter_model_config_name, 
            max_tokens_from_config=args.adapter_max_tokens,
            observation_type_from_config=args.observation_type,
            action_space_from_config=args.action_space, # Should be "pyautogui"
            max_trajectory_length_from_config=args.adapter_max_trajectory_length,

        )
    except Exception as e:
        logger_experiment.critical(f"Failed to initialize ClientAgenticAdapter: {e}", exc_info=True)
        return # Cannot proceed


    # --- Initialize DesktopEnv ---
    try:
        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=args.action_space, # This will be "pyautogui"
            screen_size=(args.screen_width, args.screen_height),
            headless=args.headless,
            os_type="Ubuntu", # Assuming Ubuntu, make this an arg if needed
            require_a11y_tree=(args.observation_type in ["a11y_tree", "screenshot_a11y_tree"]),
        )
    except Exception as e:
        logger_experiment.critical(f"Failed to initialize DesktopEnv: {e}", exc_info=True)
        return # Cannot proceed

    # --- Task Iteration ---
    if args.instruction: # Run single custom instruction
        logger_experiment.info(f"Running single custom instruction: {args.instruction}")
        example_id_custom = f"custom_task_{datetime_str}"
        example_result_dir = os.path.join(
            args.result_dir, args.action_space, args.observation_type,
            args.adapter_model_config_name, "custom", example_id_custom # Store under "custom" domain
        )
        os.makedirs(example_result_dir, exist_ok=True)
        # Create a dummy example dict for lib_run_single
        dummy_example_config = {
            "id": example_id_custom,
            "instruction": args.instruction,
            "config": [], # No setup steps for custom instruction via this route
            "evaluator": {"func": "infeasible"} # Placeholder, actual eval won't run well without ground truth
        }
        try:
            lib_run_single.run_single_example(
                agent, env, dummy_example_config, max_env_steps, args.instruction,
                args, example_result_dir, scores
            )
        except Exception as e:
            logger_experiment.error(f"Exception during custom task '{args.instruction}': {e}", exc_info=True)
            # Ensure recording stops and basic error is logged if run_single_example crashes hard
            try:
                env.controller.end_recording(os.path.join(example_result_dir, "recording_error.mp4"))
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f_err:
                    f_err.write(json.dumps({"CRITICAL_ERROR": str(e)}) + "\n")
            except Exception as e_ctrl:
                logger_experiment.error(f"Further error during error handling: {e_ctrl}")


    else: # Run tasks from meta file
        for domain in tqdm(test_file_list, desc="Domain", unit="domain"):
            if not test_file_list[domain]:
                logger_experiment.info(f"Skipping domain '{domain}' as it has no tasks left to run.")
                continue
            for example_id in tqdm(test_file_list[domain], desc=f"Example in {domain}", unit="task", leave=False):
                config_file = os.path.join(
                    args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
                )
                if not os.path.exists(config_file):
                    logger_experiment.error(f"Config file not found: {config_file}. Skipping example {example_id}.")
                    continue
                with open(config_file, "r", encoding="utf-8") as f:
                    example_config = json.load(f)

                logger_experiment.info(f"[Domain]: {domain}, [Example ID]: {example_id}")
                instruction = example_config["instruction"]
                logger_experiment.info(f"[Instruction]: {instruction}")

                example_result_dir = os.path.join(
                    args.result_dir, args.action_space, args.observation_type,
                    args.adapter_model_config_name, domain, example_id
                )
                os.makedirs(example_result_dir, exist_ok=True)

                try:
                    lib_run_single.run_single_example(
                        agent, env, example_config, max_env_steps, instruction,
                        args, example_result_dir, scores
                    )
                except Exception as e: # Catch broad exceptions from run_single_example
                    logger_experiment.error(f"Exception in {domain}/{example_id}: {e}", exc_info=True)
                    # Ensure recording stops and basic error is logged if run_single_example crashes hard
                    try:
                        env.controller.end_recording(os.path.join(example_result_dir, "recording_error.mp4"))
                        with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f_err:
                             f_err.write(json.dumps({"CRITICAL_ERROR": str(e)}) + "\n")
                    except Exception as e_ctrl:
                        logger_experiment.error(f"Further error during {domain}/{example_id} error handling: {e_ctrl}")


    env.close()
    if scores: # Only if tasks from meta file were run
        avg_score = sum(scores) / len(scores) if scores else 0
        logger_experiment.info(f"All tasks from meta file completed. Average score: {avg_score:.4f}")
        # Save overall results
        overall_summary_file = os.path.join(args.result_dir, args.action_space, args.observation_type, args.adapter_model_config_name, "overall_summary.json")
        with open(overall_summary_file, "w", encoding="utf-8") as f:
            json.dump({"average_score": avg_score, "num_tasks": len(scores), "args": vars(args)}, f, indent=4)
    elif args.instruction:
        logger_experiment.info(f"Custom instruction task '{args.instruction}' completed.")
    else:
        logger_experiment.info("No tasks were run from meta file (or scores list is empty).")


# --- Helper functions get_unfinished, get_result (Standard from DesktopEnv framework) ---
def get_unfinished(action_space_arg, model_config_name_arg, observation_type_arg, result_dir_arg, total_tasks_meta):
    target_dir = os.path.join(result_dir_arg, action_space_arg, observation_type_arg, model_config_name_arg)
    if not os.path.exists(target_dir):
        logger_experiment.info(f"Result directory not found: {target_dir}. Running all tasks.")
        return total_tasks_meta

    finished_tasks_by_domain = {}
    for domain_name in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain_name)
        if os.path.isdir(domain_path):
            finished_tasks_by_domain[domain_name] = []
            for task_id in os.listdir(domain_path):
                task_path = os.path.join(domain_path, task_id)
                if os.path.isdir(task_path) and "result.txt" in os.listdir(task_path):
                    finished_tasks_by_domain[domain_name].append(task_id)
    
    unfinished_tasks = {}
    for domain_name, all_task_ids_in_domain in total_tasks_meta.items():
        if domain_name not in finished_tasks_by_domain:
            unfinished_tasks[domain_name] = all_task_ids_in_domain
        else:
            finished_in_this_domain = set(finished_tasks_by_domain[domain_name])
            unfinished_tasks[domain_name] = [tid for tid in all_task_ids_in_domain if tid not in finished_in_this_domain]
    return unfinished_tasks

def get_result_summary(action_space_arg, model_config_name_arg, observation_type_arg, result_dir_arg):
    target_dir = os.path.join(result_dir_arg, action_space_arg, observation_type_arg, model_config_name_arg)
    if not os.path.exists(target_dir):
        logger_experiment.info("No results yet for this configuration.")
        return None

    all_results_scores = []
    for domain_name in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain_name)
        if os.path.isdir(domain_path):
            for task_id in os.listdir(domain_path):
                task_path = os.path.join(domain_path, task_id)
                result_file_path = os.path.join(task_path, "result.txt")
                if os.path.isfile(result_file_path):
                    try:
                        with open(result_file_path, "r") as f_res:
                            all_results_scores.append(float(f_res.read().strip()))
                    except ValueError:
                        logger_experiment.warning(f"Could not parse result.txt for {domain_name}/{task_id}")
                        all_results_scores.append(0.0) # Count as failure if unparseable
    
    if not all_results_scores:
        logger_experiment.info("Found result directory, but no 'result.txt' files processed.")
        return {"count": 0, "average": 0.0}
    else:
        avg = sum(all_results_scores) / len(all_results_scores)
        logger_experiment.info(f"Current aggregated results: Count={len(all_results_scores)}, Average Score={avg:.4f}")
        return {"count": len(all_results_scores), "average": avg}

# --- Main Execution Block ---
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # Common setting for HuggingFace tokenizers
    args = config() # Parse arguments

    # Setup main result directory structure
    specific_config_result_dir = os.path.join(args.result_dir, args.action_space, args.observation_type, args.adapter_model_config_name)
    os.makedirs(specific_config_result_dir, exist_ok=True) # Ensure the specific config path exists for summary later

    if args.instruction:
        test_file_list_final = {} # No tasks from file if single instruction is given
        logger_experiment.info("Running in single instruction mode.")
    else:
        if not os.path.exists(args.test_all_meta_path):
            logger_experiment.critical(f"Test meta file not found: {args.test_all_meta_path}. Exiting.")
            sys.exit(1)
        with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
            all_tasks_from_meta = json.load(f)

        if args.domain != "all":
            if args.domain not in all_tasks_from_meta:
                logger_experiment.critical(f"Specified domain '{args.domain}' not found in test meta file. Available: {list(all_tasks_from_meta.keys())}")
                sys.exit(1)
            all_tasks_from_meta = {args.domain: all_tasks_from_meta[args.domain]}
        
        logger_experiment.info("Checking for unfinished tasks...")
        test_file_list_final = get_unfinished(
            args.action_space, args.adapter_model_config_name, args.observation_type,
            args.result_dir, all_tasks_from_meta
        )
        
        num_unfinished = sum(len(tasks) for tasks in test_file_list_final.values())
        if num_unfinished == 0:
            logger_experiment.info("All tasks for the specified configuration are already completed.")
        else:
            logger_experiment.info(f"Found {num_unfinished} unfinished tasks to run.")
            for domain_name, tasks_in_domain in test_file_list_final.items():
                if tasks_in_domain: logger_experiment.info(f"  Domain '{domain_name}': {len(tasks_in_domain)} tasks remaining.")
        
        get_result_summary(
            args.action_space, args.adapter_model_config_name, args.observation_type, args.result_dir
        )

    # Run the tests
    if args.instruction or any(test_file_list_final.values()): # Only run test if there's something to do
        test(args, test_file_list_final)
        logger_experiment.info("Test execution finished.")
        if not args.instruction: # Recalculate summary if ran tasks from meta
            get_result_summary(
                args.action_space, args.adapter_model_config_name, args.observation_type, args.result_dir
            )
    else:
        logger_experiment.info("No tasks to run (either single instruction not provided or all meta tasks completed).")