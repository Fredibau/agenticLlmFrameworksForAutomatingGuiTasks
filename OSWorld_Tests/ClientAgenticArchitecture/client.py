import logging
import os
from typing import Optional, Dict, Any 
from dotenv import load_dotenv

from .planner import Planner
from .worker import Worker
from .coordinator import Coordinator
from .utils.logging_utils import setup_logger

# --- Application Configuration ---
# Load environment variables from .env file
try:
    if load_dotenv():
        logging.info("Attempted to load environment variables from .env file.")
    else:
        logging.info(".env file not found or empty. Relying on environment variables set externally.")
except ImportError:
    logging.warning("python-dotenv not found. Environment variables should be set manually if .env file usage is intended.")
except Exception as e:
    logging.warning(f"Error loading .env: {e}")


# --- Configuration Values (loaded from .env or using defaults) ---
PLANNER_API_KEY = os.getenv("OPENAI_API_KEY")
PLANNER_MODEL_NAME = os.getenv("PLANNER_MODEL_NAME", "gpt-4o")

UI_TARS_SERVER_URL = os.getenv("UI_TARS_SERVER_URL")
UI_TARS_API_KEY = os.getenv("UI_TARS_API_KEY", "empty")
UI_TARS_MODEL_NAME = os.getenv("UI_TARS_MODEL_NAME", "ui-tars")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Main entry point function for the task automation
def run_automated_task(overall_goal_param: Optional[str] = None) -> Optional[Dict[str, Any]]: 
    """
    Initializes components and runs the automated task.
    Coordinator uses its internal defaults unless MAX_OVERALL_ACTIONS_PERFORMED is set in env.
    Returns a dictionary containing the execution result, or None if setup fails early.
    """
    client_logger = setup_logger("ClientApp", LOG_LEVEL)
    client_logger.info("Client application starting...")
    client_logger.info(f"Log level set to: {LOG_LEVEL}")

    # Check for required environment variables
    if not PLANNER_API_KEY:
        client_logger.critical("Planner API Key (OPENAI_API_KEY) is not set. Exiting.")
        return None 
    if not UI_TARS_SERVER_URL:
        client_logger.critical("UI_TARS_SERVER_URL is not set. Exiting.")
        return None 

    client_logger.debug(f"Planner Model: {PLANNER_MODEL_NAME}")
    client_logger.debug(f"UI_TARS Server URL: {UI_TARS_SERVER_URL}")
    client_logger.debug(f"UI_TARS Model: {UI_TARS_MODEL_NAME}")

    coordinator_kwargs: Dict[str, Any] = {}

    # Initialize the Coordinator with the Planner and Worker components
    try:
        planner_instance = Planner(api_key=PLANNER_API_KEY, model_name=PLANNER_MODEL_NAME, logger=setup_logger("PlannerComponent", LOG_LEVEL))
        client_logger.info("Planner initialized.")
        worker_instance = Worker(ui_tars_server_url=UI_TARS_SERVER_URL, ui_tars_api_key=UI_TARS_API_KEY, ui_tars_model_name=UI_TARS_MODEL_NAME, logger=setup_logger("WorkerComponent", LOG_LEVEL))
        client_logger.info("Worker initialized.")

        coordinator_instance = Coordinator(
            planner=planner_instance,
            worker=worker_instance,
            logger=setup_logger("CoordinatorComponent", LOG_LEVEL),
        )
        client_logger.info("Coordinator initialized.")
        client_logger.debug(f"Coordinator is using max_overall_actions_performed: {coordinator_instance.max_overall_actions_performed}")

    except ImportError as e:
        client_logger.critical(f"Failed to initialize components due to missing library: {e}. Please install all dependencies.")
        return None # Return None on critical setup failure
    except Exception as e:
        client_logger.critical(f"An unexpected error occurred during component initialization: {e}", exc_info=True)
        return None # Return None on critical setup failure

    # Check for the overall goal parameter
    overall_goal: str
    if overall_goal_param:
        overall_goal = overall_goal_param
        client_logger.info(f"Using task provided as parameter: '{overall_goal}'")
    else:
        try:
            overall_goal_input = input("\nPlease describe the task you want to automate: ")
            if not overall_goal_input.strip():
                client_logger.warning("No task description provided. Exiting.")
                return None # Return None if no task given
            overall_goal = overall_goal_input.strip()
        except KeyboardInterrupt:
            client_logger.info("\nTask input cancelled by user. Exiting.")
            return None # Return None if input cancelled
        except EOFError:
            client_logger.error("Failed to get task via input (EOFError). Provide task directly or run interactively.")
            return None # Return None on input error

    # Start task execution
    client_logger.info(f"Starting task execution for: '{overall_goal}'")
    result = None # Initialize result

    try:
        result = coordinator_instance.execute_overall_goal(overall_goal)
    except KeyboardInterrupt:
        client_logger.info("\nTask execution interrupted by user.")
        result = {
            "success": False,
            "reason": "Task execution interrupted by user.",
            "overall_actions_performed_count": len(getattr(coordinator_instance, 'all_executed_actions', []))
        }
    except Exception as e:
        client_logger.critical(f"A critical error occurred during task execution: {e}", exc_info=True)
        result = {"success": False, "reason": f"Critical runtime error: {e}"}

    # Log the result of the task execution
    client_logger.info("\n--- Task Execution Summary (inside client) ---")
    if result:
        # Log summary details based on the result dictionary
        client_logger.info(f"Success: {result.get('success', False)}")
        client_logger.info(f"Reason: {result.get('reason', 'N/A')}")
        actions_count = result.get('overall_actions_performed_count', 'N/A')
        max_actions_allowed_val = getattr(coordinator_instance, 'max_overall_actions_performed', 'N/A')
        client_logger.info(f"Total individual actions performed: {actions_count} (Configured Max: {max_actions_allowed_val})")
    else:
        client_logger.error("No result dictionary available after task execution attempt.")
        result = {"success": False, "reason": "Task execution failed to produce a result dictionary."} 

    client_logger.info("--- End of Client Summary ---")

    return result
