import pandas as pd
from IPython.display import display, HTML
from typing import List, Dict, Tuple, Any, Optional
import html 

def _get_plan_segment_info(executed_actions: List[Dict]) -> Tuple[int, Dict[int, int], List[int]]:
    """
    Identifies plan segments from executed_actions and counts replans based on action flow.
    A new plan segment starts when sub_task_index resets to 0.
    """
    if not executed_actions:
        return 0, {}, []

    action_to_plan_segment: Dict[int, int] = {}
    plan_start_overall_steps: List[int] = []
    plan_segment_counter = 0
    last_sub_task_index = -1 

    for action_idx, action in enumerate(executed_actions):
        current_sub_task_index = action.get('sub_task_index')
        overall_step = action.get('overall_action_step')

        if overall_step is None: 
            continue

        is_new_segment_start = False
        if not plan_start_overall_steps: 
            is_new_segment_start = True
        elif isinstance(current_sub_task_index, int) and current_sub_task_index == 0:
            if not (isinstance(last_sub_task_index, int) and last_sub_task_index == 0):
                is_new_segment_start = True
        
        if is_new_segment_start:
            plan_segment_counter += 1
            plan_start_overall_steps.append(overall_step)
            
        action_to_plan_segment[overall_step] = plan_segment_counter
        
        if isinstance(current_sub_task_index, int):
            last_sub_task_index = current_sub_task_index

    num_replans_inferred = max(0, plan_segment_counter - 1)
    return num_replans_inferred, action_to_plan_segment, plan_start_overall_steps

def enhance_actions_dataframe_for_display(actions: List[Dict], action_to_plan_segment_map: Optional[Dict[int, int]] = None):
    """
    Creates a DataFrame from action records, adding plan segment info.
    """
    if not actions:
        return pd.DataFrame()

    display_data = []
    if action_to_plan_segment_map is None: 
        action_to_plan_segment_map = {}

    for action in actions:
        overall_step = action.get('overall_action_step', 'N/A')
        sub_task_idx_val = action.get('sub_task_index')
        record = {
            'Overall Step': overall_step,
            'Plan Segment': action_to_plan_segment_map.get(overall_step, 'N/A') if isinstance(overall_step, int) else 'N/A',
            'Sub-Task Idx': sub_task_idx_val + 1 if isinstance(sub_task_idx_val, int) else 'N/A', 
            'Sub-Task Desc': action.get('sub_task_description', 'N/A'),
            'Attempt': action.get('attempt', 'N/A'),
            'Worker Step': action.get('step_in_sub_task', 'N/A'), 
            'Action Type': action.get('action', 'unknown'),
        }
        
        details = []
        if action.get('action') == 'type':
            content = action.get('content', '')
            display_content = html.escape((content[:47] + '...') if len(content) > 50 else content)
            details.append(f"Content: '{display_content}'")
        elif action.get('action') == 'hotkey':
            details.append(f"Key: '{html.escape(action.get('key', ''))}'")
        elif action.get('action') == 'scroll':
            details.append(f"Dir: {html.escape(action.get('direction', ''))}")
            if coords := action.get('coordinates'): 
                details.append(f"Box: ({coords[0]},{coords[1]})")
        elif action.get('action') in ['click', 'left_double', 'right_single']:
            if coords := action.get('coordinates'):
                details.append(f"Coords: ({coords[0]},{coords[1]})")
        elif action.get('action') == 'drag':
            start_coords = action.get('start_coordinates')
            end_coords = action.get('end_coordinates')
            if start_coords: details.append(f"Start: ({start_coords[0]},{start_coords[1]})")
            if end_coords: details.append(f"End: ({end_coords[0]},{end_coords[1]})")

        record['Details'] = "; ".join(details) if details else None
        action_thought = action.get('thought_for_action')
        record['Action Thought'] = html.escape(action_thought) if action_thought else None

        display_data.append(record)

    df = pd.DataFrame(display_data)
    cols_order = [
        'Overall Step', 'Plan Segment', 'Sub-Task Idx', 'Sub-Task Desc', 'Attempt', 'Worker Step',
        'Action Type', 'Details', 'Action Thought'
    ]
    existing_cols_order = [col for col in cols_order if col in df.columns]
    return df[existing_cols_order]


def display_coordinator_results(result_data: Dict[str, Any]):
    """
    Displays a comprehensive summary and detailed trace from the Coordinator's result dictionary.
    """
    if not result_data:
        print("Error: No result data provided to display.")
        return

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 150)

    all_executed_actions = result_data.get("executed_actions", [])
    all_plans_archive = result_data.get("all_plans_archive", [])
    overall_success = result_data.get('success', False) 
    
    _, action_to_plan_segment_map, _ = _get_plan_segment_info(all_executed_actions)

    # --- Overall Summary ---
    display(HTML("<h2>Overall Task Summary</h2>"))
    reason = result_data.get('reason', 'N/A')
    actions_count = result_data.get('overall_actions_performed_count', 'N/A')
    max_actions = result_data.get('max_overall_actions_performed_in_run', 'N/A')
    num_replans_from_archive = max(0, len(all_plans_archive) - 1) if all_plans_archive else 0

    summary_html = f"""
    <b>Overall Success:</b> {overall_success}<br>
    <b>Reason for Outcome:</b> {html.escape(reason)}<br>
    <b>Total Actions Performed:</b> {actions_count} (Limit Configured: {max_actions})<br>
    <b>Number of Replans (Plan Segments - 1):</b> {num_replans_from_archive}
    """
    display(HTML(summary_html))

    # --- Plan Evolution / All Generated Plans ---
    display(HTML("<h2>Plan Evolution / All Generated Plans</h2>"))
    if all_plans_archive:
        plans_evolution_html = ""
        for i_archive, archived_plan_info in enumerate(all_plans_archive):
            segment_num = archived_plan_info.get("plan_segment_number", i_archive + 1)
            trigger = archived_plan_info.get("trigger_reason", "N/A")
            activated_step = archived_plan_info.get("activated_at_overall_step", "N/A")
            plan_steps = archived_plan_info.get("plan_steps", [])
            history_ctx_for_this_plan = archived_plan_info.get("history_context_for_this_replan") 
            raw_response = archived_plan_info.get("raw_planner_response", "")

            plans_evolution_html += f"<details {'open' if i_archive == len(all_plans_archive) -1 else ''}>"
            plans_evolution_html += f"<summary style='font-weight:bold; font-size:1.1em; margin-bottom:5px;'>Plan Segment #{segment_num} (Activated around Overall Step: {activated_step})</summary>"
            plans_evolution_html += "<div style='padding-left: 20px; border-left: 2px solid #ccc; margin-bottom:15px;'>"
            plans_evolution_html += f"<p><b>Generation Trigger:</b> {html.escape(trigger)}</p>"
            
            if history_ctx_for_this_plan: 
                plans_evolution_html += "<b>Context leading to this Replan:</b><ul>"
                prev_plan_from_hist = history_ctx_for_this_plan.get("previous_plan", [])
                prev_plan_text = "<li>Previous Plan (brief): "
                if prev_plan_from_hist:
                     prev_plan_text += f"{len(prev_plan_from_hist)} steps, starting with '{html.escape(str(prev_plan_from_hist[0])[:50])}...'</li>"
                else:
                    prev_plan_text += "Not available</li>"
                plans_evolution_html += prev_plan_text

                if history_ctx_for_this_plan.get("failed_sub_task_description"):
                    plans_evolution_html += f"<li>Failed Sub-Task in Prev. Plan: '{html.escape(history_ctx_for_this_plan['failed_sub_task_description'])}'</li>"
                plans_evolution_html += f"<li>Completed up to Index in Prev. Plan: {history_ctx_for_this_plan.get('completed_up_to_index', 'N/A')}</li>"
                attempt_summary = history_ctx_for_this_plan.get("attempt_summary", [])
                if attempt_summary:
                    plans_evolution_html += "<li>Attempt Summary of Prev. Plan Segment:<ul>"
                    for attempt_item in attempt_summary[:3]: 
                        outcome = html.escape(str(attempt_item.get('outcome','')))
                        sub_task_hist = html.escape(str(attempt_item.get('sub_task',''))[:70]+'...')
                        reason_hist = html.escape(str(attempt_item.get('reason','')))
                        plans_evolution_html += f"<li>'{sub_task_hist}' -> {outcome} {('('+reason_hist+')' if reason_hist else '')}</li>"
                    if len(attempt_summary) > 3:
                        plans_evolution_html += "<li>... (more entries)</li>"
                    plans_evolution_html += "</ul></li>"
                plans_evolution_html += "</ul>"

            plans_evolution_html += "<b>Plan Steps with Execution Status:</b>"
            annotated_plan_steps_html = "<ol>"
            
            if i_archive < len(all_plans_archive) - 1: 
                next_archived_plan = all_plans_archive[i_archive + 1]
                history_ctx_from_next = next_archived_plan.get('history_context_for_this_replan') 
                
                completed_idx_intermediate = -1
                failed_desc_intermediate = None
                if history_ctx_from_next: 
                    completed_idx_intermediate = history_ctx_from_next.get('completed_up_to_index', -1)
                    failed_desc_intermediate = history_ctx_from_next.get('failed_sub_task_description')

                for k_step, step_text in enumerate(plan_steps):
                    status_color = "grey"
                    status_text = "Pending when replanned"
                    if k_step <= completed_idx_intermediate:
                        status_color = "green"
                        status_text = "Processed in segment"
                    elif failed_desc_intermediate and k_step == completed_idx_intermediate + 1 :
                        status_color = "orange"
                        status_text = "Failed/Caused Replan Here"
                    
                    annotated_plan_steps_html += f"<li><span style='color:{status_color};'>({status_text})</span> {html.escape(step_text)}</li>"
            
            else: 
                num_processed_in_final_segment = result_data.get('completed_sub_tasks_in_final_plan', 0)
                
                for k_step, step_text in enumerate(plan_steps):
                    status_color = "grey"
                    status_text = "Pending"
                    if k_step < num_processed_in_final_segment:
                        status_color = "green"
                        status_text = "Processed/Completed"
                    elif k_step == num_processed_in_final_segment and not overall_success:
                        status_color = "red"
                        status_text = "Failed/Stopped Here"
                    
                    annotated_plan_steps_html += f"<li><span style='color:{status_color};'>({status_text})</span> {html.escape(step_text)}</li>"
            
            annotated_plan_steps_html += "</ol>"
            plans_evolution_html += annotated_plan_steps_html

            if raw_response:
                plans_evolution_html += f"<details><summary style='font-size:0.9em; color: #555;'>Raw Planner Response (Segment #{segment_num})</summary>"
                plans_evolution_html += f"<pre style='white-space: pre-wrap; background-color: #f9f9f9; border: 1px solid #eee; padding: 5px; font-size:0.8em;'>{html.escape(raw_response)}</pre></details>"
            
            plans_evolution_html += "</div></details><hr style='border-top: 1px dashed #ccc;'>"
        display(HTML(plans_evolution_html))
    else:
        display(HTML("<p>No archived plan history was found in the results.</p>"))

    display(HTML("<h2>Detailed Execution Trace</h2>"))
    all_worker_thoughts = result_data.get("worker_thoughts", [])

    if not all_executed_actions and not all_worker_thoughts:
        print("No detailed actions or worker thoughts were recorded.")
        return

    if all_executed_actions:
        display(HTML("<h3>Executed Actions Log</h3>"))
        try:
            actions_df = enhance_actions_dataframe_for_display(all_executed_actions, action_to_plan_segment_map)
            display(actions_df)
        except Exception as e:
            print(f"Error creating or displaying actions DataFrame: {e}")
    else:
        print("No actions were recorded.")

    if all_worker_thoughts:
        display(HTML("<h3>Chronological Worker Thoughts Log</h3>"))
        thoughts_html = "<div style='font-family: monospace; border: 1px solid #eee; padding: 10px;'>"
        current_thought_plan_segment = -1
        current_thought_sub_task_idx = -1

        for i, thought_info in enumerate(all_worker_thoughts):
            st_idx = thought_info.get('sub_task_index', -1)
            attempt = thought_info.get('attempt', 'N/A')
            thought_text = thought_info.get('thought_from_worker', '')
            context_step = thought_info.get('overall_action_step_context', 'N/A')
            
            plan_segment_for_thought = 'N/A'
            if isinstance(context_step, int) and action_to_plan_segment_map:
                plan_segment_for_thought = action_to_plan_segment_map.get(context_step, 'N/A')

            if plan_segment_for_thought != current_thought_plan_segment:
                thoughts_html += f"<div style='background-color: #f0f0f0; padding: 5px; margin-top:10px; font-weight:bold;'>--- Thoughts during Plan Segment {plan_segment_for_thought} ---</div>"
                current_thought_plan_segment = plan_segment_for_thought
                current_thought_sub_task_idx = -1

            if st_idx != current_thought_sub_task_idx:
                sub_task_display_idx = st_idx + 1 if isinstance(st_idx, int) else 'N/A'
                thoughts_html += f"<div style='margin-left: 10px; margin-top:5px; font-weight:bold;'>-- Sub-Task {sub_task_display_idx} (Attempt: {attempt}) --</div>"
                current_thought_sub_task_idx = st_idx
            
            thoughts_html += f"<div style='margin-left: 20px;'><i>Entry #{i+1} (Overall Step Context: {context_step})</i><br>"
            thoughts_html += f"<pre style='white-space: pre-wrap; margin-left: 10px; margin-top: 0px; margin-bottom: 5px;'>{html.escape(thought_text)}</pre></div>"

        thoughts_html += "</div>"
        display(HTML(thoughts_html))
    else:
        print("<p>No separate chronological worker thoughts were recorded.</p>")

    display(HTML("<hr style='border:1px solid black'><h3>End of Detailed Trace</h3><hr style='border:1px solid black'>"))