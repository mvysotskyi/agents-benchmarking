from typing import Dict, Any, List, Optional
import re
from agisdk.REAL.browsergym.webclones.utils import generate_from_model
from agisdk.REAL.logging import logger as rich_logger
import jmespath
import json
from copy import deepcopy


def _normalize_env_state(env_state: Any) -> Any:
    """
    Normalize raw environment state into a JSON-like structure when possible.

    Callers sometimes pass the parsed /finish payload, a raw JSON string, or
    placeholder sentinels when no environment state is available yet.
    """
    if env_state is None:
        return {}

    if isinstance(env_state, str):
        stripped = env_state.strip()
        if not stripped or stripped == "<>":
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            # Preserve unexpected raw values for debugging without breaking
            # dict-only consumers such as final-state merging.
            return {"_raw": env_state}
        return parsed

    return env_state


def compute_final_state(env_state: Any) -> Any:
    """
    Compute the final state of entities from multiple create/edit operations.
    
    This function processes the environment state to merge sequential operations
    (CREATE, EDIT, UPDATE) on the same entity into a single final state representation.
    
    Handles multiple patterns found in the webclone sites:
    1. Array of operations with 'operation' field: createdAccounts, createdContacts, etc.
    2. Diff structures with 'added', 'updated', 'deleted' keys
    
    Args:
        env_state: Raw environment state from /finish endpoint
        
    Returns:
        Enhanced env_state with additional 'finalState' key containing merged entity states
    """
    env_state = _normalize_env_state(env_state)
    if not isinstance(env_state, dict) or not env_state:
        return env_state
    
    result = deepcopy(env_state)
    final_state = {}

    # Pattern 1: Process arrays with operation field (e.g., createdAccounts, editedAccounts)
    # We need to combine operations across related keys (created/edited/updated/deleted) for same entity type.
    operations_by_type: Dict[str, List[dict]] = {}
    for key, value in env_state.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            normalized_type, inferred_operation = _normalize_entity_type_and_operation(key)
            ops = []
            for item in value:
                op_item = deepcopy(item)
                # If operation missing, infer from key
                if 'operation' not in op_item and inferred_operation:
                    op_item['operation'] = inferred_operation
                ops.append(op_item)
            operations_by_type.setdefault(normalized_type, []).extend(ops)

    for entity_type, ops in operations_by_type.items():
        merged_list = _merge_operations_array(ops)
        if merged_list:
            final_key = f"final{entity_type}"
            final_state[final_key] = merged_list
    
    # Pattern 2: Process diff structures (e.g., eventsDiff, profilesDiff)
    for key, value in env_state.items():
        if isinstance(value, dict) and key.endswith('Diff'):
            entity_type = key.replace('Diff', '')
            merged_list = _merge_diff_structure(value)
            if merged_list:
                final_state[f"final{entity_type.capitalize()}"] = merged_list
    
    # Pattern 3: Process 'differences' structure (gomail style)
    if 'differences' in env_state and isinstance(env_state['differences'], dict):
        for entity_key, diff_data in env_state['differences'].items():
            if isinstance(diff_data, dict):
                merged_list = _merge_diff_structure(diff_data)
                if merged_list:
                    final_state[f"final{entity_key.capitalize()}"] = merged_list
    
    if final_state:
        result['finalState'] = final_state
    
    return result


def _merge_operations_array(operations: List[dict]) -> List[dict]:
    """
    Merge an array of operations by entity ID into final states.
    
    Operations are applied in order: CREATE establishes base state,
    subsequent EDIT/UPDATE operations are merged on top.
    
    Args:
        operations: List of operation dicts with 'operation' field and optionally 'id'
        
    Returns:
        Dict mapping entity IDs to their final merged state
    """
    entities: Dict[str, dict] = {}
    last_ts: Dict[str, str] = {}
    
    # Apply operations in chronological order if timestamp present
    def _ts(o):
        return o.get("timestamp") or o.get("_timestamp") or ""
    operations_sorted = sorted(operations, key=_ts)

    for op in operations_sorted:
        # Try to find entity identifier (common keys: id, _id, entityId, name)
        entity_id = None
        for id_key in ['id', '_id', 'entityId', 'accountId', 'contactId', 'name', 'email']:
            if id_key in op and op[id_key]:
                entity_id = str(op[id_key])
                break
        
        if not entity_id:
            # Generate sequential ID for entities without explicit ID
            entity_id = f"entity_{len(entities)}"
        
        operation_type = op.get('operation', 'CREATE').upper()
        
        ts_val = _ts(op)
        if ts_val:
            last_ts[entity_id] = ts_val

        if operation_type == 'CREATE' or entity_id not in entities:
            # Initialize or overwrite with CREATE
            entities[entity_id] = deepcopy(op)
        elif operation_type in ('EDIT', 'UPDATE', 'MODIFY'):
            # Apply field-level changes if present (changes.old/new)
            if isinstance(op.get('changes'), dict):
                _apply_changes(entities[entity_id], op['changes'])
            # Merge EDIT/UPDATE onto existing entity
            _deep_merge(entities[entity_id], op)
        elif operation_type == 'DELETE':
            # Mark as deleted but keep for reference
            if entity_id in entities:
                entities[entity_id]['_deleted'] = True
                entities[entity_id]['_deletedAt'] = op.get('timestamp', op.get('_timestamp'))
    
    # Return a list to avoid id collisions and align with created/edited list shape
    # Sort by timestamp if present
    def _ts_key(e: dict) -> str:
        return last_ts.get(str(e.get("id")) or str(e.get("_id")) or "", "")

    merged_list = list(entities.values())
    merged_list.sort(key=_ts_key)
    return merged_list


def _merge_diff_structure(diff: dict) -> dict:
    """
    Merge a diff structure (added/updated/deleted) into final entity states.
    
    Args:
        diff: Dict with optional 'added', 'updated', 'deleted' keys
        
    Returns:
        Dict mapping entity IDs to their final state
    """
    entities: Dict[str, dict] = {}
    last_ts: Dict[str, str] = {}
    
    # Process 'added' items first (they establish base state)
    added = diff.get('added', {})
    if isinstance(added, dict):
        for entity_id, entity_data in added.items():
            if isinstance(entity_data, dict):
                entities[str(entity_id)] = deepcopy(entity_data)
                entities[str(entity_id)]['_operation'] = 'added'
                ts_val = entity_data.get("timestamp") or entity_data.get("_timestamp") or ""
                if ts_val:
                    last_ts[str(entity_id)] = ts_val
    elif isinstance(added, list):
        for i, entity_data in enumerate(added):
            if isinstance(entity_data, dict):
                entity_id = entity_data.get('id', str(i))
                entities[str(entity_id)] = deepcopy(entity_data)
                entities[str(entity_id)]['_operation'] = 'added'
                ts_val = entity_data.get("timestamp") or entity_data.get("_timestamp") or ""
                if ts_val:
                    last_ts[str(entity_id)] = ts_val

    # Process 'updated' items (merge onto existing or create new)
    updated = diff.get('updated', {})
    if isinstance(updated, dict):
        for entity_id, entity_data in updated.items():
            entity_id = str(entity_id)
            if isinstance(entity_data, dict):
                if entity_id in entities:
                    _deep_merge(entities[entity_id], entity_data)
                    entities[entity_id]['_operation'] = 'updated'
                else:
                    entities[entity_id] = deepcopy(entity_data)
                    entities[entity_id]['_operation'] = 'updated'
                ts_val = entity_data.get("timestamp") or entity_data.get("_timestamp") or ""
                if ts_val:
                    last_ts[entity_id] = ts_val
    elif isinstance(updated, list):
        for entity_data in updated:
            if isinstance(entity_data, dict):
                entity_id = str(entity_data.get('id', len(entities)))
                if entity_id in entities:
                    _deep_merge(entities[entity_id], entity_data)
                    entities[entity_id]['_operation'] = 'updated'
                else:
                    entities[entity_id] = deepcopy(entity_data)
                    entities[entity_id]['_operation'] = 'updated'
                ts_val = entity_data.get("timestamp") or entity_data.get("_timestamp") or ""
                if ts_val:
                    last_ts[entity_id] = ts_val

    # Process 'deleted' items (mark as deleted)
    deleted = diff.get('deleted', {})
    if isinstance(deleted, dict):
        for entity_id, entity_data in deleted.items():
            entity_id = str(entity_id)
            if entity_id in entities:
                entities[entity_id]['_deleted'] = True
            else:
                entities[entity_id] = deepcopy(entity_data) if isinstance(entity_data, dict) else {'id': entity_id}
                entities[entity_id]['_deleted'] = True
            ts_val = entity_data.get("timestamp") if isinstance(entity_data, dict) else ""
            if ts_val:
                last_ts[entity_id] = ts_val
    elif isinstance(deleted, list):
        for entity_data in deleted:
            if isinstance(entity_data, dict):
                entity_id = str(entity_data.get('id', len(entities)))
                if entity_id in entities:
                    entities[entity_id]['_deleted'] = True
                else:
                    entities[entity_id] = deepcopy(entity_data)
                    entities[entity_id]['_deleted'] = True
                ts_val = entity_data.get("timestamp") or entity_data.get("_timestamp") or ""
                if ts_val:
                    last_ts[entity_id] = ts_val
    
    # Return list to align with created/edited shape
    def _ts_key(e: dict) -> str:
        return last_ts.get(str(e.get("id")) or "", "")

    merged_list = list(entities.values())
    merged_list.sort(key=_ts_key)
    return merged_list


def _deep_merge(base: dict, updates: dict) -> None:
    """
    Deep merge updates into base dict, modifying base in place.
    
    Args:
        base: Base dictionary to merge into
        updates: Dictionary with updates to apply
    """
    for key, value in updates.items():
        if key == 'operation':
            # Track the latest operation type
            base['_lastOperation'] = value
        elif key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)


def _apply_changes(base: dict, changes: dict) -> None:
    """
    Apply a changes diff (with old/new) onto the base entity to materialize final field values.
    """
    for field, diff in changes.items():
        if isinstance(diff, dict) and 'new' in diff:
            base[field] = deepcopy(diff['new'])


def _normalize_entity_type_and_operation(key: str) -> tuple[str, Optional[str]]:
    """
    Normalize entity type from list keys like:
    - createdAccounts, editedAccounts, updated_accounts, deleted-accounts, createdAccountList
    Also infer default operation from the key when items do not carry it.
    """
    prefixes = {
        'created': 'CREATE',
        'edited': 'EDIT',
        'updated': 'UPDATE',
        'deleted': 'DELETE'
    }

    key_clean = key.strip()
    key_lower = key_clean.lower()

    for prefix, op in prefixes.items():
        if key_lower.startswith(prefix):
            remainder = key_clean[len(prefix):]
            remainder = re.sub(r'^[_\-\s]+', '', remainder)  # drop leading separators
            return _canonical_entity_type(remainder), op

    return _canonical_entity_type(key_clean), None


def _canonical_entity_type(name: str) -> str:
    """
    Produce a stable entity type label for finalState keys.
    """
    if not name:
        return "Entity"
    # strip separators at start
    name = re.sub(r'^[_\-\s]+', '', name)
    if not name:
        return "Entity"
    # Capitalize the first character to keep finalState keys readable
    return name[0].upper() + name[1:]


class WebCloneEvaluator:
    def __init__(self, task_config: Dict[str, Any], llm: str = ""):
        """
        Initializes the evaluator with an optional LLM instance for fuzzy matching.
        
        Args:
            task_config: The task configuration
            llm: The model name to use
        """
        self.llm = llm
        self.task_config = task_config
    
    def jmespath_verify(self, env_state: dict, query:str):
        """
        run jmespath query evals on data, see if they return true.
        """
        try:
            is_valid = jmespath.search(query, env_state)
        except Exception as e:
            return False, f"Error: {e}"
        return is_valid, None
    def get_value_from_path(self, env_state: dict, path: str):
        """Helper function to retrieve a value from a nested JSON (env_state) using a dot-separated path."""
        keys = path.split(".")
        value = env_state
        error_message = None
        for key in keys:
            if not isinstance(value, dict):
                error_message = f"Error: {path} was not found in the environment state."
                return f"<env state '{path}' not found>", error_message
            value = value.get(key)
            if value is None:
                break
        return value, None

    def evaluate_with_llm(self, model_response: str, rubric: str, threshold: float = 0.8):
        """Performs fuzzy matching using an LLM."""
        fuzzy_match_prompt = f"""
            Given a student's answer and a rubric, help a teacher grade the answer. Keep in mind
            that the student may use different words or phrases to express the same idea.

            Student's answer: {model_response}
            Rubric: {rubric}

            Grade the student's answer on a scale of 0 to 1, where 1 means the student's answer matches the rubric. Don't be too strict.
            Please answer only with a floating point number and nothing else.
        """
        llm_grade = generate_from_model(prompt=fuzzy_match_prompt, model=self.llm)
        # print(f"LLM grade: {llm_grade}")
        try:
            similarity = float(llm_grade)
        except ValueError:
            similarity = 0.0
            raise ValueError("LLM response is not a valid floating point number: {llm_grade}")
        is_correct = similarity > threshold
        info = {"similarity": similarity, "model_response": model_response, "rubric": rubric}
        # print(info)
        return is_correct, info

    def exact_match(self, actual_value: str, expected_value: str):
        """Checks if the actual value matches the expected value."""
        is_correct = actual_value == expected_value
        info = {"actual_value": actual_value, "expected_value": expected_value}
        return is_correct, info

    def evaluate(self, env_state: dict = None, model_response: str = None, use_final_state: bool = True):
        """
        Evaluate the task based on environment state and model response.
        
        Args:
            env_state: The raw environment state from /finish endpoint
            model_response: The model's response text
            use_final_state: If True, compute and include merged final state of entities
                           that underwent multiple create/edit operations. Default: True
        
        Returns:
            Tuple of (reward, done, message, info)
        """
        results = []
        env_state = _normalize_env_state(env_state)
        
        # Optionally compute final state from multiple operations
        if use_final_state and isinstance(env_state, dict) and env_state:
            env_state = compute_final_state(env_state)
            if 'finalState' in env_state:
                rich_logger.info("📊 Computed Final State (merged from create/edit operations):")
                final_state_str = json.dumps(env_state['finalState'], indent=4)
                rich_logger.print(f"[cyan]{final_state_str}[/cyan]")
        
        # Display environment state using Rich logging
        rich_logger.info("🌍 Environment State:")
        env_state_str = json.dumps(env_state, indent=4)
        rich_logger.print(f"[dim]{env_state_str}[/dim]")
        for i, eval in enumerate(self.task_config.get_evals()):
            if eval.type == "llm_boolean":
                is_correct = self.evaluate_with_llm(model_response, eval.rubric)
                results.append(is_correct)
                eval_outcome = f"model response: {model_response}, rubric: {eval.rubric}, is_correct: {is_correct[0]}"
            elif eval.type == "jmespath":
                actual_value, error_message = self.jmespath_verify(env_state, eval.query)
                if error_message:
                    is_correct = (False, error_message)
                    actual_value = error_message
                else:
                    is_correct = self.exact_match(actual_value, eval.expected_value)
                results.append(is_correct)
                eval_outcome = f"actual value: {actual_value} expected value: {eval.expected_value} , is_correct: {is_correct[0]}"
                                
                
            else:
                raise ValueError(f"Unknown evaluation type: {eval.type}")
            # Display criterion evaluation using Rich logging
            if is_correct[0]:
                rich_logger.success(f"✅ Criterion {i} {eval.description}: [{eval_outcome}]")
            else:
                rich_logger.error(f"❌ Criterion {i} {eval.description}: [{eval_outcome}]")


        is_correct = all(result[0] for result in results)
        reward = self.task_config.task.points if is_correct else 0.0
        done = True  # Task is always considered done after evaluation
        message = "Task completed successfully!" if is_correct else "Task not completed successfully."
        info = {"results": results}
        return reward, done, message, info
