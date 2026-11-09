import ast
import re

from search_r1.tau2_adapter.core.message import AssistantMessage, ToolCall, UserMessage


def _evaluate_ast_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_ast_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")
    if isinstance(node, ast.List):
        return [_evaluate_ast_node(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_ast_node(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _evaluate_ast_node(key): _evaluate_ast_node(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.Name):
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
        return node.id
    raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def is_functional_tool_call(text: str) -> bool:
    return bool(re.match(r"^\w+\s*\(.*\)$", text.strip()))


def parse_functional_tool_call(functional_call: str, requestor: str = "assistant") -> ToolCall:
    match = re.match(r"^(\w+)\s*\((.*)\)$", functional_call.strip())
    if not match:
        raise ValueError(f"Invalid functional call format: {functional_call}")

    function_name = match.group(1)
    arguments_str = match.group(2).strip()
    arguments = {}

    if arguments_str:
        tree = ast.parse(f"dummy({arguments_str})")
        call_node = tree.body[0].value
        for keyword in call_node.keywords:
            arguments[keyword.arg] = _evaluate_ast_node(keyword.value)

    return ToolCall(name=function_name, arguments=arguments, requestor=requestor)


def parse_action_string(action: str, requestor: str = "assistant"):
    original_action = action
    action = action.strip()
    if not action:
        raise ValueError("Action cannot be empty")

    message_cls = UserMessage if requestor == "user" else AssistantMessage

    try:
        tool_call = ToolCall.model_validate_json(action)
        return message_cls(role=requestor, content=None, tool_calls=[tool_call], raw_data={"action": original_action})
    except Exception:
        pass

    if is_functional_tool_call(action):
        try:
            tool_call = parse_functional_tool_call(action, requestor=requestor)
            return message_cls(role=requestor, content=None, tool_calls=[tool_call], raw_data={"action": original_action})
        except Exception:
            pass

    return message_cls(role=requestor, content=original_action, tool_calls=None, raw_data={"action": original_action})
