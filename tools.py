from mcp.server.fastmcp import FastMCP
import subprocess
import json

def send_rpc(process, method, params, request_id):
    message = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params
    }
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())

def call_tool(tool_name, arguments):
    print(f"[-] about to call {tool_name} ({arguments})")
    process = subprocess.Popen(
        ["python", __file__],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        send_rpc(process,
                 "initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "manual-subproc-client", "version": "1.0" }
                 }, 1)


        process.stdin.write(json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized"
        }) + "\n")
        process.stdin.flush()

        response = send_rpc(process,
                            "tools/call", {
                                "name": tool_name,
                                "arguments": arguments
                            }, 2)

        result = response.get("result", {}).get("content", [{}])[0].get("text")
        print(f"[-] result: {result}")

        return result
    except Exception as e:
        print(f"[!] call_tool exception: {e}")
    finally:
        process.terminate()

def get_tools():
    process = subprocess.Popen(
        ["python", __file__],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        send_rpc(process,
                 "initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "tool-lister", "version": "1.0"}
                }, 1)

        response = send_rpc(process, "tools/list", {}, 2)

        mcp_tools = response.get("result", {}).get("tools", [])
        openai_tools = []
        for tool in mcp_tools:
            openai_tools.append({
                "type": "function",
                "function":{
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"]
                    }
                })
        return openai_tools
    finally:
        process.terminate()


if __name__ == '__main__':
    mcp = FastMCP(name="Tools Server")

    @mcp.tool()
    def fetch_codebase_infra(cwd: str = '.') -> str:
        """Iterate through all Python files in the given directory and extract all classes and their methods using tree-sitter."""
        import os
        from tree_sitter_language_pack import(get_language, get_parser)
        import tree_sitter
        infra = {}
        parser = get_parser('python')
        language = get_language('python')

        class_query = language.query("""
            (class_definition
                name: (identifier) @class_name
                body: (block
                    (function_definition
                        name: (identifier) @method_name
                        parameters: (parameters) @method_params
                    )
                )
            ) @class_def
        """)

        func_query = language.query("""
            (module
                (function_definition
                    name: (identifier) @func_name
                    parameters: (parameters) @func_params
                )
            )
        """)

        for root, dirs, files in os.walk(cwd):
            for filename in files:
                if not filename.endswith('.py'): continue

                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'rb') as f:
                        source = f.read()
                except (IOError, OSError):
                    continue

                tree = parser.parse(source)
                root_node = tree.root_node

                file_info = {}

                cursor = tree_sitter.QueryCursor(class_query)
                class_captures = cursor.captures(root_node)

                class_defs = class_captures.get("class_def", [])
                class_names = class_captures.get("class_name", [])
                method_names = class_captures.get("method_name", [])
                method_params = class_captures.get("method_params", [])

                class_node_map = {}
                for cls_node, cls_name_node in zip(class_defs, class_names):
                    cls_name = cls_name_node.text.decode('utf-8')
                    class_node_map[cls_node.id] = cls_name
                    if cls_name not in file_info:
                        file_info[cls_name] = {}

                for meth_node, param_node in zip(method_names, method_params):
                    meth_name = meth_node.text.decode('utf-8')
                    param_text = param_node.text.decode('utf-8')
                    parent = meth_node.parent
                    while parent is not None:
                        if parent.type == 'class_definition': break
                        parent = parent.parent
                    if parent is not None and parent.id in class_node_map:
                        cls_name = class_node_map[parent.id]
                        prototype = f"def {meth_name}{param_text}"
                        file_info[cls_name][meth_name] = prototype

                cursor = tree_sitter.QueryCursor(func_query)
                func_captures = cursor.captures(root_node)
                func_names = func_captures.get("func_name", [])
                func_params_list = func_captures.get("func_params", [])

                for fn_node, fp_node in zip(func_names, func_params_list):
                    fn_name = fn_node.text.decode('utf-8')
                    fp_text = fp_node.text.decode('utf-8')
                    prototype = f"def {fn_name}{fp_text}"
                    file_info[fn_name] = prototype

                if file_info:
                    rel_path = os.path.relpath(filepath, cwd)
                    infra[rel_path] = file_info

        return json.dumps(infra, indent=2)

    mcp.run(transport="stdio")
