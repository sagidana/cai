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
    # Initialize the MCP server instance
    mcp = FastMCP(name="Tools Server")

    @mcp.tool()
    def sum(a: int, b: int) -> int:
        """Add two numbers together.

        Args:
            a: The first integer.
            b: The second integer.
        """
        return a + b

    @mcp.tool()
    def get_temperature(city):
        """
        return the temperature degrees currently in the city provided
        Args:
            city: the city name
        """

        return 0


    # Run the server using standard I/O (stdio) transport
    mcp.run(transport="stdio")



