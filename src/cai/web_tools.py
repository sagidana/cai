import urllib.request

from cai.utils import safe_path


def register(mcp):
    @mcp.tool()
    def fetch_url(url: str) -> str:
        """Fetch the content of a URL and return it as text (e.g. docs, READMEs)."""
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")


if __name__ == "__main__":
    import sys

    class _MockMCP:
        def __init__(self): self._tools = {}
        def tool(self):
            def dec(fn): self._tools[fn.__name__] = fn; return fn
            return dec

    mcp = _MockMCP()
    register(mcp)
    T = mcp._tools

    _pass = _fail = 0
    def check(name, cond, got=""):
        global _pass, _fail
        if cond:
            print(f"  PASS  {name}")
            _pass += 1
        else:
            print(f"  FAIL  {name}  →  {got!r}")
            _fail += 1

    print("=== web_tools tests ===")

    # fetch a reliable URL
    try:
        r = T["fetch_url"]("https://www.example.com")
        check("fetch_url returns HTML", "<html" in r.lower() or "<!doctype" in r.lower(), r[:200])
        check("fetch_url example domain", "example" in r.lower(), r[:200])
    except Exception as e:
        print(f"  SKIP  fetch_url (network unavailable: {e})")

    # fetch a URL that returns JSON
    try:
        r = T["fetch_url"]("https://httpbin.org/json")
        check("fetch_url JSON response", "{" in r, r[:200])
    except Exception as e:
        print(f"  SKIP  fetch_url JSON (network unavailable: {e})")

    # fetch invalid URL raises exception (tool does not swallow it)
    try:
        T["fetch_url"]("http://localhost:19999/nonexistent")
        check("fetch_url bad port raises", False, "no exception raised")
    except Exception:
        check("fetch_url bad port raises", True)

    print(f"\n{_pass} passed, {_fail} failed")
    sys.exit(0 if _fail == 0 else 1)
