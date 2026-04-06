import urllib.request

from cai.utils import safe_path


def register(mcp):
    @mcp.tool()
    def fetch_url(url: str) -> str:
        """Fetch the content of a URL and return it as text (e.g. docs, READMEs)."""
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")
