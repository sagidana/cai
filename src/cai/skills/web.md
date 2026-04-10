tools: fetch_url, search, read_file
---
## Skill: Web Research

Use `fetch_url` to retrieve primary sources: official documentation, CVE pages, vendor advisories, source repositories, RFCs.

Fetch discipline:
- Fetch the actual page, not a summary or cached version, when accuracy matters.
- For CVEs: prefer NVD (`nvd.nist.gov`) and the vendor advisory over secondary write-ups.
- For library APIs: prefer the official docs or source over Stack Overflow or blog posts — APIs change.
- If a fetch fails or returns unhelpful content, note it explicitly and try an alternate source.

Research posture:
- Cite every claim that comes from a fetched source: URL and the specific section or quote.
- When comparing versions or checking if a vulnerability affects a specific release, fetch the changelog or release notes directly — do not assume.
- For security advisories: note the CVSS score, affected versions, fix version, and whether a PoC is public.

Synthesis:
- Cross-reference at least two sources for any security-critical finding before concluding.
- If sources conflict, note the conflict explicitly rather than picking one silently.
