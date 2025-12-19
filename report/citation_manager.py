class CitationManager:
    def __init__(self):
        self.sources = []

    def collect(self, results):
        for r in results:
            self.sources.append(r["url"])

    def format_references(self):
        return "\n".join(
            [f"[{i+1}] {url}" for i, url in enumerate(set(self.sources))]
        )
