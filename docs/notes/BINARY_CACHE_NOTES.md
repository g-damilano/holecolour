Binary cache sidecars

Milestone 31 keeps user-facing CSV outputs but writes .pkl sidecars for the tables used by frame-analysis and validation cache reuse.

Why:
- avoids reparsing CSV into Python dicts on cache hits
- preserves numeric/None types without CSV coercion
- reduces rerun overhead in the heaviest cached stages

Behavior:
- CSV files are still written for inspection
- cache loaders prefer the .pkl sidecar when present
- if the sidecar is missing or unreadable, the loader falls back to CSV


Milestone 33 adds columnar .npz sidecars for numeric/string cache tables. Cache loads now prefer .npz over .pkl and CSV when available.
