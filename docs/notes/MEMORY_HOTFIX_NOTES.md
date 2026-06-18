# Memory hotfix notes

This build replaces the previous full-frame `H x W x N_holes` terrace-distance cube with a local per-hole terrace builder.

Why:
- The previous implementation could allocate multiple GiB for large frames with many holes.
- In multiprocessing runs that cost was replicated across workers.

What changed:
- terrace ownership is computed hole-by-hole inside a local crop
- only nearby competitor holes are compared
- local float32 distance fields are used
- downstream radial/sector consumers were updated to work with local terrace regions
- compatibility was kept for older full-mask terrace paths used by some perturbation utilities

Practical effect:
- 1080p / high-hole-count runs avoid the previous `ArrayMemoryError` in terrace generation
