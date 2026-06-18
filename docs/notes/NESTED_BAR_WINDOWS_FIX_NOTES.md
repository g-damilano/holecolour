Nested bar Windows fix

This build re-enables the nested stage bar on Windows terminals in a compatibility mode:
- optional colorama console fixup when available
- ASCII tqdm bars on Windows
- fixed-width bar rendering on Windows instead of dynamic_ncols
- stderr-targeted rendering for both master and nested bars

The goal is to keep the nested bar visible without the rendering regressions that led to the earlier fallback of disabling it.
