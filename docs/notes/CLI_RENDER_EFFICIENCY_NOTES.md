CLI rendering efficiency notes

This build fixes two issues seen in the Windows CLI path:
- pipeline.py now imports columns_from_records for the validation-tail column boundary path
- status.py keeps the nested tqdm stage bar enabled, but removes the extra forced refresh path that was repainting the stage bar on every heartbeat and on extra master-bar refresh calls

The goal is to preserve nested tqdm while avoiding excessive console repaint churn and duplicate lines on Windows terminals.
