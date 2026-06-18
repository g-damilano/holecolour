# Registration hotfix for variable frame sizes

## Failure fixed
OpenCV `phaseCorrelate` crashed when consecutive frames did not have identical pixel dimensions:

- `cv2.error: ... phaseCorrelate ... (-215:Assertion failed) src1.size == src2.size`

## Root cause
`holecolor/registration/rigid.py` passed decoded frames directly into `cv2.phaseCorrelate(...)`.
That function requires the moving and fixed grayscale arrays to have identical width and height.

## Fix
A canvas-normalization helper now center-crops / reflect-pads every frame to the reference-frame size before:
- phase-correlation shift estimation
- transform application
- residual comparison

This makes registration robust to small per-frame size differences while standardizing the stabilized sequence onto the reference canvas.

## Files changed
- `holecolor/registration/rigid.py`
- `holecolor/tests/test_registration.py`

## Verification
- `pytest -q holecolor/tests/test_registration.py` -> 2 passed
- direct synthetic `run_milestone24(...)` smoke progressed past registration and emitted registration + geometry artifacts without the OpenCV size-mismatch crash
