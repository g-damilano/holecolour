# Parallelization and progress notes

This build introduces purposeful parallelization only in stages that are naturally independent across frames.

Main implementation choices:
- independent per-frame analysis is executed through a configurable parallel map abstraction
- process pools are preferred for larger CPU-bound independent workloads
- thread pools remain available as a lower-overhead fallback
- OpenCV worker thread counts are capped inside worker processes to reduce oversubscription
- tqdm-backed progress bars are used for frame loading and per-frame processing

Key references used to guide this implementation:
- Python `concurrent.futures` documentation:
  - https://docs.python.org/3/library/concurrent.futures.html
- Python `multiprocessing` documentation:
  - https://docs.python.org/3/library/multiprocessing.html
- tqdm documentation:
  - https://tqdm.github.io/
  - https://tqdm.github.io/docs/contrib.concurrent/
- joblib parallelization guidance:
  - https://joblib.readthedocs.io/en/latest/parallel.html
  - https://joblib.readthedocs.io/en/latest/auto_examples/parallel_memmap.html
- OpenCV thread-control guidance:
  - https://docs.opencv.org/3.4/db/de0/group__core__utils.html
  - https://docs.opencv.org/4.x/dc/ddf/tutorial_how_to_use_OpenCV_parallel_for_new.html


## Milestone 20 additions

- perturbation sweeps are now parallelized with a **thread backend by default** to avoid repeatedly serializing the full stabilized frame stack into worker processes
- directory image-stack loading now uses threaded loading when enough files are present
- each run now writes a Jupyter notebook explorer into `notebooks/holecolor_results_explorer.ipynb`
