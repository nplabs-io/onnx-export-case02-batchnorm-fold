# Raw results

Console output from the original runs (Windows, Python 3.11.9, CPU).
Exporter progress lines and library warnings were removed; every number is
unchanged. The "Windows 10" in the headers is how Python 3.11 reports
Windows 11 (`platform.release()`). The scripts were named `issue01_repro.py` / `issue01_narrow.py` at
the time; they are `repro.py` / `narrow.py` in this repo, with identical logic.

| File | Stack | Script |
|---|---|---|
| `01_broken_stack_repro.txt` | torch 2.9.0, onnxscript 0.5.6 | repro.py |
| `02_broken_stack_narrow.txt` | torch 2.9.0, onnxscript 0.5.6 | narrow.py |
| `03_fixed_onnxscript_narrow.txt` | torch 2.9.0, onnxscript 0.7.1 | narrow.py |
| `04_current_stack_repro.txt` | torch 2.11.0+cu130, onnxscript 0.7.1 | repro.py |
