# Validation performed before packaging

The project was checked with Python 3.12 and PyTorch on CPU.

## Automated checks

- Python compilation of the package and command-line wrappers
- Parser behavior for numeric strings and attack labels
- Official nested ZIP split extraction
- Cold-start padding and masks for one-message Sybil identities
- RSSI missingness and same-receiver similarity handling
- Three-class model output and normalized attention weights
- End-to-end preprocessing of real VeReMi NextGen archive subsets
- Two-epoch CPU integration training, calibration, evaluation, and checkpoint export

## Official-data preprocessing smoke sample

The smoke procedure read up to 20 receiver files per official split from the `highway_2` baseline, `trafficCongestionSybil`, and `constantPositionOffset` archives. It produced all three labels in Train, Validation, and Test. This establishes parser and pipeline compatibility; it is not a scientific benchmark and its temporary model is intentionally excluded from the deliverable.

For publishable metrics, download the `recommended` or `full` profile and run the complete configuration. Use the untouched Test files only after model and threshold selection are finished.
