# Dataset guide

## Primary dataset: VeReMi NextGen

The downloader retrieves the public [VeReMi NextGen record on Zenodo](https://zenodo.org/records/19665762). The project uses only the attacks that map cleanly to the requested classes.

| Project label | Official archive suffix | What it represents |
|---|---|---|
| Normal | no suffix | Legitimate vehicle messages |
| Sybil | `trafficCongestionSybil` | One attacker presents multiple false identities to simulate congestion |
| Illusion | `constantPositionOffset` | False position shifted by a constant offset |
| Illusion | `randomPositionOffset` | False position shifted by changing random offsets |
| Illusion | `positionMirroring` | False/mirrored position claims |

Each scenario therefore uses five ZIP archives: one baseline, one Sybil attack, and three Illusion attacks. Each outer archive contains the official nested Train, Validation, and Test archives. The program preserves these splits.

## Download profiles

| Profile | Scenarios | Files | Approximate download | Purpose |
|---|---:|---:|---:|---|
| `quick` | `highway_2` | 5 | 0.11 GiB | Code check and early experiments |
| `recommended` | `highway_2`, `urban_2` | 10 | 0.96 GiB | Best practical starting point |
| `full` | all four scenarios | 20 | 11.44 GiB | Maximum diversity and final research run |

Download a verified profile:

```bash
python download_data.py --profile recommended --output data/raw
```

The downloader obtains the official record manifest, checks every size and MD5 checksum, skips files that are already valid, and re-downloads invalid files.

## Why the project does not include generated samples

No rows are synthesized by an AI or by a hand-written fake-data generator. The project trains from the official simulated VANET traces only. Simulation data is valid for controlled research, but results on it do not automatically prove real-road performance. A real deployment should be tested on receiver logs from the target radio and environment.

## Leakage controls

- The official Train split fits the scaler and class weights.
- Validation selects the epoch, calibrates probability temperature, and selects the alert threshold.
- Test is used once for final reporting.
- Raw sender IDs, aliases, receiver IDs, filenames, labels, and attacker flags are never model features.
- Absolute sender coordinates are converted into motion consistency and receiver-relative features rather than fed directly to the network.
- Normal targets from attack archives are excluded by default, preventing the same baseline behavior from being counted repeatedly.

Ground-truth JSON files are not required because the attack records already identify attacker messages. They can be downloaded for independent auditing with `--include-ground-truth`, but the model never receives ground truth as an input feature.
