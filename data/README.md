# data/

Put the competition files in `raw/` under these names. `scripts/00_check_data.py` checks them
against the authors' copies (sha256, first 16 characters).

| put at | competition file | sha256[:16] |
|---|---|---|
| `raw/train_features.csv` | `train_features_TMQTWsB.csv` | `71bea3abb76a1cff` |
| `raw/train_labels.csv` | `train_labels_44ujmj2.csv` | `d98ee4389e5cde3f` |
| `raw/train_transcripts.zip` | `train_transcripts.zip` | `e685b85b04694e13` |
| `raw/test_features.csv` | `test_features_fRc23PD.csv` | `fffe0b1a5371b6e6` |
| `raw/test_labels.csv` | `test_labels.csv` | `998e9c0d64f50450` |
| `raw/test_transcripts.zip` | `test_transcripts.zip` | `4fec5947771fd4f2` |
| `raw/qualitative_metadata.csv` | `qualitative_metadata.csv` | `2965dbd8c219ba52` |

`folds/` holds the cross-validation folds (row indices only), aligned to the row order of
`raw/train_features.csv`. `derived/` is created by the scripts.
