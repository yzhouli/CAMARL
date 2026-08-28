# Data contract

The dataset is not included in this GitHub repository. Download and extract the
separate dataset artifact from:

```text
REPLACE_WITH_CAMARL_DATASET_DOWNLOAD_URL
```

Set `CAMARL_DATASET_ROOT` to the extracted directory before running the
commands in the main README.

## Raw files

- `cascades.txt`: cascade events consumed by `build_protocol_pools.py`.
- `edges.txt`: directed user-user relations.
- `news_all.pkl`: news/topic metadata keyed by news ID.
- `users_all.pkl`: user metadata keyed by user ID.
- `test_aligned.pkl`: frozen aligned test records.
- `mm/mm/`: source images and videos referenced by news metadata.

Additional raw artifacts are retained in the dataset snapshot for provenance.

## Processed files

- `protocol/manifests.json`: deterministic split IDs and hashes.
- `protocol/{train,validation}_records_N1000.pkl`: protocol records.
- `graphhard_large/{validation,test}_graphhard_pools_N*.pkl`: nested candidate
  pools.
- `graphhard_v2/graphhard_protocol_report.json`: audit and protocol identity.
- `topic_frames/`: deterministic images/contact sheets and their manifest.

## Security warning

Python pickle is not safe for untrusted input. Verify release hashes before
loading any `.pkl` file. The dataset manifest generator records SHA-256 and
size for every file.
