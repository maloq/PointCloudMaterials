# mace-causal-implementation-smoke-20260916

[All datasets](../README.md) · [Browsable card](mace-causal-implementation-smoke-20260916-00d29d10.html) · [Full metadata](../records/mace-causal-implementation-smoke-20260916-00d29d10.json)

Previously unregistered derived data. Inspect schemas, source plans and cache protocol before reuse.

- ID: `mace-causal-implementation-smoke-20260916`
- Materials: Al
- Classification: **fixture**; role: training_cache
- Location: `/home/ids/vmorozov/training-cache/mace-causal-implementation-smoke-20260916`
- Present on this machine: True
- Potentials: Unknown / not applicable
- Complete binary records with arrays present: 0; these are not independent-source counts.
- Stored frames: 0; duplicate-group records: 0
- Allocated storage, excluding registered nested datasets: 0.025 GiB
- Missing metadata: generating potential identity

## Notes and relationships

```json
{
  "title": "mace-causal-implementation-smoke-20260916",
  "role": "training_cache",
  "classification": "fixture",
  "materials": [
    "Al"
  ],
  "description": "Previously unregistered derived data. Inspect schemas, source plans and cache protocol before reuse.",
  "evidence": [
    "${dataset:mace-causal-implementation-smoke-20260916}"
  ]
}
```

## Recorded fields

| Field | Values |
| --- | --- |
| protocol | ["mace_causal_physical_state_v1"] |
| units | [[{"label_sha256": "49e7c932f56ed2cd2b7c052c9a1ab3c4ea077dbc6bbe2f57cfb38fa6d8c3a6c3", "path": "source-0860.pt", "samples": 8, "sha256": "80097414f4837fad883263e8be09b2aef95685e2dd3e90f33674a1fe94ed67b3", "source_id": 860, "source_identity": {"manifest_sha256": "eb84ac1ca4c1d44ee2cc70174f360fa416893d6bf61859972a68e4dff6d1541e", "selected_arrays_sha256": "ab285119b3fc45b74f72917bbc09a071080707d0e13d079b629b2bb6d3301adb", "storage_dtype": "float16"}}, {"label_sha256": "be28401e2bf6522c230304713b0759dced661dc68ee6b2bc4204bc3f67fda56f", "path": "source-0861.pt", "samples": 8, "sha256": "9b1eaf89f5e03f87ad4e99b0a2ccac14c1b083310c4a2c294fbe4323fb89c8c0", "source_id": 861, "source_identity": {"manifest_sha256": "1a1e8f1135a4038d296d4e43d5141dc104a90e40cfa10164a4c913965b04c631", "selected_arrays_sha256": "9b3c16856e92570f2800ebe827389ee9cb0287cb152b3bb3798cd0a66ec3e870", "storage_dtype": "float16"}}, {"label_sha256": "c7a66d98605291fdeea5c6d3087d6c77c789197a639a3e81d7299e8e7d83683f", "path": "source-0875.pt", "samples": 8, "sha256": "cac85d5111935917aff7f9efbd0bdac9ac7c44467df869712edfea7ea91d6da7", "source_id": 875, "source_identity": {"manifest_sha256": "553e12062781331b663cf7bceb904ec3ea0e794c230aac866ff9993641f79892", "selected_arrays_sha256": "62ab52bb5313f871f73bdafa07e84030843ccf1a8481c2ca4ae2c8800f3ef312", "storage_dtype": "float16"}}, {"label_sha256": "0ae9a5f5c03c6cbf5ff4af2fa5d2f3cf48c49615579e6279c7a6a839bd401cc8", "path": "source-0876.pt", "samples": 8, "sha256": "62dc019db70ee8f8a53191e87eb8f06cb2b83ca6f175ec1152e9d092a28b2c2f", "source_id": 876, "source_identity": {"manifest_sha256": "aadf48427d665b6255029070b982ce171d2051df60b4ebeb318635307ea154ec", "selected_arrays_sha256": "4fcdb5bf5edb6bc436c588ce392179b29708c0a0ea99d3b4c466d440af7c0831", "storage_dtype": "float16"}}, {"label_sha256": "e53bab1cef97cc4cd1881ea92ef8b2ce9ab2d2ad4420081dfe4ecfb664897b87", "path": "source-0881.pt", "samples": 8, "sha256": "5b5a5b91b017ce8f0c9eee532789558514e6cfce787c9619ee0eadf4d6d63980", "source_id": 881, "source_identity": {"manifest_sha256": "1f84a0d56c1f30ed4625c3ebff71d6ebb25a6e2fd2f480cce993eb0b15916405", "selected_arrays_sha256": "92cdc4ea7b677036202db6738dd8e02272c7268d2615819a3dfcbfd472e64c52", "storage_dtype": "float16"}}, {"label_sha256": "b3eae8aaa0f8b9eecf05d856b39e74561f18f3c93b4633677deb17a14ddb77ee", "path": "source-0882.pt", "samples": 8, "sha256": "415aa7e8d03530d87aa0622c8bb188d451e3aea150ca426c8115ee71c4f9f129", "source_id": 882, "source_identity": {"manifest_sha256": "aab95d7579a9a8a5f3f57d97d43ed6f0ed0eb4f37e9df735748f36cf20100139", "selected_arrays_sha256": "ae5588f30e56b97c365b37efadc0a827ce66fb63c3f469a2cde4ff78b11baba3", "storage_dtype": "float16"}}]] |
| storage_dtype | ["float16"] |
| seed | [20260916] |
| history_offsets_ps | [[-2.25, -1.5, -0.75, 0.0]] |
| future_lags_ps | [[0.75, 1.5, 3.0]] |
| cutoff_A | [5.0] |
| atomic_numbers | [[13]] |
| ptm_rmsd_cutoff | [0.1] |
| split | ["test", "train", "val"] |
| lineage | ["independent_melt_410549883", "independent_melt_455949696", "independent_melt_511618678", "independent_melt_556912370", "independent_melt_617047783", "independent_melt_780540798"] |
| timestep_fs | [3.0] |
| frame_count | [801] |
| temperature_K | [520.0] |
| atom_count | [70304] |

## Evidence

All 2 producer records, their hashes, field paths and current array schemas: [metadata JSON](../records/mace-causal-implementation-smoke-20260916-00d29d10.json).

Sources remain at their original locations. Referenced configs/code do not prove actual training use.

Observed 2026-09-18T18:40:33.535128+00:00.

Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.
