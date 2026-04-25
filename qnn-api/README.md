# qnn-api/

This directory contains files **extracted verbatim from the Qualcomm AI
Runtime (QAIRT) SDK**. They are *not* part of the Apache-2.0 licensed
portion of this project.

## Source

- **SDK:** Qualcomm AI Runtime SDK (QAIRT), also referred to as the
  Qualcomm AI Engine Direct SDK.
- **Download:** https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
- **Version at extraction:** v2.45.0.260326.

## License

These files are governed by the **QAIRT SDK End User License Agreement**
shipped with the SDK download, not by this project's Apache-2.0 license.
The original per-file Qualcomm copyright and "Confidential and Proprietary"
markings are preserved intentionally — they accurately describe the
licensing status of these files.

Do **not** rewrite or strip these headers. If you need to update the files,
replace them from a fresh SDK download (see *Refreshing* below).

See [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for the full
third-party component list.

## Layout

- `include/QNN/` — QAIRT public C API headers (`QnnTypes.h`, backend
  extension headers, etc.).
- `include/HTP/` — HTP (Hexagon Tensor Processor) backend headers.
- `include/*.hpp` — additional SDK-provided C++ helpers (`MmappedReader`,
  `QnnConfig`, etc.).
- `src/*.cpp` — SDK-provided wrapper implementations.

## Refreshing

To update these files to a newer SDK version:

1. Download the target QAIRT SDK from the link above.
2. From the extracted SDK, copy:
   - `include/QNN/**` → `qnn-api/include/QNN/`
   - `include/HTP/**` → `qnn-api/include/HTP/`
   - The matching sample/wrapper `.cpp` / `.hpp` files → `qnn-api/src/` and `qnn-api/include/`
3. Update the version recorded in this file and in
   `../THIRD_PARTY_NOTICES.md`.
4. Rebuild and run smoke tests against an existing model.
5. Update the bundled runtime binaries under `third-party/{windows,android,linux-gcc11.2}/` if they are from the same SDK release.
