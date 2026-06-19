# List of workflows and actions
This folder contains workflows that are helpful for maintaining a smooth and secure development process. The workflows should be enabled for open-source projects.

Workflows:
1. `qcom-preflight-checks.yml` - This workflow runs several preflight checks, including copyight, email, repolinter, and security checks.  See [qualcomm/qcom-actions](https://github.com/qualcomm/qcom-actions)
2. `stale-issues.yaml` - This workflow will periodically run every 30 days to check for stalled issues and PRs. If the workflow detects any stalled issues and/or PRs, it will automatically leave just a comment to draw attention.
3. `build-and-test.yml` - On every PR and push to `main`, compiles the full tree (`geniex_core`, `geniex_vlm`, and all LLM/VLM examples) and runs the CPU-only unit tests (CTest label `unit`) on a Windows ARM64 runner. The unit tests need no NPU; on-device generation testing is done separately on a Snapdragon device.