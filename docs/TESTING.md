# Testing and CI

## CI (GitHub Actions)

| Workflow | Trigger | Runner | Does |
|---|---|---|---|
| `ci.yml` | push (any branch), PR to main, manual | hosted | format check, CI-script tests, then `linux-build.yml` |
| `linux-build.yml` | called by CI | hosted | Docker build (host + NPU kernels + CPU tests), uploads `gufo-npu-linux-x86_64-<sha>` tarball, pushes the image on non-PR runs |
| `npu-smoke.yml` | manual | self-hosted `[linux, strix-halo, npu]` | pulls an image, runs `diagnose` and `smoke` on the NPU |
| `pr-title.yml` | PR | hosted | Conventional Commit title |
| `release-please.yml` | daily, manual | hosted | release PR and tags |

Images: `ghcr.io/vantreeseba/gufo-npu:<branch>` and `:sha-<commit>`.

## Testing a CI build on your Strix Halo

```sh
docker pull ghcr.io/vantreeseba/gufo-npu:<branch>
run="docker run --rm --device /dev/accel/accel0 --ulimit memlock=-1:-1 -v /opt/xilinx/xrt:/opt/xilinx/xrt:ro"
$run ghcr.io/vantreeseba/gufo-npu:<branch> diagnose
$run ghcr.io/vantreeseba/gufo-npu:<branch> smoke --iters 1000
$run ghcr.io/vantreeseba/gufo-npu:<branch> bench dispatch
```

Or without Docker: `gh run download <run-id>` the tarball, extract it, and run
`gufo-npu/bin/gufo-npu` with your host XRT on `LD_LIBRARY_PATH`.

## Registering a self-hosted NPU runner (optional)

To run `npu-smoke.yml` from GitHub, register an Actions runner on the Strix
Halo machine for this repository with labels `linux,strix-halo,npu`
(Settings → Actions → Runners → New self-hosted runner). The runner user needs
Docker access. The workflow passes the NPU and memlock flags itself.
