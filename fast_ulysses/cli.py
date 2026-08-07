"""``fast-ulysses doctor``: can this machine run the operator, and over what links?

This exists because the answer is not obvious on the hardware range the operator targets --
sm80 through sm120, NVLink and PCIe alike. The transport writes peer windows with plain
``cudaMemcpy*Async`` into addresses from ``nvshmem_ptr``, so the one thing that decides whether a
group can be formed is whether every pair is P2P-mappable. On an NVSwitch box that is always
true and never interesting; on a two-socket PCIe box it depends on the root complex layout, on
IOMMU and ACS settings, and on the driver.

What this reports is what can be established WITHOUT a process group: the build, the devices,
and the pairwise P2P matrix. The authoritative check is ``nvshmem_ptr`` returning non-null for
every peer, which needs NVSHMEM up -- UlyssesGroup's constructor does it and names the failing
pair. A green matrix here is a necessary condition for that, not a substitute.
"""

from __future__ import annotations

import argparse
import sys


def _load():
    """(module, error). Import is deferred so `doctor` can report a load failure as its output."""
    try:
        import fast_ulysses

        return fast_ulysses, None
    except Exception as exc:  # noqa: BLE001 -- the failure IS the report
        return None, exc


def _doctor() -> int:
    pkg, err = _load()
    if err is not None:
        print(f"extension: FAILED TO LOAD\n  {type(err).__name__}: {err}")
        return 1
    print(f"extension: {pkg._C.__file__}")
    for key, value in sorted(pkg._C.build_info().items()):
        print(f"  {key}: {value}")

    import torch

    if not torch.cuda.is_available():
        print("devices: none visible to CUDA")
        return 1

    n = torch.cuda.device_count()
    built = {a.strip() for a in pkg._C.build_info()["cuda_arch_list"].split(",") if a.strip()}
    print(f"devices: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        arch = f"{p.major}{p.minor}"
        mark = "" if arch in built else "  <-- NOT IN THIS BUILD"
        print(f"  [{i}] {p.name}  sm_{arch}  {p.total_memory >> 30} GiB{mark}")

    # cudaDeviceCanAccessPeer for every ordered pair. Asymmetry is possible and worth seeing, so
    # print the full matrix rather than the upper triangle.
    print("p2p (cudaDeviceCanAccessPeer), row=from col=to:")
    print("     " + " ".join(f"{j:>3}" for j in range(n)))
    unreachable = []
    for i in range(n):
        row = []
        for j in range(n):
            ok = i == j or torch.cuda.can_device_access_peer(i, j)
            row.append(" . " if i == j else (" y " if ok else " N "))
            if not ok:
                unreachable.append((i, j))
        print(f"  {i:>2} " + " ".join(row))

    if unreachable:
        print(
            f"\nNOT P2P-mappable: {len(unreachable)} pair(s), e.g. {unreachable[:4]}.\n"
            "A UlyssesGroup spanning any of those pairs cannot be formed -- nvshmem_ptr returns\n"
            "NULL and the constructor refuses. Groups within a reachable subset still work."
        )
    else:
        print(f"\nall {n} devices are mutually P2P-mappable")
    return 1 if unreachable else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fast-ulysses")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report the build, the devices, and pairwise P2P reachability")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
