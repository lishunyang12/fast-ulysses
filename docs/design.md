# Design

The group owns symmetric output allocations. Rendezvous exposes the base of every rank's allocation
to every other rank. Equal splits make addressing a closed-form calculation, so no plan objects or
cache are needed. Each exchange issues at most `B * world_size` pitched peer copies.

Native-atomic fabrics use two device flag barriers around the copies. PCIe peer writes are retained,
but the unsafe system-scope spin barrier is not: the Python wrapper uses blocking process-group
barriers and synchronizes the selected stream. This makes PCIe a correctness path rather than an
overlap path.
