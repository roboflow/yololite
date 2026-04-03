"""Canonical yololite variant definitions — single source of truth."""

# (config_subdir, yaml_file, jobs_per_gpu)
# jobs_per_gpu = floor(80 GB / estimated peak VRAM per training job) - 1.
YOLOLITE_VARIANTS: dict[str, tuple[str, str, int]] = {
    "yololite-n":       ("v2_models", "yololite_n.yaml",  4),   # ~11 GB peak
    "yololite-edge-n":  ("models",    "edge_n.yaml",      4),   # ~11 GB peak
    "yololite-s":       ("v2_models", "yololite_s.yaml",  4),   # ~14 GB peak
    "yololite-edge-s":  ("models",    "edge_s.yaml",      4),   # ~14 GB peak
    "yololite-m":       ("v2_models", "yololite_m.yaml",  4),   # ~16 GB peak
    "yololite-edge-m":  ("models",    "edge_m.yaml",      4),   # ~16 GB peak
    "yololite-l":       ("v2_models", "yololite_l.yaml",  2),   # ~30 GB peak
    "yololite-edge-l":  ("models",    "edge_l.yaml",      2),   # ~30 GB peak
    "yololite-xl":      ("models",    "yololite_xl.yaml", 1),   # ~52 GB peak
    "yololite-edge-xl": ("models",    "edge_xl.yaml",     1),   # ~52 GB peak
}

# Display ordering (edge first, then standard; small → large within each group)
VARIANT_ORDER: list[str] = [
    "yololite-edge-n", "yololite-edge-s", "yololite-edge-m", "yololite-edge-l", "yololite-edge-xl",
    "yololite-n", "yololite-s", "yololite-m", "yololite-l", "yololite-xl",
]

VARIANT_NAMES: list[str] = list(YOLOLITE_VARIANTS.keys())

# Standard vs edge groupings (small → large within each group)
STANDARD_VARIANTS: list[str] = [
    "yololite-n", "yololite-s", "yololite-m", "yololite-l", "yololite-xl",
]
EDGE_VARIANTS: list[str] = [
    "yololite-edge-n", "yololite-edge-s", "yololite-edge-m", "yololite-edge-l", "yololite-edge-xl",
]
