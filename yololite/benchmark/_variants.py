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

# Display ordering (small → large)
VARIANT_ORDER: list[str] = [
    "yololite-n", "yololite-edge-n",
    "yololite-s", "yololite-edge-s",
    "yololite-m", "yololite-edge-m",
    "yololite-l", "yololite-edge-l",
    "yololite-xl", "yololite-edge-xl",
]

VARIANT_NAMES: list[str] = list(YOLOLITE_VARIANTS.keys())
