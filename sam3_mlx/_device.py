def is_mlx_runtime_device(device: object) -> bool:
    return device in (None, "mlx")
