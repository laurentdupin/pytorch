from torch._inductor import config


if config.is_fbcode():
    import torch._inductor.fb.tlx_templates.registry  # noqa: F401  # type: ignore[import-not-used]
<<<<<<< HEAD

# TODO. Move the registry to this file once the TLX template is more complete.
=======
>>>>>>> b0f830d929c (Revert "Support kernels with opaque types (#174211)")
