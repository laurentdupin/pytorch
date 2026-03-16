import torch
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorConverter
from torch.fx.experimental.symbolic_shapes import ShapeEnv

shape_env = ShapeEnv()
converter = FakeTensorConverter()

torch._C._create_and_enter_fake_tensor_mode(converter, shape_env)
try:
    # --- Concrete shapes ---
    real_a = torch.randn(3, 4)
    real_b = torch.randn(3, 4)
    fake_a = torch._C._make_fake_tensor(real_a)
    fake_b = torch._C._make_fake_tensor(real_b)
    c = fake_a + fake_b
    print(f"c.shape = {c.shape}")
    print(f"is_fake = {torch._C._is_fake_tensor(c)}")

    # --- Dynamic shapes ---
    real_x = torch.randn(5, 8)
    real_y = torch.randn(5, 8)
    fake_x = torch._C._make_fake_tensor(real_x, source=LocalSource("x"))
    fake_y = torch._C._make_fake_tensor(real_y, source=LocalSource("y"))

    print(f"\nfake_x.shape = {fake_x.shape}")
    print(f"fake_y.shape = {fake_y.shape}")
    print(f"fake_x.shape[0].node = {fake_x.shape[0].node}")

    s0 = fake_x.shape[0]
    s1 = fake_x.shape[1]
    print(f"s0 + s1 = {s0 + s1}")
    print(f"s0 * 2 = {s0 * 2}")
    print(f"s0 == fake_y.shape[0]: {s0 == fake_y.shape[0]}")
finally:
    torch._C._exit_fake_tensor_mode()
