# Owner(s): ["oncall: mobile"]

import importlib.util
import os

from torch.testing._internal.common_utils import run_tests, TestCase


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
GENERATOR_PATH = os.path.join(
    REPO_ROOT,
    "tools",
    "vulkan_cleanup",
    "generate_surface_inventory.py",
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "vulkan_cleanup_surface_inventory",
        GENERATOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVulkanCleanupInventory(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.generator = _load_generator()

    def test_generated_inventory_is_current(self):
        self.generator.check_inventory()

    def test_new_surface_requires_explicit_classification(self):
        ledger = self.generator.load_ledger()
        surfaces = self.generator.discover_surfaces()
        surfaces.append(
            {
                "id": "operator_schema:vulkan_prepack::unclassified_test_only",
                "kind": "operator_schema",
                "name": "vulkan_prepack::unclassified_test_only",
            }
        )
        with self.assertRaisesRegex(
            self.generator.InventoryError,
            "unclassified surfaces",
        ):
            self.generator.classify_surfaces(surfaces, ledger)

    def test_compatibility_state_matches_deployment_audit(self):
        ledger = self.generator.load_ledger()
        inventory = self.generator.build_inventory(ledger=ledger)
        self.assertEqual("empty", ledger["compatibility_audit"]["status"])
        self.assertEqual(0, inventory["counts"]["by_state"].get("Compatibility", 0))


if __name__ == "__main__":
    run_tests()
