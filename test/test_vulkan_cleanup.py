# Owner(s): ["oncall: mobile"]

import copy
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
SYNC_COUNTERS_PATH = os.path.join(
    REPO_ROOT,
    "aten",
    "src",
    "ATen",
    "native",
    "vulkan",
    "api",
    "SyncCounters.cpp",
)
SYNC_PATH = os.path.join(
    REPO_ROOT,
    "aten",
    "src",
    "ATen",
    "native",
    "vulkan",
    "api",
    "Sync.cpp",
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

    def test_deleted_scope_decision_rejects_restored_symbol(self):
        ledger = copy.deepcopy(self.generator.load_ledger())
        decision = next(
            item
            for item in ledger["scope_decisions"]
            if item.get("status") == "deleted"
        )
        decision["forbidden_code_symbols"] = ["VulkanRuntimePolicy"]
        with self.assertRaisesRegex(
            self.generator.InventoryError,
            "restored symbols",
        ):
            self.generator.validate_scope_decisions(ledger)

    def test_sync_counter_substrate_is_separate_from_stack_control_plane(self):
        with open(SYNC_COUNTERS_PATH, encoding="utf-8") as file:
            counter_source = file.read()
        with open(SYNC_PATH, encoding="utf-8") as file:
            sync_source = file.read()

        definitions = (
            "VulkanSyncCounters& vulkan_sync_counters()",
            "vulkan_graph_program_invocation_counters() {",
            "void note_vulkan_queue_submit(VulkanSubmitOrigin origin)",
            "std::vector<std::string> submit_origin_phase_snapshot()",
            "std::vector<int64_t> retire_drain_counters_snapshot()",
            "std::vector<std::string> retire_call_site_counters_snapshot()",
            "void note_vulkan_forced_sync(VulkanForcedSyncReason reason)",
        )
        for definition in definitions:
            self.assertIn(definition, counter_source)
            self.assertNotIn(definition, sync_source)

        for evidence_path in (
            "FallbackPolicyReadback",
            "ProfilingTimestampReset",
            "ProfilingTimestampReadback",
            "StackPlannedRecordingSubmit",
        ):
            self.assertIn(evidence_path, counter_source)

        self.assertIn("stack_internal_temp_retire_batch_counters()", sync_source)
        self.assertNotIn(
            "stack_internal_temp_retire_batch_counters()",
            counter_source,
        )


if __name__ == "__main__":
    run_tests()
