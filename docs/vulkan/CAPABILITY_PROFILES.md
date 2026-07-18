# Vulkan Capability Profiles

Capability profiles are a planner/test harness for reduced Vulkan feature
envelopes. They are not GPU emulation.

A requested profile is intersected with the live adapter capabilities before
the planner sees it. A profile can disable features, lower limits, and remove
optional ML capability bits. It must never enable unsupported hardware behavior
and must not route by GPU family or profile name.

Real hardware validation remains:

- RX 9070: primary optimization signal.
- RX 6700 XT: compatibility check.
- GTX 1080: compatibility check.

Profile tests answer: would a contract or planner route be admitted under this
reduced capability envelope on the current real device? They do not answer:
does the current device behave like another GPU family?

## Manifest

The repo-local manifest is `docs/vulkan/capability_profiles.json`.

Each row has:

- `id`: stable test selector.
- `family`: human-readable profile family.
- `kind`: `vendor_family_bucket` or `standard_floor`.
- `description`: short diagnostic explanation.
- `features`: normalized feature bits used by tests and mirrored by C++.
- `limits`: conservative compute limits used by tests and mirrored by C++.

The first manifest includes these profile IDs:

- `amd_polaris`
- `amd_vega`
- `amd_rdna1`
- `amd_rdna2`
- `amd_rdna3`
- `amd_rdna4`
- `nvidia_pascal`
- `nvidia_volta`
- `nvidia_turing`
- `nvidia_ampere`
- `nvidia_ada`
- `nvidia_blackwell`
- `vk_min_1_1_compute`
- `vk_min_1_2_compute`
- `roadmap_2022`
- `roadmap_2024`

Vendor-family names are documentation and test-selection handles only. They are
not production dispatch predicates.

## Runtime Override

`PYTORCH_VULKAN_CAPABILITY_PROFILE=<profile_id>` requests a capability mask.
The effective profile is:

1. Query the real adapter.
2. Find the requested manifest/profile row.
3. Intersect booleans with logical AND.
4. Lower API version and numeric limits with `min`.
5. Intersect bitmasks with bitwise AND.
6. Clamp subgroup ranges; invalid ranges disable subgroup-derived features.
7. Clear cooperative-matrix fields when cooperative matrix is unavailable
   after intersection.

Unknown profile IDs fail loudly.

Unset or empty `PYTORCH_VULKAN_CAPABILITY_PROFILE` preserves current behavior.

Live capability discovery also records `buffer_device_address`,
`push_descriptor`, and `descriptor_buffer`. Existing named reduced profiles
mask all three off until adapter/profile evidence promotes them. Merely
reporting an extension does not select an execution path; planner and executor
work must still add an explicit capability-gated consumer.

## Scope

This MVP covers runtime/planner capability admission and cheap route-legality
tests. It does not refactor direct adapter-query op paths and does not claim
shader correctness for absent hardware features.

Future work may add profile-aware contract admission tests per family, but those
tests must continue to route from normalized feature bits rather than profile
names.
