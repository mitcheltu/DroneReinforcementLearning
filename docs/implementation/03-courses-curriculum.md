# 03 — Courses, randomization and curriculum

## 1. Supported envelope

All standard courses have exactly ten ordered, upright gates with geometry from document 02. Bounds include both endpoints unless a strict comparison is explicitly stated.

| Property | Supported limit |
|---|---|
| Workspace | x,y in [-60,60] m; z in [0,20] m |
| Gate center height | 3–10 m |
| Consecutive horizontal center spacing | 6–8 m |
| Consecutive height change | abs(delta_z)<=1.5 m |
| Horizontal heading change between segments | <=30 degrees |
| Gate-normal heading offset from incoming segment | <=10 degrees |
| Gate-normal heading offset from outgoing segment, except last | <=40 degrees |
| Relative segment heading from first segment | <=60 degrees |
| Gate frame distance from workspace boundary | >=0.5 m |
| Pairwise gate frame separation | conservative enclosing-box separation >=0.6 m |
| Start-to-first horizontal distance | 6–8 m |
| Start-to-first height difference | <=0.25 m |
| Start/other-frame collision | forbidden |
| Episode limit | 45 s |

The first segment is start reference point to gate 1; later segments are consecutive gate centers. The start reference point is the unperturbed start center, stored with generation metadata and as `start_reference_m` in the course schema. Editor changes preserve it. Heading comparisons wrap differences to [-pi,pi). Segment lengths use horizontal distance; explicit height constraints apply separately.

For separation, compute each rotated gate's full world-axis bounding box. Expand each box by 0.30 m. Two expanded closed AABBs may not intersect (touching fails). This is a conservative and deterministic filter, not exact oriented frame geometry. Use exact frame boxes for runtime collisions, with the conservative sphere expansion in document 02.

Supported mode also rejects crossing nonadjacent course centerline segments in XY when their interpolated heights differ by less than 3 m at the intersection. Parallel/collinear overlapping projections are rejected if their segment height ranges expanded by 1.5 m overlap. Adjacent segments share endpoints and are exempt. This reduces confusing intersections, but is not an obstacle-avoidance guarantee.

The supported envelope is an engineering constraint set. Held-out test results determine which performance claims are warranted. Passing geometric validation is not a proof that a policy or vehicle can complete a course.

## 2. Random generator `course-generator-v1`

Use NumPy Generator(PCG64) with SeedSequence-derived independent streams. All uniform draws are continuous over the stated interval (NumPy's upper endpoint exclusion is accepted); integer draws use explicit exclusive upper bounds. Store the exact course, not just its seed.

At reset, draw one unsigned 64-bit course seed from that environment's course stream and one unsigned 64-bit reset seed from its reset stream. Construct temporary PCG64 generators from these integers for proposal geometry and reset perturbations respectively. Store both integers as decimal strings to avoid JavaScript integer precision loss. Persist the parent stream states at checkpoints. UUIDv5 project namespace is `ca2d61ea-1dc2-4bd4-894e-d6a8a29e41f1`; course name input is `version:stage:course_seed:reset_seed`, and gate name input appends `:gate:index`. Episode IDs are UUIDv5 of `run_id:env_index:episode_counter` under the same namespace.

For a stage with gate count N, horizontal turn bound T, height change H, orientation jitter J and reset perturbations from section 4:

1. Draw initial reference height uniformly in [4,6] m. Set local reference point p0=[0,0,h0] and heading theta0=0.
2. For j=1..N, draw horizontal length L_j uniformly in [6,8] m. For gate 1, theta_1=0 and dz_1 uniform [-min(H,0.25),min(H,0.25)]. For j>1, draw delta_theta uniformly [-T,T], set theta_j=theta_(j-1)+delta_theta; reject the entire proposal if abs(theta_j)>60 degrees. Draw dz_j uniformly [-H,H].
3. Set p_j=p_(j-1)+[L_j*cos(theta_j),L_j*sin(theta_j),dz_j]. Reject proposal if any gate height leaves [3,10]. Do not clip heights; clipping would change the distribution.
4. Gate yaw_j=theta_j+uniform(-J,J). All pitch/roll are zero.
5. Draw a global yaw uniformly [-pi,pi) and rotate all horizontal coordinates and yaws, including start heading. Compute the world XY bounding box of all gate frames plus start sphere. Recenter its horizontal midpoint at [0,0].
6. Determine allowable XY translation intervals that keep this full bounding box inside [-59.5,59.5] on both axes. If either interval is empty, reject. Sample one translation independently and uniformly from each allowable interval. Apply it to every point. There is no global vertical translation after height sampling.
7. Validate all section 1 constraints with the appropriate stage bounds. Reject on failure. Limit proposals to 512 per course; if exhausted, raise `course_generation_exhausted` with seed/stage, without silently substituting a straight track.
8. Generate initial state from section 3. If a reset perturbation collides or lies outside the workspace, resample the perturbation up to 64 times; exhaustion rejects the course proposal and continues within the 512 total proposal limit.
9. Assign deterministic IDs derived from the course seed and gate index (UUIDv5 with a fixed project namespace stored in generator config). Store proposal count, seed, stage, unperturbed start reference, generated geometry and all initial state values.

Gate positions are sampled once at episode reset. The geometry remains fixed throughout that episode. Each vector environment has its own stream; simultaneous resets do not receive identical tracks. A curriculum promotion affects tasks sampled at later resets, not episodes already in progress.

The generated distribution is the result of the above sampling followed by rejection. It is not uniform over all geometrically valid tracks. Record rejection reasons and acceptance fraction so excessive rejection is visible.

## 3. Initial state

Yaw is first segment heading plus the stage yaw perturbation. Roll and pitch perturbations are sampled independently and composed as `Rz(yaw)*Ry(pitch)*Rx(roll)`, then converted to wxyz quaternion. Position offset components are sampled independently from their stage ranges in the heading-aligned start frame and transformed to world. Initial velocity components are independently sampled in that same frame and transformed to world. Omega components are independently sampled in body coordinates. All motors start at mg/4. Previous action is hover action.

Position perturbations do not alter `start_reference_m`. For a standard user-edited course, Restart uses the stored initial state exactly. Editing gate 1 or reordering gates recomputes the initial yaw toward gate 1 with zero roll/pitch, zero velocity/omega and hover motors at the existing start reference point. Users cannot drag the start reference in v1. If start-to-gate constraints fail, Supported mode disallows Run until repaired.

## 4. Curriculum stages

Angles below are degrees in human-readable documentation and radians in stored configuration. `pos` is per-axis range ±m; `vel` is per-axis ±m/s; `rate` is per-axis ±rad/s. Roll/pitch and yaw are independent uniform ±angles.

| Stage | Task / N | T | H | J | pos | vel | roll/pitch | yaw | rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | hover / 0 | 0 | 0 | 0 | 0.10 | 0.10 | 3 | 5 | 0.10 |
| 1 | one gate / 1 | 0 | 0 | 0 | 0.10 | 0.10 | 3 | 5 | 0.10 |
| 2 | one gate / 1 | 0 | 0.25 | 5 | 0.25 | 0.25 | 5 | 10 | 0.20 |
| 3 | three straight gates / 3 | 0 | 0 | 0 | 0.25 | 0.25 | 5 | 10 | 0.20 |
| 4 | three turning gates / 3 | 15 | 0.50 | 5 | 0.25 | 0.25 | 5 | 10 | 0.20 |
| 5 | ten moderate gates / 10 | 20 | 0.75 | 5 | 0.25 | 0.25 | 5 | 10 | 0.20 |
| 6 | ten full-envelope gates / 10 | 30 | 1.50 | 10 | 0.25 | 0.25 | 5 | 10 | 0.20 |

For hover, sample target XY uniformly in [-10,10] m and height [4,6] m, random yaw [-pi,pi), then apply stage-0 state perturbations. There are no physical gates. Curriculum courses are valid only under their declared stage; ordinary editor validation uses full standard constraints.

At frontier stage f, task sampling is 80% stage f and 20% stage max(0,f-1); at f=0 use 100% stage 0. Do not mix all earlier stages, and do not silently lower difficulty after failure. Reward mode is reliability until the speed condition below.

Every 131,072 total training transitions, run the current frontier on its fixed 64 validation cases with deterministic actions. Promote one stage after two consecutive evaluations each achieve at least 61/64 successes and the frontier has received at least 131,072 transitions since activation. Both conditions are required. Reset the consecutive-pass counter on failure and promotion. Log the transition at promotion. Promotion is applied at the next reset in each environment.

After stage 6 achieves at least 95% completion on the 512-case development suite twice consecutively, switch time coefficient from 0.20 to 0.50 and begin speed refinement. Keep the same stage sampling. Preserve the best reliability checkpoint before switching. If development completion falls below 95%, select the better reliability candidate for release; do not disguise speed improvement as success when reliability degrades.

## 5. Seed partitions and case sets

Three training runs, if compute permits the final reproducibility gate: root seeds 101, 202 and 303. Each run constructs a SeedSequence from `[root_seed,stream_kind,env_index]`; stream kinds: 0 course generation, 1 reset perturbation, 2 task-stage selection, 3 recording priority. Recording must not consume training RNG draws.

Use separate immutable validation artifacts generated before training:

- Stage promotion: 64 courses/states per stage, seed namespace 10001 + stage.
- Development/model selection: 512 stage-6 cases, namespace 20001.
- Final test: 1,000 stage-6 cases, namespace 30001.
- Stress: 100 experimental cases, namespace 40001, explicitly labeled outside distribution and excluded from standard completion percentage.

These are namespaces passed into SeedSequence, not overlapping consecutive training seeds. Save full geometry/initial state and hashes. Check exact duplicate course+initial-state hashes across partitions and reject duplicates. Training does not load final-test cases. Once final-test results inform design changes, retire that final-test version and generate a new untouched version with a new recorded namespace before making a fresh held-out claim.

## 6. Editor validation and experimental mode

Validation returns all failures with stable codes, affected gate IDs and a concise corrective message. Required codes: gate_count, nonfinite, duplicate_id, invalid_label, unsupported_geometry, gate_height, frame_outside_workspace, spacing, altitude_change, turn_angle, heading_drift, approach_angle, outgoing_angle, frame_overlap, route_intersection, invalid_start.

Experimental mode relaxes spacing, altitude-change, turn, heading-drift, approach/outgoing-angle and route-intersection bounds. It still requires ten upright fixed-size gates, finite valid schema, no overlapping frames, frame containment and a collision-free start inside the workspace. Gate center height can use any value that keeps the full frame inside the workspace with 0.5 m margin. The UI displays which supported checks fail, and uses the same 45 s budget. Experimental results must never be mixed into supported benchmarks.
