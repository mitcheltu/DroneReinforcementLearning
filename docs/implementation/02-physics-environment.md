# 02 — Physics, controller and environment

## 1. Fixed constants

These constants define simulator `quadrotor-rk4-v1` and controller `rate-p-v1`.

| Constant | Value |
|---|---:|
| Mass m | 0.8 kg |
| Gravity g | 9.81 m/s² |
| Inertia diagonal I | [0.005,0.005,0.009] kg m² |
| Center-to-rotor distance l | 0.20 m |
| Rotor maximum thrust | 4.0 N each |
| Rotor response time tau | 0.030 s |
| World linear drag coefficient | 0.08 kg/s, isotropic |
| Reaction torque / thrust coefficient kappa | 0.016 m |
| Rate gains K | [0.08,0.08,0.04] N m per rad/s |
| Drone collision sphere radius rho | 0.30 m |
| Additional gate-pass clearance | 0.05 m |
| Physics dt | 1/120 s |
| Control dt | 1/60 s |
| Numerical geometry epsilon | 1e-9 m |

The collision sphere is a conservative vehicle envelope, not detailed propeller geometry. Visualization must fit the vehicle inside this envelope; propeller disk radius <=0.075 m at arm radius 0.20 m satisfies it. No wind, rotor wake, ground effect, angular drag, battery or aerodynamic lift is modeled.

## 2. Rotor ordering and mixer

Let b=l/sqrt(2). Rotor indices in stored arrays are 0 through 3:

| Index | Body position (x,y,z) | Reaction torque sign about +Z |
|---|---|---:|
| 0 | (+b,+b,0) | +1 |
| 1 | (-b,+b,0) | -1 |
| 2 | (-b,-b,0) | +1 |
| 3 | (+b,-b,0) | -1 |

For actual thrust vector f, collective is sum(f), and body torque is:

`tx = b*(f0+f1-f2-f3)`

`ty = b*(-f0+f1+f2-f3)`

`tz = kappa*(f0-f1+f2-f3)`.

Reaction sign is the torque on the body. Rendered rotor spin, if animated, is opposite the body reaction sign. The simulator does not integrate rotor spin angle.

At each physics tick, compute:

`tau_des = K * (omega_des - omega) + cross(omega, I*omega)`.

Multiplication by K and I above is componentwise diagonal multiplication. There is no integral or derivative state. The gyroscopic term is feedforward compensation. This is a proportional rate controller, not a PID controller; documentation and UI must name it correctly.

Find zero-sum differential thrust delta by the fixed inverse mixer:

`delta0 = tx/(4*b) - ty/(4*b) + tz/(4*kappa)`

`delta1 = tx/(4*b) + ty/(4*b) - tz/(4*kappa)`

`delta2 = -tx/(4*b) + ty/(4*b) + tz/(4*kappa)`

`delta3 = -tx/(4*b) - ty/(4*b) - tz/(4*kappa)`.

Set c=F_des/4. Set lambda=1, then for each positive delta_i reduce lambda to min(lambda,(4-c)/delta_i); for each negative delta_i reduce to min(lambda,c/(-delta_i)). Zero delta contributes no constraint. Clamp lambda to [0,1]. Set motor command `f_cmd_i=clip(c+lambda*delta_i,0,4)`.

This preserves requested collective and scales all torque contributions uniformly when necessary. At zero or maximum collective, maneuvering torque authority can vanish. Log lambda and saturation; do not hide this actuator limitation.

## 3. Derivatives and RK4

State order: `[px,py,pz,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz,f0,f1,f2,f3]`.

`p_dot = v`

`v_dot = [0,0,-g] + R(q)*[0,0,sum(f)]/m - 0.08*v/m`

`omega_dot = inverse(I)*(torque(f)-cross(omega,I*omega))`

`q_dot = 0.5 * hamilton_product(q,[0,wx,wy,wz])`

`f_dot = (f_cmd-f)/0.030`.

All dynamics arithmetic is float64 in Python and JavaScript Number in TypeScript. Rotation-matrix evaluation uses the normalized intermediate quaternion. The q derivative uses the intermediate quaternion itself. Invalid intermediate norm <1e-12 is a numerical error.

Use classical coupled RK4: k1=D(x), k2=D(x+dt*k1/2), k3=D(x+dt*k2/2), k4=D(x+dt*k3), x_next=x+dt*(k1+2*k2+2*k3+k4)/6. Hold f_cmd fixed throughout all four evaluations of one physics tick. Recompute the rate controller only at the next physics tick. Normalize the final quaternion. Clamp final motor thrust to [0,4] only for roundoff <=1e-10; a larger violation is an error, not normal saturation.

Detect nonfinite values after each tick. A numerical error aborts the current training invocation, retains the last valid state and error diagnostics, and does not turn into an ordinary collision reward. In the browser it pauses with an error. A finite angular-speed norm above 100 rad/s or linear-speed norm above 100 m/s is `state_limit` task failure; these are protective operating bounds, not substitute integration checks.

## 4. Gate geometry

Gate local coordinates are defined in document 01. Clear aperture is local y in (-1.25,1.25), z in (-1.25,1.25). Frame local boxes are:

- Left/right: x in [-0.05,0.05], y in [-1.35,-1.25] / [1.25,1.35], z in [-1.35,1.35].
- Bottom/top: x in [-0.05,0.05], y in [-1.25,1.25], z in [-1.35,-1.25] / [1.25,1.35].

All gate frames are collision-active throughout an attempt, including already-passed and out-of-order gates.

## 5. Swept collision and gate events

Each physics tick defines a straight swept center segment between its integrated endpoint positions. Version 1 uses conservative sphere-versus-box collision by expanding each gate box on all three axes by rho and intersecting the center segment with that expanded box using a slab test. This intentionally overestimates rounded corner contact. Python and TypeScript must use the same approximation. Do not claim exact sphere/box continuous collision geometry.

For each axis, handle a segment direction magnitude <1e-12 as parallel: reject intersection if its starting coordinate is outside the slab, otherwise leave the entry/exit interval unchanged. Intersect all parameter intervals with [0,1]. Contact counts as collision, including equality within geometry epsilon. A starting point inside any expanded box collides at fraction zero.

Workspace center bounds for the drone are x,y in [-60+rho,60-rho], z in [rho,20-rho]. Crossing/touching the lower z face is `ground_collision`; crossing/touching any other face is `workspace_exit`. Compute earliest swept intersection with these faces.

For current target gate only, let signed plane distances be d0 and d1. A forward crossing candidate has d0<0 and d1>=0. Crossing fraction alpha=-d0/(d1-d0). Project the interpolated crossing center into gate coordinates. It is a valid pass when abs(local_y)<=1.25-rho-0.05 and abs(local_z)<=1.25-rho-0.05. Thus effective pass half-width/height is 0.90 m. A center passing outside this smaller opening without touching a frame is a miss and may recover. Backward crossing does not count. A segment starting exactly on the plane does not count; the drone must approach from negative distance on a subsequent segment.

Find all collisions and valid target crossings in temporal order within a tick. Collision at or before a pass (fractions equal within 1e-9) wins. If a valid nonfinal pass happens before a later collision, credit the pass, advance target, then terminate on collision. Continue testing the remaining segment against the new target to make event ordering explicit even though supported gate spacing makes multiple passes per tick unlikely. A valid final pass before a later collision ends the episode at that pass; geometry after finishing is not simulated.

At a terminal event fraction, freeze position, velocities and motor states by linear interpolation between tick endpoints, and attitude by shortest-arc SLERP. This is the defined event-state approximation; do not rerun RK4 for an event fraction in only one runtime. The previous stored endpoint remains unchanged, and the terminal record uses its fractional timestamp. Nonterminal gate events do not insert an extra physics integration step.

Event types: `gate_pass`, `gate_miss_forward`, `gate_cross_backward`, `collision`, `success`, `timeout`, `state_limit`, `user_abort`. Miss/backward events are emitted for the current gate only and carry no separate penalty. Progression never skips a gate. A later gate crossed early is ignored for scoring but its frame remains solid.

## 6. Environment API

Implement a Gymnasium Env with `reset(seed=None,options=None) -> (obs,info)` and `step(action) -> (obs,reward,terminated,truncated,info)`. reset calls `super().reset(seed=seed)`. A reset without a seed advances the existing RNG; it must not reseed with a fixed value.

`options` accepts either a validated explicit CourseV1/initial state or a curriculum task descriptor, never both. Step after terminal requires explicit reset and raises an error. A single environment does not auto-reset itself. SB3 vector wrapping performs autoreset above the recorder.

Step order:

1. Snapshot previous state, target index and distances needed by reward.
2. Validate action; clamp/map it and record the applied command.
3. Execute up to two physics ticks, recomputing the low-level controller each tick.
4. Detect/process chronological events after each tick; stop on terminal.
5. Update elapsed simulation time and previous_action.
6. Compute reward from this transition and events.
7. Build next observation, including new target and remaining time.
8. Record final transition/state before returning to any autoreset wrapper.

Task budgets: hover 5 s, one gate 12 s, three gates 20 s, ten gates 45 s. These are part of the task and feature 39 exposes the remaining budget. Reaching a task budget without success is `terminated=True`, `truncated=False`, `reason=timeout`, `is_success=False`. Use integer tick deadlines (600/1440/2400/5400 respectively), not a floating accumulated-time comparison. Same-time collision beats a pass; a same-time valid final pass beats timeout. This deliberate finite-horizon task definition differs from an external Gymnasium TimeLimit wrapper. Do not add that wrapper. External collector interruption is not an episode outcome; retain `incomplete` metadata and reset on continuation.

Success, collision, workspace exit and state_limit also set terminated=True. Normal transitions set both flags False. Truncated is reserved for a future external interruption API; it is not emitted by ordinary v1 step(). Hover success occurs at 5 seconds only if the hover quality limits below held throughout; otherwise it terminates earlier with `hover_departure`.

Info always contains episode_id, task_kind, stage, gates_passed, target_gate_index, elapsed_seconds, is_success, terminal_reason (null until terminal), reward_components, events and state timestamp. Do not attach megabytes of trajectory data to info; the recorder owns trajectories.

## 7. Racing reward

The selected baseline deliberately omits a repeated alignment reward. Time and gate completion provide the speed incentive; flight without forward course progress must not earn alignment points.

Per policy transition:

`reward = progress + gate_bonus + finish_bonus - time_cost - smoothness_cost - failure_cost`.

- progress coefficient: 0.5 reward/m.
- gate bonus: +10 per valid passage.
- final finish bonus: +50 in addition to gate 10 bonus.
- time cost: 0.20 * actual transition seconds in reliability mode, 0.50 * actual seconds in speed mode.
- smoothness: 0.02*sum((u-u_previous)^2), once per policy transition, including a partial terminal interval.
- failure cost: 50 for collision, workspace_exit, state_limit or timeout.

Distance progress must be segmented at gate-pass events. For each portion of the center path while target j is active, add `0.5*(distance(segment_start,center_j)-distance(segment_end,center_j))`. On a gate change, start the next portion against the new gate; never subtract distances to different gate centers. If the final gate completes, there is no later progress portion. Distances are Euclidean in meters. Segment endpoints are event interpolations from section 5. Retain each reward component separately and their exact sum.

User abort is excluded from training and evaluation outcomes; the browser marks it aborted rather than failed. No success is possible merely by surviving a timeout. Do not assume a penalty alone mathematically prevents all reward exploits: specific exploit regression tests and evaluation completion metrics are required.

## 8. Hover warm-up reward

Hover target is the initial position before reset perturbation. Desired heading is the unperturbed start yaw. Hover observation uses the target convention in document 01. For actual transition duration h:

`reward = h*(1 - 0.5*||p-target||² - 0.1*||v||² - 0.1*(1-R(q)[2,2]) - 0.02*||omega||²) - 0.02*||u-u_previous||²`.

Use end-of-transition state for these terms. Departure is position error >1 m or body +Z tilt >30 degrees; it ends with an additional -5. Other physical failure ends with -5. Surviving 5 s within these limits is hover success and adds +5. Hover has no racing gate/finish bonuses.

## 9. Reference controller

Before PPO training, implement a deterministic position/velocity reference controller for hover and straight gate courses. Its purpose is to validate signs, coordinate transforms, thrust authority and collision handling. It uses the same body-rate/thrust actuator interface and does not establish that every supported course is dynamically feasible. PPO release evaluation must not fall back to it on difficult tracks.

Reference controller `position-reference-v1`, evaluated at 60 Hz:

1. Hover target is the hover target point. Racing target is current gate center +1.0 m*gate normal, deliberately beyond its plane. Desired heading is initial heading for hover or current gate yaw for racing.
2. Set v_des=1.5*(target-p), then limit its Euclidean norm to 3 m/s. Set a_des=2*(v_des-v), then limit its Euclidean norm to 4 m/s².
3. Set force vector h=m*(a_des+[0,0,g]); desired body z is normalize(h). Desired heading vector c=[cos(yaw),sin(yaw),0]. Set desired y=normalize(cross(z_des,c)) and desired x=cross(y_des,z_des). With the acceleration limit, h has positive vertical component so this construction is not singular. R_des has these axes as columns.
4. Define E=0.5*(R_des^T*R-R^T*R_des); vee(E)=[E[2,1],E[0,2],E[1,0]]. Set desired body rates=-4*vee(E), component-clipped to [6,6,3] rad/s limits.
5. Desired collective=clip(dot(h,R*[0,0,1]),0,16). Map back to normalized action `[2*F/16-1,wx_des/6,wy_des/6,wz_des/3]` and use the ordinary environment step.

Record all constants above in a separate reference-controller config. Test hover and straight-course cases only for its milestone acceptance; this simple controller is not a trajectory planner or a guaranteed recovery controller.

## 10. Quaternion helper formulas

For q=(w,x,y,z) and r=(a,b,c,d), Hamilton product is `[w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c, w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a]`. Inverse of a unit quaternion is `[w,-x,-y,-z]`.

For normalized q, the body-to-world rotation matrix rows are:

`[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]`

`[2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]`

`[2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]`.

Shortest-arc SLERP first negates the second quaternion if dot<0. Clamp dot to [0,1] after that. If dot>0.9995, normalized linear interpolation is used. Otherwise theta=acos(dot) and output is `(sin((1-alpha)*theta)*q0+sin(alpha*theta)*q1)/sin(theta)`, normalized for roundoff. Alpha is clamped to [0,1]. Tests cover equivalent opposite-sign quaternions and rotations close to 180 degrees. Never interpolate Euler angles for drone attitude.
