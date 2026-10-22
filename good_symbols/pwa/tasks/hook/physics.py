"""Hook: finite-volume Pymunk dynamics and the factored contact-mode interface.

A planar gripper holds an L-shaped hook and must reach around a square block and pull it to the middle wall.
The mode ``z`` (16 binary factors) selects grasp state, contact regime, the hook surface and block face in
contact (or, while separated, the free-space route region), and whether the block touches the wall.
"""
from __future__ import annotations

import numpy as np
import pymunk


# The local discrete contact interface, shared by mode inference and the MILP.
Z_NAMES = (
    "gripper_released", "gripper_held",
    "contact_separate", "contact_active",
    "hook_normal_x", "hook_normal_y",
    "hook_normal_negative", "hook_normal_positive",
    "hook_patch_primary", "hook_patch_secondary",
    "block_normal_x", "block_normal_y",
    "block_normal_negative", "block_normal_positive",
    "wall_clear", "wall_swept_contact",
)
Z_DIM = len(Z_NAMES)

HOOK_SURFACES = {
    "arm_top": (5, 7, 8),
    "arm_end": (4, 6, 8),
    "arm_bottom": (5, 6, 8),
    "handle_left": (4, 6, 9),
    "handle_end": (5, 6, 9),
    "handle_right": (4, 7, 8),
}
BLOCK_FACES = {
    "left": (10, 12), "right": (10, 13),
    "bottom": (11, 12), "top": (11, 13),
}
CONTACT_PAIRS = {
    "arm_top": "bottom", "arm_end": "right", "arm_bottom": "top",
    "handle_left": "right", "handle_end": "top", "handle_right": "left",
}


def local_equalities() -> tuple[np.ndarray, np.ndarray]:
    """Return the TU factor-flow equalities ``E^d z = e^d``."""
    E = np.zeros((8, Z_DIM), dtype=int)
    E[0, [0, 1]] = 1
    E[1, [2, 3]] = 1
    E[2, [4, 5]] = 1
    E[3, [4, 5]], E[3, [6, 7]] = -1, 1
    E[4, [6, 7]], E[4, [8, 9]] = -1, 1
    E[5, [8, 9]], E[5, [10, 11]] = -1, 1
    E[6, [10, 11]], E[6, [12, 13]] = -1, 1
    E[7, [14, 15]] = 1
    return E, np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=int)


def local_inequalities() -> tuple[np.ndarray, np.ndarray]:
    """Return physical compatibility constraints ``P^d z <= p^d``."""
    P = np.zeros((9, Z_DIM), dtype=int)
    P[0, 3], P[0, 1] = 1, -1
    P[1, 9], P[1, 6] = 1, -1
    P[2, 4], P[2, 10] = 1, -1
    P[3, 10], P[3, 4] = 1, -1
    P[4, 6], P[4, 13] = 1, -1
    P[5, 13], P[5, 6] = 1, -1
    P[6, 3], P[6, 5] = 1, -1
    P[7, 3], P[7, 6] = 1, -1
    P[8, 3], P[8, 8] = 1, -1
    return P, np.zeros(len(P), dtype=int)


def mode_vector(
    *, held: bool, hook_surface: str, block_face: str,
    contact_active: bool = False, wall_swept_contact: bool = False,
) -> np.ndarray:
    z = np.zeros(Z_DIM, dtype=int)
    z[1 if held else 0] = z[3 if contact_active else 2] = 1
    z[15 if wall_swept_contact else 14] = 1
    if CONTACT_PAIRS.get(hook_surface) != block_face:
        raise ValueError(f"incompatible contact pair: {hook_surface}, {block_face}")
    z[list(HOOK_SURFACES[hook_surface])] = 1
    z[list(BLOCK_FACES[block_face])] = 1
    return z


def validate_mode(z: np.ndarray) -> bool:
    E, e = local_equalities()
    P, p = local_inequalities()
    z = np.asarray(z, dtype=int)
    return bool(
        z.shape == (Z_DIM,)
        and np.all(np.isin(z, (0, 1)))
        and np.array_equal(E @ z, e)
        and np.all(P @ z <= p)
    )


# s = gripper(4), Hook(6), block(6).  Observation appends z and a validity bit.
S_DIM = 16
OBS_DIM = S_DIM + Z_DIM + 1
G_POS = slice(0, 2)
G_TH, G_GAP = 2, 3
H_POS = slice(4, 6)
H_TH, H_VX, H_VY, H_OM = 6, 7, 8, 9
B_POS = slice(10, 12)
B_TH, B_VX, B_VY, B_OM = 12, 13, 14, 15

WORLD_X = (0.0, 3.5)
WORLD_Y = (0.0, 3.5)
WALL_CENTER_Y = 1.75
WALL_THICKNESS = 0.05
WALL_Y = WALL_CENTER_Y + WALL_THICKNESS / 2  # upper surface reached by block

# Exact DynPushPullHook2D geometry, with the block-size interval fixed at its
# midpoint.  The rigid L is rotated so its long side is the gripper handle.
GRIPPER_BASE_WIDTH = 0.06
GRIPPER_BASE_HEIGHT = 0.32
HOOK_THICKNESS = GRIPPER_BASE_HEIGHT / 3
HOOK_HANDLE = 3.5 * 2 / 5
HOOK_LONG = 3.5 / 6
BLOCK_SIDE_MIN = GRIPPER_BASE_HEIGHT / 2
BLOCK_SIDE_MAX = HOOK_LONG - HOOK_THICKNESS
BLOCK_SIDE = 0.5 * (BLOCK_SIDE_MIN + BLOCK_SIDE_MAX)
BLOCK_HALF = BLOCK_SIDE / 2
BLOCK_MASS = 1.0
BLOCK_INERTIA = BLOCK_MASS * BLOCK_SIDE**2 / 6

# theta_h=0: a long vertical handle ending at H_POS and a short top arm.
HOOK_MASS = 1.0
FINGER_LENGTH = 0.20
FINGER_WIDTH = 0.06
GRIP_CLOSED = HOOK_THICKNESS + FINGER_WIDTH
GRIP_OPEN = GRIPPER_BASE_HEIGHT

CONTROL_DT = 0.25
PHYSICS_SUBSTEPS = 12
SPACE_DAMPING = 0.01
VELOCITY_DAMPING = SPACE_DAMPING**CONTROL_DT
FRICTION_MU = 0.35
SHAPE_FRICTION = float(np.sqrt(FRICTION_MU))
CONTACT_TOL = 0.012
CONTACT_PATCH_MARGIN = 0.30
HOOK_CONTACT_CHART_ANGLE = 0.004
BLOCK_CONTACT_CHART_ANGLE = 0.08
BLOCK_RESET_ANGLE = 0.06
PENETRATION_TOL = 0.012
SUCCESS_TOL = 0.020
MAX_DXY = 0.10
MAX_DTHETA = np.pi / 12
MAX_DGAP = 0.02
ROUTE_CLEAR_REL_Y = BLOCK_HALF - (HOOK_HANDLE - HOOK_THICKNESS / 2) + 0.06
ROUTE_LOW_CHART_REL_Y = -(HOOK_HANDLE + HOOK_THICKNESS / 2 + BLOCK_HALF + 0.04)
ROUTE_RIGHT_REL_X = HOOK_LONG + BLOCK_HALF + 0.04
ROUTE_HANDLE_REL_X = BLOCK_HALF + HOOK_THICKNESS / 2 + 0.04
ROUTE_CONTACT_REL_X = 0.5 * (HOOK_LONG + HOOK_THICKNESS / 2)
ROUTE_ALIGN_SWITCH_X = ROUTE_CONTACT_REL_X + 0.08
ROTATE_RIGHT_REL_X = 0.90
ROTATE_Y_SLOPE = 0.20
ROTATE_Y_BAND = 0.08
ROUTE_CHART_ANGLE = 0.08
WALL_SWEEP_CENTER_MARGIN = MAX_DXY + CONTACT_TOL + BLOCK_HALF * BLOCK_CONTACT_CHART_ANGLE

BLOCK_CATEGORY, HOOK_CATEGORY = 1, 2
MIDDLE_WALL_CATEGORY, BOUNDARY_CATEGORY = 4, 8


def wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2 * np.pi) - np.pi)


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def transform(points: np.ndarray, position: np.ndarray, theta: float) -> np.ndarray:
    return np.asarray(points, float) @ rotation(theta).T + np.asarray(position, float)


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def block_vertices(state: np.ndarray) -> np.ndarray:
    local = BLOCK_HALF * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    return transform(local, state[B_POS], float(state[B_TH]))


def hook_local_polygons() -> tuple[np.ndarray, np.ndarray]:
    t, lv, lh = HOOK_THICKNESS, HOOK_HANDLE, HOOK_LONG
    vertical = np.array([[-t / 2, 0], [t / 2, 0], [t / 2, lv], [-t / 2, lv]])
    horizontal = np.array([[-lh, lv - t / 2], [t / 2, lv - t / 2], [t / 2, lv + t / 2], [-lh, lv + t / 2]])
    return vertical, horizontal


def hook_polygons(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return tuple(transform(poly, state[H_POS], float(state[H_TH])) for poly in hook_local_polygons())


def _rect_local(cx: float, y0: float, width: float, height: float) -> np.ndarray:
    return np.array([
        [cx - width / 2, y0], [cx + width / 2, y0],
        [cx + width / 2, y0 + height], [cx - width / 2, y0 + height],
    ])


def gripper_polygons(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gap = float(state[G_GAP])
    left = _rect_local(-gap / 2, 0.0, FINGER_WIDTH, FINGER_LENGTH)
    right = _rect_local(gap / 2, 0.0, FINGER_WIDTH, FINGER_LENGTH)
    return tuple(transform(poly, state[G_POS], float(state[G_TH])) for poly in (left, right))


def _surface_segments_local() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    t, lv, lh = HOOK_THICKNESS, HOOK_HANDLE, HOOK_LONG
    return {
        "arm_top": (np.array([-lh, lv + t / 2]), np.array([t / 2, lv + t / 2]), np.array([0.0, 1.0])),
        "arm_end": (np.array([-lh, lv - t / 2]), np.array([-lh, lv + t / 2]), np.array([-1.0, 0.0])),
        "arm_bottom": (np.array([-lh, lv - t / 2]), np.array([-t / 2, lv - t / 2]), np.array([0.0, -1.0])),
        "handle_left": (np.array([-t / 2, 0.0]), np.array([-t / 2, lv - t / 2]), np.array([-1.0, 0.0])),
        "handle_end": (np.array([-t / 2, 0.0]), np.array([t / 2, 0.0]), np.array([0.0, -1.0])),
        "handle_right": (np.array([t / 2, 0.0]), np.array([t / 2, lv + t / 2]), np.array([1.0, 0.0])),
    }


def hook_surfaces(state: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    R = rotation(float(state[H_TH]))
    return {
        name: (state[H_POS] + R @ a, state[H_POS] + R @ b, R @ normal)
        for name, (a, b, normal) in _surface_segments_local().items()
    }


def block_faces(state: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    v = block_vertices(state)
    R = rotation(float(state[B_TH]))
    return {
        "bottom": (v[0], v[1], R @ np.array([0.0, -1.0])),
        "right": (v[1], v[2], R @ np.array([1.0, 0.0])),
        "top": (v[2], v[3], R @ np.array([0.0, 1.0])),
        "left": (v[3], v[0], R @ np.array([-1.0, 0.0])),
    }


def closest_points_segments(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    """Closest points and normalized coordinates on two finite segments."""
    eps = 1e-12
    u, v = b - a, d - c
    uu, vv = float(u @ u), float(v @ v)
    if uu <= eps or vv <= eps:
        raise ValueError("contact surface segment has zero length")

    determinant = cross2(u, v)
    scale = np.sqrt(uu * vv)
    if abs(determinant) > eps * scale:
        offset = c - a
        s = cross2(offset, v) / determinant
        t = cross2(offset, u) / determinant
        if -eps <= s <= 1.0 + eps and -eps <= t <= 1.0 + eps:
            s, t = float(np.clip(s, 0.0, 1.0)), float(np.clip(t, 0.0, 1.0))
            point = 0.5 * (a + s * u + c + t * v)
            return point, point, 0.0, s, t
    else:
        # Parallel segments with overlapping tangential projections have a
        # continuum of closest pairs.  Return its midpoint so a face contact
        # is not misclassified as a corner contact.
        s0, s1 = float((c - a) @ u / uu), float((d - a) @ u / uu)
        lo, hi = max(0.0, min(s0, s1)), min(1.0, max(s0, s1))
        if lo <= hi + eps:
            s = 0.5 * (lo + hi)
            pa = a + s * u
            t = float(np.clip((pa - c) @ v / vv, 0.0, 1.0))
            pb = c + t * v
            return pa, pb, float(np.linalg.norm(pa - pb)), float(s), t

    def project(point: np.ndarray, start: np.ndarray, delta: np.ndarray, norm2: float):
        coordinate = float(np.clip((point - start) @ delta / norm2, 0.0, 1.0))
        return start + coordinate * delta, coordinate

    candidates = []
    for s, point in ((0.0, a), (1.0, b)):
        pb, t = project(point, c, v, vv)
        candidates.append((float(np.linalg.norm(point - pb)), point, pb, s, t))
    for t, point in ((0.0, c), (1.0, d)):
        pa, s = project(point, a, u, uu)
        candidates.append((float(np.linalg.norm(pa - point)), pa, point, s, t))
    distance, pa, pb, s, t = min(candidates, key=lambda item: item[0])
    return pa, pb, distance, float(s), float(t)


def nearest_contact(
    state: np.ndarray,
    allowed_hook_surfaces: tuple[str, ...] | None = None,
) -> dict:
    best = None
    active = None
    hooks, faces = hook_surfaces(state), block_faces(state)
    for hook_name, block_name in CONTACT_PAIRS.items():
        if allowed_hook_surfaces is not None and hook_name not in allowed_hook_surfaces:
            continue
        ha, hb, hn = hooks[hook_name]
        ba, bb, bn = faces[block_name]
        # Match the MILP's controlled-contact manifold by searching only the
        # interior patches.  Convert the resulting coordinates back to the
        # normalized coordinates of the complete finite segments.
        margin = CONTACT_PATCH_MARGIN
        hs, bs = hb - ha, bb - ba
        ph, pb, distance, rh, rb = closest_points_segments(
            ha + margin * hs, hb - margin * hs,
            ba + margin * bs, bb - margin * bs,
        )
        rh = margin + (1.0 - 2.0 * margin) * rh
        rb = margin + (1.0 - 2.0 * margin) * rb
        alignment = float(hn @ bn)
        item = {
            "hook_surface": hook_name, "block_face": block_name,
            "hook_point": ph, "block_point": pb, "point": 0.5 * (ph + pb),
            "distance": distance, "hook_coordinate": rh, "block_coordinate": rb,
            "normal_alignment": alignment,
        }
        if alignment <= -0.75 and (best is None or distance < best["distance"]):
            best = item
        # CONTACT_PAIRS has a fixed canonical order.  In a degenerate L-corner
        # contact, use its first active face as the unique representative of
        # the physical multi-contact set instead of switching on tiny distance
        # differences from the collision solver.
        if alignment <= -0.75 and distance <= CONTACT_TOL and active is None:
            active = item
    return active or best or {
        "hook_surface": None, "block_face": None,
        "hook_point": np.zeros(2), "block_point": np.zeros(2), "point": np.zeros(2),
        "distance": np.inf, "hook_coordinate": 0.0, "block_coordinate": 0.0,
        "normal_alignment": 1.0,
    }


def linearized_separation_certificate(
    state: np.ndarray,
    wall_swept_contact: bool = False,
) -> tuple[str, str]:
    """Decode the active polyhedral free-space region.

    The same factors select finite contact surfaces in ``contact_active`` and
    collision-free route regions in ``contact_separate``.  The reset starts in
    the low region; after that the fixed MILP keeps every predicted state in
    one of the right/above/left regions until physical contact is established.
    """
    rel = state[H_POS] - state[B_POS]
    theta = wrap_angle(float(state[H_TH]))
    chart = abs(theta) <= ROUTE_CHART_ANGLE + 1e-9
    numeric_angle_tol = 1e-7
    if theta > ROUTE_CHART_ANGLE + numeric_angle_tol:
        return "arm_top", "bottom"
    if theta < -ROUTE_CHART_ANGLE - numeric_angle_tol:
        return "handle_left", "right"
    if wall_swept_contact and rel[0] <= ROUTE_ALIGN_SWITCH_X + 1e-9:
        return "arm_bottom", "top"
    if chart and rel[1] <= ROUTE_LOW_CHART_REL_Y + 1e-9:
        return "handle_right", "left"
    if chart and rel[1] < ROUTE_CLEAR_REL_Y - 1e-9:
        return "arm_end", "right"
    if chart and rel[0] >= ROUTE_ALIGN_SWITCH_X - 1e-9:
        return "handle_end", "top"
    return "arm_bottom", "top"


def _grasp_contact(state: np.ndarray) -> bool:
    rel = rotation(float(state[G_TH])).T @ (state[H_POS] - state[G_POS])
    angle_error = wrap_angle(float(state[H_TH] - state[G_TH]))
    inner_face = 0.5 * float(state[G_GAP]) - 0.5 * FINGER_WIDTH
    bilateral = abs(inner_face - HOOK_THICKNESS / 2) <= 0.014
    return bool(np.linalg.norm(rel) <= 0.018 and abs(angle_error) <= 0.055 and bilateral)


def infer_z(state: np.ndarray, held: bool) -> np.ndarray:
    contact = nearest_contact(state, ("arm_bottom",))
    # Controlled contact excludes both bodies' corners.  Endpoint impacts are
    # still handled by Pymunk, but they are not a stable local contact chart.
    interior_patch = bool(
        CONTACT_PATCH_MARGIN <= contact["hook_coordinate"] <= 1.0 - CONTACT_PATCH_MARGIN
        and CONTACT_PATCH_MARGIN <= contact["block_coordinate"] <= 1.0 - CONTACT_PATCH_MARGIN
    )
    controlled_contact = bool(
        held
        and contact["distance"] <= CONTACT_TOL
        and interior_patch
        and abs(wrap_angle(float(state[H_TH]))) <= HOOK_CONTACT_CHART_ANGLE + 1e-7
        and abs(wrap_angle(float(state[B_TH]))) <= BLOCK_CONTACT_CHART_ANGLE + 1e-7
    )
    swept_wall_contact = bool(
        np.min(block_vertices(state)[:, 1]) - WALL_Y <= MAX_DXY + CONTACT_TOL
    )
    if controlled_contact:
        return mode_vector(
            held=held,
            hook_surface=contact["hook_surface"],
            block_face=contact["block_face"],
            contact_active=True,
            wall_swept_contact=swept_wall_contact,
        )
    hook_surface, block_face = linearized_separation_certificate(
        state, swept_wall_contact,
    )
    return mode_vector(
        held=held,
        hook_surface=hook_surface,
        block_face=block_face,
        contact_active=False,
        wall_swept_contact=swept_wall_contact,
    )


def pack_observation(state: np.ndarray, z: np.ndarray, valid: bool = True) -> np.ndarray:
    return np.r_[np.asarray(state, float), np.asarray(z, float), float(valid)]


def unpack_observation(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    observation = np.asarray(observation, float)
    if observation.shape != (OBS_DIM,):
        raise ValueError(f"expected observation {(OBS_DIM,)}, got {observation.shape}")
    return (
        observation[:S_DIM].copy(),
        np.rint(observation[S_DIM:S_DIM + Z_DIM]).astype(int),
        bool(observation[-1] > 0.5),
    )


def _hook_shapes(body: pymunk.Body) -> list[pymunk.Poly]:
    shapes = [pymunk.Poly(body, poly.tolist()) for poly in hook_local_polygons()]
    for shape in shapes:
        shape.friction = SHAPE_FRICTION
        shape.elasticity = 0.0
        shape.filter = pymunk.ShapeFilter(
            categories=HOOK_CATEGORY,
            mask=BLOCK_CATEGORY | BOUNDARY_CATEGORY,
        )
    return shapes


def _hook_moment() -> float:
    polys = hook_local_polygons()
    return float(sum(pymunk.moment_for_poly(HOOK_MASS / 2, poly.tolist()) for poly in polys))


HOOK_INERTIA = _hook_moment()


def _make_space(state: np.ndarray, held: bool):
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    space.damping = SPACE_DAMPING
    space.collision_slop = 0.0015
    space.iterations = 30

    block = pymunk.Body(BLOCK_MASS, BLOCK_INERTIA)
    block.position = tuple(state[B_POS])
    block.angle = float(state[B_TH])
    block.velocity = tuple(state[[B_VX, B_VY]])
    block.angular_velocity = float(state[B_OM])
    block_shape = pymunk.Poly.create_box(block, (BLOCK_SIDE, BLOCK_SIDE))
    block_shape.friction = SHAPE_FRICTION
    block_shape.elasticity = 0.0
    block_shape.filter = pymunk.ShapeFilter(
        categories=BLOCK_CATEGORY,
        mask=HOOK_CATEGORY | MIDDLE_WALL_CATEGORY | BOUNDARY_CATEGORY,
    )

    hook = pymunk.Body(body_type=pymunk.Body.KINEMATIC) if held else pymunk.Body(HOOK_MASS, HOOK_INERTIA)
    hook.position = tuple(state[H_POS])
    hook.angle = float(state[H_TH])
    hook.velocity = tuple(state[[H_VX, H_VY]])
    hook.angular_velocity = float(state[H_OM])
    hook_shapes = _hook_shapes(hook)

    wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    wall_body.position = (0.5 * sum(WORLD_X), WALL_CENTER_Y)
    wall = pymunk.Poly.create_box(wall_body, (WORLD_X[1] - WORLD_X[0], WALL_THICKNESS))
    wall.friction = SHAPE_FRICTION
    wall.elasticity = 0.0
    wall.filter = pymunk.ShapeFilter(
        categories=MIDDLE_WALL_CATEGORY,
        mask=BLOCK_CATEGORY,
    )

    boundary_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    boundary_specs = (
        ((WORLD_X[0], WORLD_Y[0]), (WORLD_X[1], WORLD_Y[0])),
        ((WORLD_X[0], WORLD_Y[1]), (WORLD_X[1], WORLD_Y[1])),
        ((WORLD_X[0], WORLD_Y[0]), (WORLD_X[0], WORLD_Y[1])),
        ((WORLD_X[1], WORLD_Y[0]), (WORLD_X[1], WORLD_Y[1])),
    )
    boundaries = []
    for start, end in boundary_specs:
        shape = pymunk.Segment(boundary_body, start, end, radius=0.0125)
        shape.friction = SHAPE_FRICTION
        shape.elasticity = 0.0
        shape.filter = pymunk.ShapeFilter(
            categories=BOUNDARY_CATEGORY,
            mask=BLOCK_CATEGORY | HOOK_CATEGORY,
        )
        boundaries.append(shape)
    space.add(
        block, block_shape, hook, *hook_shapes,
        wall_body, wall, boundary_body, *boundaries,
    )
    return space, block, block_shape, hook, hook_shapes


def _convex_penetration(a: np.ndarray, b: np.ndarray) -> float:
    """Return SAT overlap depth, or zero for separated convex polygons."""
    depth = np.inf
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for edge in edges:
            axis = np.array([-edge[1], edge[0]], dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                continue
            axis /= norm
            pa, pb = a @ axis, b @ axis
            overlap = min(float(pa.max()), float(pb.max())) - max(float(pa.min()), float(pb.min()))
            if overlap <= 0.0:
                return 0.0
            depth = min(depth, overlap)
    return float(depth if np.isfinite(depth) else 0.0)


def finite_volume_penetration(state: np.ndarray) -> float:
    block = block_vertices(state)
    return max(_convex_penetration(polygon, block) for polygon in hook_polygons(state))


def step(observation: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Execute one command in independent Pymunk contact dynamics."""
    state, z, valid = unpack_observation(observation)
    u = np.asarray(control, float)
    if u.shape != (4,):
        raise ValueError(f"expected action (4,), got {u.shape}")
    if not valid:
        return observation.copy()
    limits = np.array([MAX_DXY, MAX_DXY, MAX_DTHETA, MAX_DGAP])
    if np.any(np.abs(u) > limits + 1e-7):
        raise ValueError(f"action exceeds limits: {u}")

    held = bool(z[1])
    space, block, block_shape, hook, hook_shapes = _make_space(state, held)
    if held:
        hook.velocity = tuple(u[:2] / CONTROL_DT)
        hook.angular_velocity = float(u[2] / CONTROL_DT)

    sub_dt = CONTROL_DT / PHYSICS_SUBSTEPS
    for _ in range(PHYSICS_SUBSTEPS):
        space.step(sub_dt)

    nxt = state.copy()
    nxt[G_POS] += u[:2]
    nxt[G_TH] = wrap_angle(float(nxt[G_TH] + u[2]))
    nxt[G_GAP] = float(np.clip(nxt[G_GAP] + u[3], GRIP_CLOSED, GRIP_OPEN))
    nxt[H_POS] = hook.position
    nxt[H_TH] = wrap_angle(float(hook.angle))
    nxt[[H_VX, H_VY]] = hook.velocity
    nxt[H_OM] = float(hook.angular_velocity)
    nxt[B_POS] = block.position
    nxt[B_TH] = wrap_angle(float(block.angle))
    nxt[[B_VX, B_VY]] = block.velocity
    nxt[B_OM] = float(block.angular_velocity)

    if held and nxt[G_GAP] >= GRIP_OPEN:
        held_next = False
    elif held:
        held_next = True
    else:
        held_next = _grasp_contact(nxt)

    penetration = finite_volume_penetration(nxt)
    rigid_points = np.vstack((*hook_polygons(nxt), block_vertices(nxt)))
    in_bounds = bool(
        rigid_points[:, 0].min() >= WORLD_X[0] - PENETRATION_TOL
        and rigid_points[:, 0].max() <= WORLD_X[1] + PENETRATION_TOL
        and rigid_points[:, 1].min() >= WORLD_Y[0] - PENETRATION_TOL
        and rigid_points[:, 1].max() <= WORLD_Y[1] + PENETRATION_TOL
    )
    valid_next = bool(valid and penetration <= PENETRATION_TOL and in_bounds)
    return pack_observation(nxt, infer_z(nxt, held_next), valid_next)


def block_wall_distance(state: np.ndarray) -> float:
    return float(np.min(block_vertices(state)[:, 1]) - WALL_Y)


def success(observation: np.ndarray) -> tuple[bool, float, float]:
    state, _, valid = unpack_observation(observation)
    distance = max(block_wall_distance(state), 0.0)
    return bool(valid and distance <= SUCCESS_TOL), float(distance), 0.0


def _rotation_corridor_valid(state: np.ndarray) -> bool:
    """Check the fixed affine rotate-to-chart corridor used by the MILP."""
    theta0 = wrap_angle(float(state[H_TH]))
    rel_y0 = float(state[H_POS.start + 1] - state[B_POS.start + 1])
    nominal0 = ROUTE_LOW_CHART_REL_Y + ROTATE_Y_SLOPE * abs(theta0)
    offset = rel_y0 - nominal0
    for fraction in np.linspace(0.0, 1.0, 25):
        probe = state.copy()
        theta = (1.0 - fraction) * theta0
        rel_y = (
            ROUTE_LOW_CHART_REL_Y
            + ROTATE_Y_SLOPE * abs(theta)
            + (1.0 - fraction) * offset
        )
        probe[H_TH] = probe[G_TH] = theta
        probe[H_POS.start + 1] = probe[B_POS.start + 1] + rel_y
        probe[G_POS] = probe[H_POS]
        hook_points = np.vstack(hook_polygons(probe))
        gripper_points = np.vstack(gripper_polygons(probe))
        if (
            hook_points[:, 0].min() < WORLD_X[0] + 0.02
            or hook_points[:, 0].max() > WORLD_X[1] - 0.02
            or hook_points[:, 1].min() < WORLD_Y[0] + 0.02
            or hook_points[:, 1].max() > WORLD_Y[1] - 0.02
            or gripper_points[:, 0].min() < WORLD_X[0] + 0.02
            or gripper_points[:, 0].max() > WORLD_X[1] - 0.02
            or gripper_points[:, 1].min() < WORLD_Y[0] + 0.02
            or gripper_points[:, 1].max() > WALL_CENTER_Y - 0.03
            or finite_volume_penetration(probe) > 1e-8
        ):
            return False
    return True


def sample_reset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample a non-enclosing reset with randomized rigid-body poses."""
    state = np.zeros(S_DIM, dtype=float)
    # Keep this draw outside the position-rejection loop: the accepted angle
    # distribution is therefore exactly the requested uniform prior.
    state[H_TH] = rng.uniform(-np.pi, np.pi)
    state[B_TH] = rng.uniform(-BLOCK_RESET_ANGLE, BLOCK_RESET_ANGLE)
    for _ in range(50_000):
        state[B_POS] = [
            rng.uniform(WORLD_X[0] + BLOCK_HALF + 0.20, WORLD_X[1] - BLOCK_HALF - 0.20),
            rng.uniform(
                WALL_Y + BLOCK_HALF + 0.35,
                WALL_Y + HOOK_HANDLE - BLOCK_HALF - 0.13,
            ),
        ]
        x_low = state[B_POS.start] + ROTATE_RIGHT_REL_X
        x_high = WORLD_X[1] - 0.12
        if x_low >= x_high:
            continue
        state[H_POS.start] = rng.uniform(x_low, x_high)
        state[H_POS.start + 1] = (
            state[B_POS.start + 1]
            + ROUTE_LOW_CHART_REL_Y
            + ROTATE_Y_SLOPE * abs(float(state[H_TH]))
        )
        state[G_POS] = state[H_POS]
        state[G_TH] = state[H_TH]
        state[G_GAP] = GRIP_CLOSED
        z = infer_z(state, held=True)
        obs = pack_observation(state, z, True)
        if (
            _rotation_corridor_valid(state)
            and block_wall_distance(state) > 0.30
        ):
            break
    else:
        raise RuntimeError("failed to sample a collision-free full-angle Hook reset")

    goal = state.copy()
    goal[B_POS.start + 1] = WALL_Y + BLOCK_HALF
    return obs, goal
