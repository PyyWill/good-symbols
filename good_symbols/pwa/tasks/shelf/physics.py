"""Shelf: rigid-body dynamics (Pymunk, with gravity) for packing heterogeneous rectangles into a shelf.

The mode ``z`` is one-hot over primitives: grasp an object, or place it on the shelf floor or on another
object.  Objects are packed from the back stack to the entrance stack through the shelf's single opening.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pymunk


OBJECTS = tuple("ABCDEFGH")
OBJECT_DIMS = {
    "A": (1.10, 0.22),
    "B": (0.78, 0.28),
    "C": (0.58, 0.30),
    "D": (0.24, 0.20),
    "E": (0.55, 0.42),
    "F": (0.22, 0.40),
    "G": (0.30, 0.35),
    "H": (0.28, 0.20),
}
GOAL_ORIENTATIONS = {
    "A": 0.0,
    "B": 0.0,
    "C": 0.0,
    "D": 0.0,
    "E": np.pi / 2,
    "F": np.pi / 2,
    "G": 0.0,
    "H": np.pi / 2,
}
SUPPORT_MARGIN = 0.04
DESTINATIONS = ("gripper", "shelf_floor", *OBJECTS)


def _oriented_dimensions(obj: str) -> tuple[float, float]:
    width, height = OBJECT_DIMS[obj]
    theta = GOAL_ORIENTATIONS[obj]
    c, s = abs(float(np.cos(theta))), abs(float(np.sin(theta)))
    return c * width + s * height, s * width + c * height


PLACEMENT_DIMS = {obj: _oriented_dimensions(obj) for obj in OBJECTS}
SUPPORT_ARCS = tuple(
    (source, destination)
    for source, destination in product(OBJECTS, repeat=2)
    if source != destination
    and PLACEMENT_DIMS[source][0]
    <= PLACEMENT_DIMS[destination][0] - 2.0 * SUPPORT_MARGIN + 1e-12
)
PRIMITIVES = tuple(
    primitive
    for source in OBJECTS
    for primitive in (
        (f"grasp_{source}", source, "gripper"),
        (f"place_{source}_on_floor", source, "shelf_floor"),
        *((f"place_{source}_on_{destination}", source, destination)
          for src, destination in SUPPORT_ARCS if src == source),
    )
)


def discrete_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed one-step ``(P^d, E^d, B^d)`` interface."""
    n = len(PRIMITIVES)
    E = np.ones((1, n), dtype=int)
    P = np.zeros((2 * len(OBJECTS), n), dtype=int)
    B = np.zeros((len(OBJECTS) + len(DESTINATIONS), n), dtype=int)
    for column, (_, source, destination) in enumerate(PRIMITIVES):
        source_index = OBJECTS.index(source)
        P[source_index, column] = 1
        B[source_index, column] = 1
        if destination in OBJECTS:
            P[len(OBJECTS) + OBJECTS.index(destination), column] = 1
        B[len(OBJECTS) + DESTINATIONS.index(destination), column] = -1
    return P, E, B


N_OBJECTS = len(OBJECTS)
Z_DIM = len(PRIMITIVES)
S_DIM = 6 * N_OBJECTS + 6
OBS_DIM = S_DIM + Z_DIM + N_OBJECTS + 1
GOAL_DIM = S_DIM + Z_DIM

WORLD_X = (0.0, 8.6)
WORLD_Y = (0.0, 2.2)
SHELF_BACK_X = 0.18
SHELF_OPEN_X = 1.88
SHELF_FLOOR_Y = 0.24
SHELF_CEILING_Y = 1.55
SHELF_THICKNESS = 0.045
SHELF_SUPPORT_Y = SHELF_FLOOR_Y + SHELF_THICKNESS / 2

CONTROL_DT = 0.20
PHYSICS_SUBSTEPS = 16
RELEASE_SUBSTEPS = 20
RESET_SETTLE_SUBSTEPS = 80
GRAVITY = 5.0
SPACE_DAMPING = 0.72
FRICTION = 0.70
MAX_DX = 0.38
MAX_DY = 0.95
MAX_DTHETA = 0.28
CONTACT_TOL = 0.055
RELEASE_GAP = 0.095
PENETRATION_TOL = 0.025
POS_TOL = 0.075
ANG_TOL = 0.12
VEL_TOL = 0.16

PACKING_CHAINS = (("E", "G", "H"), ("A", "B", "C", "F", "D"))
# The shelf has one opening on the right, so objects must be packed from the
# back stack to the entrance stack to avoid crossing already placed bodies.
PACKING_ORDER = PACKING_CHAINS[0] + PACKING_CHAINS[1]


def object_slice(index: int) -> slice:
    return slice(6 * index, 6 * index + 6)


def pose_slice(index: int) -> slice:
    return slice(6 * index, 6 * index + 3)


GRASP_FRAME = slice(6 * N_OBJECTS, 6 * N_OBJECTS + 6)


def wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2 * np.pi) - np.pi)


def angdiff(a: float, b: float) -> float:
    return wrap_angle(a - b)


def canonical_orientation(obj: str, value: float) -> float:
    """Represent an unmarked rectangle in the goal-centered SO(2)/C2 chart."""
    goal = GOAL_ORIENTATIONS[obj]
    return float(goal + (value - goal + np.pi / 2) % np.pi - np.pi / 2)


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)), dtype=float)


def rectangle_vertices(pose: np.ndarray, dimensions: tuple[float, float]) -> np.ndarray:
    width, height = dimensions
    local = 0.5 * np.array(
        ((-width, -height), (width, -height), (width, height), (-width, height)),
        dtype=float,
    )
    return local @ rotation(float(pose[2])).T + np.asarray(pose[:2], dtype=float)


def _sat_penetration(a: np.ndarray, b: np.ndarray) -> float:
    depth = np.inf
    for polygon in (a, b):
        for edge in np.roll(polygon, -1, axis=0) - polygon:
            axis = np.array((-edge[1], edge[0]), dtype=float)
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


def primitive_index(source: str, destination: str) -> int:
    return next(i for i, (_, src, dst) in enumerate(PRIMITIVES) if src == source and dst == destination)


def primitive_vector(source: str, destination: str) -> np.ndarray:
    z = np.zeros(Z_DIM, dtype=int)
    z[primitive_index(source, destination)] = 1
    return z


def decode_primitive(z: np.ndarray) -> tuple[int, str, str, str]:
    z = np.asarray(z, dtype=float)
    if z.shape != (Z_DIM,) or abs(float(z.sum()) - 1.0) > 1e-6:
        raise ValueError("z must be a one-hot primitive vector")
    index = int(np.argmax(z))
    name, source, destination = PRIMITIVES[index]
    return index, name, source, destination


def availability_rhs(active: np.ndarray) -> np.ndarray:
    active = np.asarray(active, dtype=int)
    return np.r_[active, active]


def primitive_available(z: np.ndarray, active: np.ndarray) -> bool:
    _, _, source, destination = decode_primitive(z)
    return bool(
        active[OBJECTS.index(source)]
        and (destination not in OBJECTS or active[OBJECTS.index(destination)])
    )


def pack_observation(state: np.ndarray, z: np.ndarray, active: np.ndarray, valid: bool = True) -> np.ndarray:
    return np.r_[np.asarray(state, float), np.asarray(z, float), np.asarray(active, float), float(valid)]


def unpack_observation(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    observation = np.asarray(observation, dtype=float)
    if observation.shape != (OBS_DIM,):
        raise ValueError(f"expected observation {(OBS_DIM,)}, got {observation.shape}")
    return (
        observation[:S_DIM].copy(),
        np.rint(observation[S_DIM:S_DIM + Z_DIM]).astype(int),
        np.rint(observation[S_DIM + Z_DIM:S_DIM + Z_DIM + N_OBJECTS]).astype(int),
        bool(observation[-1] > 0.5),
    )


def pack_goal(state: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.r_[np.asarray(state, float), np.asarray(z, float)]


def unpack_goal(goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    goal = np.asarray(goal, dtype=float)
    if goal.shape != (GOAL_DIM,):
        raise ValueError(f"expected goal {(GOAL_DIM,)}, got {goal.shape}")
    return goal[:S_DIM].copy(), np.rint(goal[S_DIM:]).astype(int)


def _goal_layout(active: np.ndarray, x_shift: float = 0.0) -> tuple[np.ndarray, dict[str, str]]:
    """Construct two stable, collision-free stacks for any 4--8 object mask."""
    active_set = {obj for obj, present in zip(OBJECTS, active) if present}
    state = np.zeros(S_DIM, dtype=float)
    support: dict[str, str] = {}
    centers = (0.43 + x_shift, 1.28 + x_shift)
    for center_x, chain in zip(centers, PACKING_CHAINS):
        present = [obj for obj in chain if obj in active_set]
        top = SHELF_SUPPORT_Y
        previous = "shelf_floor"
        for obj in present:
            _, height = PLACEMENT_DIMS[obj]
            state[pose_slice(OBJECTS.index(obj))] = (
                center_x, top + height / 2, GOAL_ORIENTATIONS[obj],
            )
            support[obj] = previous
            previous = obj
            top += height
        if top > SHELF_CEILING_Y - SHELF_THICKNESS / 2 + 1e-9:
            raise RuntimeError("active mask exceeds shelf clearance")

    inactive_pose = (WORLD_X[1] - 0.25, WORLD_Y[1] - 0.25, 0.0)
    for i, obj in enumerate(OBJECTS):
        if obj not in active_set:
            state[pose_slice(i)] = inactive_pose
    last = next(obj for obj in reversed(PACKING_ORDER) if obj in active_set)
    state[GRASP_FRAME] = state[object_slice(OBJECTS.index(last))]
    return state, support


def _floor_pose(x: float, theta: float, obj: str) -> np.ndarray:
    pose = np.array((x, 0.0, theta), dtype=float)
    pose[1] = SHELF_SUPPORT_Y - rectangle_vertices(pose, OBJECT_DIMS[obj])[:, 1].min() + 0.006
    return pose


def sample_reset(
    rng: np.random.Generator,
    *,
    active_count: int | None = None,
    rotation_span: float = np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a guaranteed-feasible goal and a collision-free staging line."""
    if active_count is None:
        active_count = int(rng.integers(4, N_OBJECTS + 1))
    if not 4 <= active_count <= N_OBJECTS:
        raise ValueError(f"active_count must be in [4,{N_OBJECTS}]")
    if not 0.0 <= rotation_span <= np.pi:
        raise ValueError("rotation_span must be in [0, pi]")
    active = np.zeros(N_OBJECTS, dtype=int)
    active[rng.choice(N_OBJECTS, size=active_count, replace=False)] = 1
    goal_state, support = _goal_layout(active, x_shift=float(rng.uniform(-0.018, 0.018)))

    state = np.zeros(S_DIM, dtype=float)
    cursor = 2.05 + float(rng.uniform(0.0, 0.08))
    for obj in PACKING_ORDER:
        i = OBJECTS.index(obj)
        if not active[i]:
            state[pose_slice(i)] = goal_state[pose_slice(i)]
            continue
        theta = canonical_orientation(
            obj, GOAL_ORIENTATIONS[obj] + float(rng.uniform(-rotation_span, rotation_span)),
        )
        rotation_diameter = float(np.hypot(*OBJECT_DIMS[obj]))
        center_x = cursor + rotation_diameter / 2
        state[pose_slice(i)] = _floor_pose(center_x, theta, obj)
        cursor += rotation_diameter + float(rng.uniform(0.13, 0.19))
    if cursor > WORLD_X[1] - 0.05:
        raise RuntimeError("staging line exceeds the world boundary")

    first = next(obj for obj in PACKING_ORDER if active[OBJECTS.index(obj)])
    last = next(obj for obj in reversed(PACKING_ORDER) if active[OBJECTS.index(obj)])
    settle_space, settle_bodies, _ = _make_space(state, active, OBJECTS.index(first))
    for _ in range(RESET_SETTLE_SUBSTEPS):
        settle_space.step(CONTROL_DT / PHYSICS_SUBSTEPS)
    state = _read_bodies(state, settle_bodies)
    for i in range(N_OBJECTS):
        state[object_slice(i)][3:] = 0.0
    state[GRASP_FRAME] = state[object_slice(OBJECTS.index(first))]
    z0 = primitive_vector(first, "gripper")
    goal_z = primitive_vector(last, support[last])
    geometry_ok, _ = _geometry_valid(state, active)
    if not geometry_ok:
        raise RuntimeError("reset sampler produced intersecting finite-volume objects")
    return pack_observation(state, z0, active), pack_goal(goal_state, goal_z)


def _make_space(state: np.ndarray, active: np.ndarray, grasped: int | None):
    space = pymunk.Space()
    space.gravity = (0.0, -GRAVITY)
    space.damping = SPACE_DAMPING
    space.iterations = 40
    space.collision_slop = 0.001

    static = pymunk.Body(body_type=pymunk.Body.STATIC)
    shelf_segments = (
        ((WORLD_X[0], SHELF_FLOOR_Y), (WORLD_X[1], SHELF_FLOOR_Y)),
        ((SHELF_BACK_X, SHELF_FLOOR_Y), (SHELF_BACK_X, SHELF_CEILING_Y)),
        ((SHELF_BACK_X, SHELF_CEILING_Y), (SHELF_OPEN_X, SHELF_CEILING_Y)),
        ((WORLD_X[0], WORLD_Y[0]), (WORLD_X[0], WORLD_Y[1])),
        ((WORLD_X[1], WORLD_Y[0]), (WORLD_X[1], WORLD_Y[1])),
    )
    walls = [pymunk.Segment(static, a, b, SHELF_THICKNESS / 2) for a, b in shelf_segments]
    for wall in walls:
        wall.friction = FRICTION
        wall.elasticity = 0.0
    space.add(static, *walls)

    bodies: list[pymunk.Body | None] = [None] * N_OBJECTS
    shapes: list[pymunk.Poly | None] = [None] * N_OBJECTS
    for i, obj in enumerate(OBJECTS):
        if not active[i]:
            continue
        width, height = OBJECT_DIMS[obj]
        body = (
            pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            if i == grasped
            else pymunk.Body(1.0 + width * height, pymunk.moment_for_box(1.0 + width * height, (width, height)))
        )
        block = state[object_slice(i)]
        body.position = tuple(block[:2])
        body.angle = float(block[2])
        body.velocity = tuple(block[3:5])
        body.angular_velocity = float(block[5])
        shape = pymunk.Poly.create_box(body, (width, height))
        shape.friction = FRICTION
        shape.elasticity = 0.0
        space.add(body, shape)
        bodies[i], shapes[i] = body, shape
    return space, bodies, shapes


def _geometry_valid(state: np.ndarray, active: np.ndarray) -> tuple[bool, float]:
    polygons = [rectangle_vertices(state[pose_slice(i)], OBJECT_DIMS[obj]) for i, obj in enumerate(OBJECTS)]
    max_penetration = 0.0
    for i in range(N_OBJECTS):
        if not active[i]:
            continue
        vertices = polygons[i]
        if (
            vertices[:, 0].min() < WORLD_X[0] - PENETRATION_TOL
            or vertices[:, 0].max() > WORLD_X[1] + PENETRATION_TOL
            or vertices[:, 1].min() < SHELF_FLOOR_Y - PENETRATION_TOL
            or vertices[:, 1].max() > WORLD_Y[1] + PENETRATION_TOL
        ):
            return False, np.inf
        inside_x = vertices[:, 0].min() < SHELF_OPEN_X
        if inside_x and (
            vertices[:, 0].min() < SHELF_BACK_X - PENETRATION_TOL
            or vertices[:, 1].max() > SHELF_CEILING_Y + PENETRATION_TOL
        ):
            return False, np.inf
        for j in range(i):
            if active[j]:
                max_penetration = max(max_penetration, _sat_penetration(vertices, polygons[j]))
    return bool(max_penetration <= PENETRATION_TOL), float(max_penetration)


def support_contact(
    state: np.ndarray, source: str, destination: str, *, tolerance: float = CONTACT_TOL,
) -> bool:
    src = rectangle_vertices(state[pose_slice(OBJECTS.index(source))], OBJECT_DIMS[source])
    if destination == "shelf_floor":
        gap = abs(float(src[:, 1].min() - SHELF_SUPPORT_Y))
        return bool(gap <= tolerance and src[:, 0].max() <= SHELF_OPEN_X + CONTACT_TOL)
    dst = rectangle_vertices(state[pose_slice(OBJECTS.index(destination))], OBJECT_DIMS[destination])
    vertical_gap = abs(float(src[:, 1].min() - dst[:, 1].max()))
    overlap = min(float(src[:, 0].max()), float(dst[:, 0].max())) - max(
        float(src[:, 0].min()), float(dst[:, 0].min())
    )
    return bool(vertical_gap <= tolerance and overlap >= 2.0 * SUPPORT_MARGIN)


def _read_bodies(state: np.ndarray, bodies: list[pymunk.Body | None]) -> np.ndarray:
    nxt = state.copy()
    for i, body in enumerate(bodies):
        if body is None:
            continue
        block = nxt[object_slice(i)]
        block[:2] = body.position
        block[2] = canonical_orientation(OBJECTS[i], float(body.angle))
        block[3:5] = body.velocity
        block[5] = float(body.angular_velocity)
    return nxt


def step(observation: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Execute the first MILP action with rigid collisions and gravity."""
    state, z, active, valid = unpack_observation(observation)
    if not valid:
        raise ValueError("cannot execute from an invalid physical state")
    action = np.asarray(action, dtype=float)
    if action.shape != (3 + Z_DIM,):
        raise ValueError(f"expected action {(3 + Z_DIM,)}, got {action.shape}")
    u, next_z = action[:3], np.rint(action[3:]).astype(int)
    if np.any(np.abs(u) > np.array((MAX_DX, MAX_DY, MAX_DTHETA)) + 1e-7):
        raise ValueError(f"action exceeds limits: {u}")
    if not primitive_available(z, active) or not primitive_available(next_z, active):
        return pack_observation(state, next_z, active, False)

    _, _, source, destination = decode_primitive(z)
    source_i = OBJECTS.index(source)
    grasped = source_i if destination == "gripper" else None
    if grasped is None and np.linalg.norm(u) > 1e-7:
        return pack_observation(state, next_z, active, False)

    space, bodies, _ = _make_space(state, active, grasped)
    if grasped is not None:
        body = bodies[grasped]
        assert body is not None
        body.velocity = tuple(u[:2] / CONTROL_DT)
        body.angular_velocity = float(u[2] / CONTROL_DT)
    for _ in range(PHYSICS_SUBSTEPS):
        space.step(CONTROL_DT / PHYSICS_SUBSTEPS)

    nxt = _read_bodies(state, bodies)
    if grasped is not None:
        nxt[object_slice(grasped)][3:] = 0.0

    _, _, next_source, next_destination = decode_primitive(next_z)
    release_admissible = (
        next_destination == "gripper"
        or support_contact(nxt, next_source, next_destination, tolerance=RELEASE_GAP)
    )
    if next_destination != "gripper" and release_admissible:
        settle_space, settle_bodies, _ = _make_space(nxt, active, grasped=None)
        for _ in range(RELEASE_SUBSTEPS):
            settle_space.step(CONTROL_DT / RELEASE_SUBSTEPS)
        nxt = _read_bodies(nxt, settle_bodies)
    next_i = OBJECTS.index(next_source)
    nxt[GRASP_FRAME] = nxt[object_slice(next_i)]
    geometry_ok, _ = _geometry_valid(nxt, active)
    mode_ok = (
        next_destination == "gripper"
        or support_contact(nxt, next_source, next_destination, tolerance=CONTACT_TOL)
    )
    return pack_observation(nxt, next_z, active, bool(geometry_ok and release_admissible and mode_ok))


def errors(observation: np.ndarray, goal: np.ndarray) -> tuple[float, float, float]:
    state, _, active, _ = unpack_observation(observation)
    goal_state, _ = unpack_goal(goal)
    present = [i for i in range(N_OBJECTS) if active[i]]
    pos = [np.linalg.norm(state[pose_slice(i)][:2] - goal_state[pose_slice(i)][:2]) for i in present]
    angles = [abs(angdiff(state[pose_slice(i)][2], goal_state[pose_slice(i)][2])) for i in present]
    speeds = [np.linalg.norm(state[object_slice(i)][3:]) for i in range(N_OBJECTS) if active[i]]
    return float(max(pos, default=0.0)), float(max(angles, default=0.0)), float(max(speeds, default=0.0))


def success(observation: np.ndarray, goal: np.ndarray) -> tuple[bool, float, float]:
    state, z, active, valid = unpack_observation(observation)
    dpos, dang, speed = errors(observation, goal)
    _, _, _, destination = decode_primitive(z)
    inside = True
    for i, obj in enumerate(OBJECTS):
        if not active[i]:
            continue
        vertices = rectangle_vertices(state[pose_slice(i)], OBJECT_DIMS[obj])
        inside &= bool(
            vertices[:, 0].min() >= SHELF_BACK_X - CONTACT_TOL
            and vertices[:, 0].max() <= SHELF_OPEN_X + CONTACT_TOL
            and vertices[:, 1].min() >= SHELF_FLOOR_Y - CONTACT_TOL
            and vertices[:, 1].max() <= SHELF_CEILING_Y + CONTACT_TOL
        )
    ok = valid and inside and destination != "gripper" and dpos <= POS_TOL and dang <= ANG_TOL and speed <= VEL_TOL
    return bool(ok), dpos, dang
