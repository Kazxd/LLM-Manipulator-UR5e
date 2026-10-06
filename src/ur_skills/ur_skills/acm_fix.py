"""Allow the Robotiq gripper's self-contacts in MoveIt's allowed collision matrix at runtime.

The stock UR SRDF does not know the gripper, so its overlapping collision meshes (base/knuckles/fingers) and the
gripper base touching wrist_3 make every start state invalid. This edits the matrix in the running move_group.
"""
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents, AllowedCollisionEntry

GRIPPER_LINKS = [
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link", "robotiq_85_right_knuckle_link",
    "robotiq_85_left_inner_knuckle_link", "robotiq_85_right_inner_knuckle_link",
    "robotiq_85_left_finger_link", "robotiq_85_right_finger_link",
    "robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link",
]
ARM_LINKS = ["forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link", "flange", "tool0", "tool0_link"]


def allow_gripper_collisions(node, wait, callback_group=None, timeout=10.0):
    """wait(future, timeout) -> result or None. Returns (ok, message)."""
    get = node.create_client(GetPlanningScene, "/get_planning_scene", callback_group=callback_group)
    app = node.create_client(ApplyPlanningScene, "/apply_planning_scene", callback_group=callback_group)
    try:
        if not (get.wait_for_service(timeout_sec=5.0) and app.wait_for_service(timeout_sec=5.0)):
            return False, "/get_planning_scene or /apply_planning_scene not available (is move_group running?)"
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        res = wait(get.call_async(req), timeout)
        if res is None:
            return False, "get_planning_scene timed out"
        acm = res.scene.allowed_collision_matrix
        names = list(acm.entry_names)
        rows = [list(e.enabled) for e in acm.entry_values]

        def idx(n):
            if n not in names:
                for r in rows:
                    r.append(False)
                names.append(n)
                rows.append([False] * len(names))
            return names.index(n)

        for a in GRIPPER_LINKS:
            for b in GRIPPER_LINKS + ARM_LINKS:
                if a == b:
                    continue
                i, j = idx(a), idx(b)
                rows[i][j] = rows[j][i] = True

        acm.entry_names = names
        acm.entry_values = [AllowedCollisionEntry(enabled=r) for r in rows]
        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = acm
        req2 = ApplyPlanningScene.Request()
        req2.scene = scene
        r2 = wait(app.call_async(req2), timeout)
        if r2 is None:
            return False, "apply_planning_scene timed out"
        return bool(r2.success), "gripper collision pairs allowed" if r2.success else "apply_planning_scene refused the update"
    finally:
        node.destroy_client(get)
        node.destroy_client(app)