#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import inspect
import os
import pybullet as p
from Dualenv import Dualenv
from Helper import Helper

current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent = os.path.dirname(os.path.dirname(current))
os.sys.path.insert(0, parent)


def main():
    env = Dualenv(renders=True, is_discrete=False)
    hp = Helper()
    motorsIds = []

    # dv = math.pi
    dv = 3.5
    # Arm
    motorsIds.append(env.p.addUserDebugParameter("posX_right", -dv, dv, 0))  # right arm 0
    motorsIds.append(env.p.addUserDebugParameter("posY_right", -dv, dv, 0))  # right arm 1
    motorsIds.append(env.p.addUserDebugParameter("posZ_right", -dv, dv, 0))  # right arm 2

    motorsIds.append(env.p.addUserDebugParameter("right_roll", -0.1, 0.1, 0))  # right arm 3
    motorsIds.append(env.p.addUserDebugParameter("right_pitch", -0.1, 0.1, 0))  # right arm 4
    motorsIds.append(env.p.addUserDebugParameter("right_yaw", -0.1, 0.1, 0))  # right arm 5

    motorsIds.append(env.p.addUserDebugParameter("j_88", -0.1, 0.1, 0))  # right index 6
    motorsIds.append(env.p.addUserDebugParameter("j_92", -0.1, 0.1, 0))  # right mid 7
    motorsIds.append(env.p.addUserDebugParameter("j_105", -0.1, 0.1, 0))  # right ring 8
    motorsIds.append(env.p.addUserDebugParameter("j_100", -0.1, 0.1, 0))  # right pinky 9
    motorsIds.append(env.p.addUserDebugParameter("j_81", -0.1, 0.1, 0))  # right thumb 10

    motorsIds.append(env.p.addUserDebugParameter("posX_left", -dv, dv, 0))  # left arm 11
    motorsIds.append(env.p.addUserDebugParameter("posY_left", -dv, dv, 0))  # left arm 12
    motorsIds.append(env.p.addUserDebugParameter("posZ_left", -dv, dv, 0))  # left arm 13

    motorsIds.append(env.p.addUserDebugParameter("left_roll", -0.1, 0.1, 0))  # right arm 14
    motorsIds.append(env.p.addUserDebugParameter("left_pitch", -0.1, 0.1, 0))  # right arm 15
    motorsIds.append(env.p.addUserDebugParameter("left_yaw", -0.1, 0.1, 0))  # right arm 16

    motorsIds.append(env.p.addUserDebugParameter("j_42", -0.1, 0.1, 0))  # left index 17
    motorsIds.append(env.p.addUserDebugParameter("j_46", -0.1, 0.1, 0))  # left mid 18
    motorsIds.append(env.p.addUserDebugParameter("j_59", -0.1, 0.1, 0))  # left ring 19
    motorsIds.append(env.p.addUserDebugParameter("j_54", -0.1, 0.1, 0))  # left pinky 20
    motorsIds.append(env.p.addUserDebugParameter("j_35", -0.1, 0.1, 0))  # left thumb 21

    motorsIds.append(env.p.addUserDebugParameter("w1", -1, 1, 0))  # front_left_wheel 22
    motorsIds.append(env.p.addUserDebugParameter("w2", -1, 1, 0))  # front_right_wheel 23
    motorsIds.append(env.p.addUserDebugParameter("w3", -1, 1, 0))  # rear_left_wheel 24
    motorsIds.append(env.p.addUserDebugParameter("w4", -1, 1, 0))  # rear_right_wheel 25

    done = False
    while not done:
        action = []
        for motorId in motorsIds:
            action.append(env.p.readUserDebugParameter(motorId))

        state, reward, done, info = env.step(action)
        # obs = env.getExtendedObservation()
        qKey = ord('q')
        keys = p.getKeyboardEvents()
        if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
            env.close()
            break

if __name__ == "__main__":
    main()
