import os
import time

import pybullet as p
from Helper import Helper

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print(parent_dir)
class Dualcontrol:

    def __init__(self, timeStep=0.01, grasp='relax', object_id=-1):

        self.move2pos = 0
        self.table = None
        self.wait_grasp = None
        self.endEffectorPos_right = None
        self.endEffectorPos_left = None

        self.timeStep = timeStep
        self.object_id = object_id
        ##################### Joint related info  ############################
        self.hp = Helper()
        self.topology = grasp
        self.grasp = self.hp.grasp_pose[grasp]
        self.arm_pos = self.hp.arm_pos
        self.dualEndEffectorIndex_right = self.hp.dualEndEffectorIndex_right
        self.dualEndEffectorIndex_left = self.hp.dualEndEffectorIndex_left
        self.endEffectorOrn_right = None
        self.endEffectorOrn_left = None
        self.max_force = self.hp.max_force
        self.joint_damp = self.hp.joint_damp
        self.r_joint_id = self.hp.r_joint_id
        #self.l_joint_id = self.hp.l_joint_id
        self.lower_limit = self.hp.lower_limit
        self.upper_limit = self.hp.upper_limit
        self.joint_range = self.hp.joint_range
        self.max_velocity = self.hp.max_velocity
        self.hand_maxForce = self.hp.hand_maxForce
        #self.rest_pos = self.arm_pos + self.hp.grasp_pose["pPdAb23"]
        self.rest_pos = self.arm_pos + self.hp.grasp_pose["relax"]
        #self.rest_pos = self.arm_pos + [0]*19
        self.dualUid = -100
        self.finger_initial_r = None
        self.finger_initial_l = None
        self.final_index = None
        self.final_mid = None
        self.final_ring = None
        self.final_pinky = None
        self.final_thumb = None
        self.reset()

    def reset(self):
        self.wait_grasp = 0
        self.final_index = -1
        self.final_mid = -1
        self.final_ring = -1
        self.final_pinky = -1
        self.final_thumb = -1
        self.endEffectorPos_right = self.hp.endEffectorPos_right
        self.endEffectorPos_left = self.hp.endEffectorPos_left
        self.endEffectorOrn_right = self.hp.q_origin
        self.endEffectorOrn_left = self.hp.q_origin
        self.finger_initial_r = [self.grasp[6], self.grasp[9], self.grasp[17], self.grasp[13], self.grasp[2]]
        self.finger_initial_l = [self.grasp[6], self.grasp[9], self.grasp[17], self.grasp[13], self.grasp[2]]
        self.dualUid = p.loadURDF(os.path.join(parent_dir, "robots/dual_2hand.urdf"), useFixedBase=0)
        p.resetBasePositionAndOrientation(self.dualUid, self.hp.p_origin, self.hp.q_origin)
        self.table = p.loadURDF(os.path.join(parent_dir, "robots/table/table.urdf"), self.hp.p_table, self.hp.q_origin,
                                useFixedBase=1)
        for i in range(len(self.r_joint_id)):
            p.resetJointState(self.dualUid, self.r_joint_id[i], self.rest_pos[i])
            p.setJointMotorControl2(self.dualUid, jointIndex=self.r_joint_id[i], controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.rest_pos[i], targetVelocity=0, force=self.max_force[i],
                                    maxVelocity=self.max_velocity[i], positionGain=0.03, velocityGain=1)

    def applyAction_right(self, action):
        # if self.orn_right == 1:
        #     self.endEffectorOrn_right = [0,0,0,1]
        #     self.orn_right == 0
        ############################################# actions ###################################################
        x_right = action[0]
        y_right = action[1]
        z_right = action[2]
        roll_right = action[3]
        pitch_right = action[4]
        yaw_right = action[5]
        j_88 = action[6]   # index right
        j_92 = action[7]   # mid right
        j_105 = action[8]  # ring right
        j_100 = action[9]  # pinky right
        j_81 = action[10]  # thumb right

        ############################################# control arm ###################################################

        # move right hand
        self.move_arm(self.dualEndEffectorIndex_right, self.endEffectorPos_right, self.endEffectorOrn_right)
        # define moving range of right arm
        self.endEffectorPos_right[0] = self.endEffectorPos_right[0] + x_right
        if self.endEffectorPos_right[0] > 0.9:
            self.endEffectorPos_right[0] = 0.9
        if self.endEffectorPos_right[0] < 0.6:
            self.endEffectorPos_right[0] = 0.6

        self.endEffectorPos_right[1] = self.endEffectorPos_right[1] + y_right
        if self.endEffectorPos_right[1] <= -0.7:
            self.endEffectorPos_right[1] = -0.7
        if self.endEffectorPos_right[1] >= -0.2:
            self.endEffectorPos_right[1] = -0.2

        self.endEffectorPos_right[2] = self.endEffectorPos_right[2] + z_right
        if self.endEffectorPos_right[2] >= 0.7:
            self.endEffectorPos_right[2] = 0.7
        if self.endEffectorPos_right[2] <= 0.2:
            self.endEffectorPos_right[2] = 0.2

        self.endEffectorOrn_right = list(p.getEulerFromQuaternion(self.endEffectorOrn_right))
        self.endEffectorOrn_right[0] = self.endEffectorOrn_right[0] + roll_right
        self.endEffectorOrn_right[1] = self.endEffectorOrn_right[1] + pitch_right
        self.endEffectorOrn_right[2] = self.endEffectorOrn_right[2] + yaw_right
        self.endEffectorOrn_right = p.getQuaternionFromEuler(self.endEffectorOrn_right)

        ############################################# control hand ###############################################
        # index
        self.move_finger_joint(86, 0)
        self.finger_model_index_r(j_88)
        # mid
        self.finger_model_mid_r(j_92)
        # ring
        self.move_finger_joint(103, 0)
        self.finger_model_ring_r(j_105)
        # pinky
        self.move_finger_joint(98, 0)
        self.finger_model_pinky_r(j_100)
        # thumb
        self.finger_model_thumb_r(j_81)
        self.move_finger_joint(78, self.grasp[6])

        # can also control each joint of the hand
        # joint id included in dual_joint_info.xlsx
        # self.move_finger_joint(78, self.grasp[6])
        # self.move_finger_joint(80, self.hp.relax[1])
        # self.move_finger_joint(81, self.hp.relax[2])
        # self.move_finger_joint(82, self.hp.relax[3])
        # self.move_finger_joint(86, self.hp.relax[4])
        # self.move_finger_joint(87, self.hp.relax[5])
        # self.move_finger_joint(88, self.hp.relax[6])
        # self.move_finger_joint(89, self.hp.relax[7])
        # self.move_finger_joint(91, self.hp.relax[8])
        # self.move_finger_joint(92, self.hp.relax[9])
        # self.move_finger_joint(93, self.hp.relax[10])
        # self.move_finger_joint(98, self.hp.relax[11])
        # self.move_finger_joint(99, self.hp.relax[12])
        # self.move_finger_joint(100, self.hp.relax[13])
        # self.move_finger_joint(101, self.hp.relax[14])
        # self.move_finger_joint(103, self.hp.relax[15])
        # self.move_finger_joint(104, self.hp.relax[16])
        # self.move_finger_joint(105, self.hp.relax[17])
        # self.move_finger_joint(106, self.hp.relax[18])

    def applyAction_left(self, action):

        #q_new = self.mirror_orientation(q_new)
        ############################################# actions ###################################################
        x_left = action[11]
        y_left = action[12]
        z_left = action[13]
        roll_left = action[14]
        pitch_left = action[15]
        yaw_left = action[16]
        j_42 = action[17]  # index left
        j_46 = action[18]  # mid left
        j_59 = action[19]  # ring left
        j_54 = action[20]  # pinky left
        j_35 = action[21]  # thumb left
        ############################################# control arm ###################################################
        # move left hand
        self.move_arm(self.dualEndEffectorIndex_left, self.endEffectorPos_left, self.endEffectorOrn_left)
        # define moving range of left arm
        self.endEffectorPos_left[0] = self.endEffectorPos_left[0] + x_left
        if self.endEffectorPos_left[0] > 0.9:
            self.endEffectorPos_left[0] = 0.9
        if self.endEffectorPos_left[0] < 0.6:
            self.endEffectorPos_left[0] = 0.6

        self.endEffectorPos_left[1] = self.endEffectorPos_left[1] + y_left
        if self.endEffectorPos_left[1] >= 0.75:
            self.endEffectorPos_left[1] = 0.75
        if self.endEffectorPos_left[1] <= 0.2:
            self.endEffectorPos_left[1] = 0.2

        self.endEffectorPos_left[2] = self.endEffectorPos_left[2] + z_left
        if self.endEffectorPos_left[2] >= 0.7:
            self.endEffectorPos_left[2] = 0.7
        if self.endEffectorPos_left[2] <= 0.2:
            self.endEffectorPos_left[2] = 0.2

        self.endEffectorOrn_left = list(p.getEulerFromQuaternion(self.endEffectorOrn_left))
        self.endEffectorOrn_left[0] = self.endEffectorOrn_left[0] + roll_left
        self.endEffectorOrn_left[1] = self.endEffectorOrn_left[1] + pitch_left
        self.endEffectorOrn_left[2] = self.endEffectorOrn_left[2] + yaw_left
        self.endEffectorOrn_left = p.getQuaternionFromEuler(self.endEffectorOrn_left)

        ############################################# control hand ###############################################
        # index
        self.move_finger_joint(40, 0)
        self.finger_model_index_l(j_42)
        # mid
        self.finger_model_mid_l(j_46)
        # ring
        self.move_finger_joint(57, 0)
        self.finger_model_ring_l(j_59)
        # pinky
        self.move_finger_joint(52, 0)
        self.finger_model_pinky_l(j_54)
        # thumb
        self.finger_model_thumb_l(j_35)
        self.move_finger_joint(32, self.grasp[6])

    def applyAction_move_car(self, action):

        w1 = action[22]
        w2 = action[23]
        w3 = action[24]
        w4 = action[25]
        wheel_id = self.hp.wheel_joints
        self.move_wheel(wheel_id[0], w1)
        self.move_wheel(wheel_id[1], w1)
        self.move_wheel(wheel_id[2], w1)
        self.move_wheel(wheel_id[3], w1)

    ############################################# helper functions ################################################

    def move_wheel(self, joint_id, velocity):
        p.setJointMotorControl2(bodyUniqueId=self.dualUid, jointIndex=joint_id, controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=velocity, force=1000.0)

    def move_finger_joint(self, joint_id, target_position):

        p.setJointMotorControl2(self.dualUid, jointIndex=joint_id, controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position, targetVelocity=0, force=self.hand_maxForce,
                                maxVelocity=1, positionGain=0.03, velocityGain=1)

    def move_arm_joint(self, joint_id, target_position):

        p.setJointMotorControl2(self.dualUid, jointIndex=joint_id, controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position, targetVelocity=0, force=self.hand_maxForce,
                                maxVelocity=0.3, positionGain=0.03, velocityGain=1)

    def move_arm(self, end_effector_id, target_position, target_orientation):

        joint_poses = p.calculateInverseKinematics(self.dualUid, end_effector_id, target_position,
                                                       target_orientation, lowerLimits=self.lower_limit,
                                                       upperLimits=self.upper_limit, jointRanges=self.joint_range,
                                                       restPoses=self.rest_pos)
        for i in range(len(self.r_joint_id)):
            p.setJointMotorControl2(self.dualUid, jointIndex=self.r_joint_id[i], controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_poses[i], targetVelocity=0, force=self.max_force[i],
                                    maxVelocity=1000, positionGain=0.3, velocityGain=0.3)

    def move_down(self, end_effector_id, target_position, target_orientation):
        jointPoses = p.calculateInverseKinematics(self.dualUid, end_effector_id, target_position, target_orientation,
                                                  lowerLimits=self.lower_limit, upperLimits=self.upper_limit,
                                                  jointRanges=self.joint_range, restPoses=self.rest_pos)

        for i in range(len(self.r_joint_id)):
            p.setJointMotorControl2(self.dualUid, jointIndex=self.r_joint_id[i], controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], targetVelocity=0, force=self.max_force[i],
                                    maxVelocity=0.1, positionGain=0.3, velocityGain=0.3)

    # θ_TDIP ≈ 0.5⋅θ_TMCP
    # Index: θDIP = 0.77⋅θPIP
    # Middle: θDIP = 0.75⋅θPIP
    # Ring: θDIP = 0.75⋅θPIP
    # Little: θDIP = 0.57⋅θPIP
    # θMCP = (0.53-0.71)⋅θPIP    (θPIP = (1.4 − 1.9)⋅θMCP)
    # thumb flexion and opposition move freely
    # < initial angle = initial angle
    # if MCP contact point: pip, dip keep going (flexing)
    # if pip contact point: no effect
    # if dip contact point: all keep moving
    # if reach force threshold: stop

    def thumb_model(self, pip):
        # pip is angle of PIP
        # thumb joints in [PIP, DIP] format
        if pip <= 0.98506:  # upper limit of pip thumb
            dip = self.hp.thumb_alpha_dip * pip
        else:
            dip = pip
        self.move_finger_joint(self.hp.thumb_joint[1], pip)
        self.move_finger_joint(self.hp.thumb_joint[2], dip)

    def check_contact_points(self, joint_id, threshold=200):
        contacts = p.getContactPoints(self.object_id, self.dualUid, -1, joint_id)
        return any(contact[9] >= threshold for contact in contacts)

    def set_self_collision(self):
        for thumb_link in self.hp.thumb_joint:
            for finger_link in self.hp.finger_joints:
                if thumb_link != finger_link:
                    p.setCollisionFilterPair(self.dualUid, self.dualUid, thumb_link, finger_link,
                                             enableCollision=1)

    def check_finger_collision(self, threshold=400):
        for thumb_link in self.hp.thumb_joint_r:
            for finger_link in self.hp.finger_joints:
                if thumb_link != finger_link:
                    contacts = p.getContactPoints(self.dualUid, self.dualUid, thumb_link, finger_link)
                    if any(contact[9] >= threshold for contact in contacts):
                        return True
        return False

    def finger_model_index_r(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [87, 88, 89]
        # joint range [0.79849, 1.334, 1.394]
        # θDIP = 0.77⋅θPIP; max_dip = 1.394/0.77 = 1.81039
        # θMCP = 0.67⋅θPIP; max_mcp = 0.79849/0.67 = 1.192
        # pip is angle of PIP
        max_mcp = self.hp.index_joint_max[0] / self.hp.index_alpha_mcp
        max_dip = self.hp.index_joint_max[2] / self.hp.index_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial_r[0] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial_r[0] = max_pip
        if self.finger_initial_r[0] < 0:
            self.finger_initial_r[0] = 0
        self.finger_initial_r[0] += delta_pip
        mcp = min(self.hp.index_joint_max[0], self.hp.index_alpha_mcp * self.finger_initial_r[0])
        dip = min(self.hp.index_joint_max[2], self.hp.index_alpha_dip * self.finger_initial_r[0])
        pip = self.finger_initial_r[0]
        mcp_contact = self.check_contact_points(self.hp.index_joint_r[0])
        pip_contact = self.check_contact_points(self.hp.index_joint_r[1])
        dip_contact = self.check_contact_points(self.hp.index_joint_r[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.index_joint_r[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.index_joint_r[1], pip)
                self.move_finger_joint(self.hp.index_joint_r[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.index_joint_r[0], mcp)
                self.move_finger_joint(self.hp.index_joint_r[1], pip)
                self.move_finger_joint(self.hp.index_joint_r[2], dip)

    def finger_model_index_l(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [87, 88, 89]
        # joint range [0.79849, 1.334, 1.394]
        # θDIP = 0.77⋅θPIP; max_dip = 1.394/0.77 = 1.81039
        # θMCP = 0.67⋅θPIP; max_mcp = 0.79849/0.67 = 1.192
        # pip is angle of PIP
        max_mcp = self.hp.index_joint_max[0] / self.hp.index_alpha_mcp
        max_dip = self.hp.index_joint_max[2] / self.hp.index_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial_l[0] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial_l[0] = max_pip
        if self.finger_initial_l[0] < 0:
            self.finger_initial_l[0] = 0
        self.finger_initial_l[0] += delta_pip
        mcp = min(self.hp.index_joint_max[0], self.hp.index_alpha_mcp * self.finger_initial_l[0])
        dip = min(self.hp.index_joint_max[2], self.hp.index_alpha_dip * self.finger_initial_l[0])
        pip = self.finger_initial_l[0]
        mcp_contact = self.check_contact_points(self.hp.index_joint_l[0])
        pip_contact = self.check_contact_points(self.hp.index_joint_l[1])
        dip_contact = self.check_contact_points(self.hp.index_joint_l[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.index_joint_l[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.index_joint_l[1], pip)
                self.move_finger_joint(self.hp.index_joint_l[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.index_joint_l[0], mcp)
                self.move_finger_joint(self.hp.index_joint_l[1], pip)
                self.move_finger_joint(self.hp.index_joint_l[2], dip)

    def finger_model_mid_r(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [91, 92, 93]
        # joint range [0.79849, 1.334, 1.334]
        # θDIP = 0.75⋅θPIP; max_dip = 1.334/0.75 = 1.77867
        # θMCP = 0.67⋅θPIP; max_mcp = 0.79849/0.67 = 1.19178
        # pip is angle of PIP
        max_mcp = self.hp.mid_joint_max[0] / self.hp.mid_alpha_mcp
        max_dip = self.hp.mid_joint_max[2] / self.hp.mid_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial_r[1] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial_r[1] = max_pip
        if self.finger_initial_r[1] < 0:
            self.finger_initial_r[1] = 0
        self.finger_initial_r[1] += delta_pip
        mcp = min(self.hp.mid_joint_max[0], self.hp.mid_alpha_mcp * self.finger_initial_r[1])
        dip = min(self.hp.mid_joint_max[2], self.hp.mid_alpha_dip * self.finger_initial_r[1])
        pip = self.finger_initial_r[1]
        mcp_contact = self.check_contact_points(self.hp.mid_joint_r[0])
        pip_contact = self.check_contact_points(self.hp.mid_joint_r[1])
        dip_contact = self.check_contact_points(self.hp.mid_joint_r[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.mid_joint_r[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.mid_joint_r[1], pip)
                self.move_finger_joint(self.hp.mid_joint_r[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.mid_joint_r[0], mcp)
                self.move_finger_joint(self.hp.mid_joint_r[1], pip)
                self.move_finger_joint(self.hp.mid_joint_r[2], dip)

    def finger_model_mid_l(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [91, 92, 93]
        # joint range [0.79849, 1.334, 1.334]
        # θDIP = 0.75⋅θPIP; max_dip = 1.334/0.75 = 1.77867
        # θMCP = 0.67⋅θPIP; max_mcp = 0.79849/0.67 = 1.19178
        # pip is angle of PIP
        max_mcp = self.hp.mid_joint_max[0] / self.hp.mid_alpha_mcp
        max_dip = self.hp.mid_joint_max[2] / self.hp.mid_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial_l[1] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial_l[1] = max_pip
        if self.finger_initial_l[1] < 0:
            self.finger_initial_l[1] = 0
        self.finger_initial_l[1] += delta_pip
        mcp = min(self.hp.mid_joint_max[0], self.hp.mid_alpha_mcp * self.finger_initial_l[1])
        dip = min(self.hp.mid_joint_max[2], self.hp.mid_alpha_dip * self.finger_initial_l[1])
        pip = self.finger_initial_l[1]
        mcp_contact = self.check_contact_points(self.hp.mid_joint_l[0])
        pip_contact = self.check_contact_points(self.hp.mid_joint_l[1])
        dip_contact = self.check_contact_points(self.hp.mid_joint_l[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.mid_joint_l[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.mid_joint_l[1], pip)
                self.move_finger_joint(self.hp.mid_joint_l[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.mid_joint_l[0], mcp)
                self.move_finger_joint(self.hp.mid_joint_l[1], pip)
                self.move_finger_joint(self.hp.mid_joint_l[2], dip)

    def finger_model_ring_r(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [104, 105, 106]
        # joint range [0.98175, 1.334, 1.395]
        # θDIP = 0.75⋅θPIP; max_dip = 1.395/0.57 = 2.44737
        # θMCP = 0.67⋅θPIP; max_mcp = 0.98175/0.67 = 1.4653
        # pip is angle of PIP
        max_mcp = self.hp.ring_joint_max[0] / self.hp.ring_alpha_mcp
        max_dip = self.hp.ring_joint_max[2] / self.hp.ring_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial_r[2] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial_r[2] = max_pip
        if self.finger_initial_r[2] < 0:
            self.finger_initial_r[2] = 0
        self.finger_initial_r[2] += delta_pip
        mcp = min(self.hp.ring_joint_max[0], self.hp.ring_alpha_mcp * self.finger_initial_r[2])
        dip = min(self.hp.ring_joint_max[2], self.hp.ring_alpha_dip * self.finger_initial_r[2])
        pip = self.finger_initial_r[2]
        mcp_contact = self.check_contact_points(self.hp.ring_joint_r[0])
        pip_contact = self.check_contact_points(self.hp.ring_joint_r[1])
        dip_contact = self.check_contact_points(self.hp.ring_joint_r[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.ring_joint_r[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.ring_joint_r[1], pip)
                self.move_finger_joint(self.hp.ring_joint_r[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.ring_joint_r[0], mcp)
                self.move_finger_joint(self.hp.ring_joint_r[1], pip)
                self.move_finger_joint(self.hp.ring_joint_r[2], dip)

    def finger_model_ring_l(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [104, 105, 106]
        # joint range [0.98175, 1.334, 1.395]
        # θDIP = 0.75⋅θPIP; max_dip = 1.395/0.57 = 2.44737
        # θMCP = 0.67⋅θPIP; max_mcp = 0.98175/0.67 = 1.4653
        # pip is angle of PIP
        max_mcp = self.hp.ring_joint_max[0] / self.hp.ring_alpha_mcp
        max_dip = self.hp.ring_joint_max[2] / self.hp.ring_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial_l[2] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial_l[2] = max_pip
        if self.finger_initial_l[2] < 0:
            self.finger_initial_l[2] = 0
        self.finger_initial_l[2] += delta_pip
        mcp = min(self.hp.ring_joint_max[0], self.hp.ring_alpha_mcp * self.finger_initial_l[2])
        dip = min(self.hp.ring_joint_max[2], self.hp.ring_alpha_dip * self.finger_initial_l[2])
        pip = self.finger_initial_l[2]
        mcp_contact = self.check_contact_points(self.hp.ring_joint_l[0])
        pip_contact = self.check_contact_points(self.hp.ring_joint_l[1])
        dip_contact = self.check_contact_points(self.hp.ring_joint_l[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.ring_joint_l[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.ring_joint_l[1], pip)
                self.move_finger_joint(self.hp.ring_joint_l[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.ring_joint_l[0], mcp)
                self.move_finger_joint(self.hp.ring_joint_l[1], pip)
                self.move_finger_joint(self.hp.ring_joint_l[2], dip)

    def finger_model_pinky_r(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [99, 100, 101]
        # joint range [0.98175, 1.334, 1.3971]
        # θDIP = 0.57⋅θPIP; max_dip = 1.395/0.57 = 2.44737
        # θMCP = 0.67⋅θPIP; max_mcp = 0.98175/0.67 = 1.4653
        # pip is angle of PIP
        max_mcp = self.hp.pinky_joint_max[0] / self.hp.pinky_alpha_mcp
        max_dip = self.hp.pinky_joint_max[2] / self.hp.pinky_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial_r[3] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial_r[3] = max_pip
        if self.finger_initial_r[3] < 0:
            self.finger_initial_r[3] = 0
        self.finger_initial_r[3] += delta_pip
        mcp = min(self.hp.pinky_joint_max[0], self.hp.pinky_alpha_mcp * self.finger_initial_r[3])
        dip = min(self.hp.pinky_joint_max[2], self.hp.pinky_alpha_dip * self.finger_initial_r[3])
        pip = self.finger_initial_r[3]
        mcp_contact = self.check_contact_points(self.hp.pinky_joint_r[0])
        pip_contact = self.check_contact_points(self.hp.pinky_joint_r[1])
        dip_contact = self.check_contact_points(self.hp.pinky_joint_r[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.pinky_joint_r[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.pinky_joint_r[1], pip)
                self.move_finger_joint(self.hp.pinky_joint_r[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.pinky_joint_r[0], mcp)
                self.move_finger_joint(self.hp.pinky_joint_r[1], pip)
                self.move_finger_joint(self.hp.pinky_joint_r[2], dip)

    def finger_model_pinky_l(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [99, 100, 101]
        # joint range [0.98175, 1.334, 1.3971]
        # θDIP = 0.57⋅θPIP; max_dip = 1.395/0.57 = 2.44737
        # θMCP = 0.67⋅θPIP; max_mcp = 0.98175/0.67 = 1.4653
        # pip is angle of PIP
        max_mcp = self.hp.pinky_joint_max[0] / self.hp.pinky_alpha_mcp
        max_dip = self.hp.pinky_joint_max[2] / self.hp.pinky_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial_l[3] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial_l[3] = max_pip
        if self.finger_initial_l[3] < 0:
            self.finger_initial_l[3] = 0
        self.finger_initial_l[3] += delta_pip
        mcp = min(self.hp.pinky_joint_max[0], self.hp.pinky_alpha_mcp * self.finger_initial_l[3])
        dip = min(self.hp.pinky_joint_max[2], self.hp.pinky_alpha_dip * self.finger_initial_l[3])
        pip = self.finger_initial_l[3]
        mcp_contact = self.check_contact_points(self.hp.pinky_joint_l[0])
        pip_contact = self.check_contact_points(self.hp.pinky_joint_l[1])
        dip_contact = self.check_contact_points(self.hp.pinky_joint_l[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.pinky_joint_l[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.pinky_joint_l[1], pip)
                self.move_finger_joint(self.hp.pinky_joint_l[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.pinky_joint_l[0], mcp)
                self.move_finger_joint(self.hp.pinky_joint_l[1], pip)
                self.move_finger_joint(self.hp.pinky_joint_l[2], dip)

    def finger_model_thumb_r(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [80, 81, 82]
        # joint range [0.9704, 0.98506, 1.406]
        # pip is angle of PIP
        max_mcp = self.hp.thumb_joint_max[0] / self.hp.thumb_alpha_mcp
        max_dip = self.hp.thumb_joint_max[2] / self.hp.thumb_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial_r[4] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial_r[4] = max_pip
        if self.finger_initial_r[4] < 0:
            self.finger_initial_r[4] = 0
        self.finger_initial_r[4] += delta_pip
        mcp = min(self.hp.thumb_joint_max[0], self.hp.thumb_alpha_mcp * self.finger_initial_r[4])
        dip = min(self.hp.thumb_joint_max[2], self.hp.thumb_alpha_dip * self.finger_initial_r[4])
        pip = self.finger_initial_r[4]
        mcp_contact = self.check_contact_points(self.hp.thumb_joint_r[0])
        pip_contact = self.check_contact_points(self.hp.thumb_joint_r[1])
        dip_contact = self.check_contact_points(self.hp.thumb_joint_r[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.thumb_joint_r[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.thumb_joint_r[1], pip)
                self.move_finger_joint(self.hp.thumb_joint_r[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.thumb_joint_r[0], mcp)
                self.move_finger_joint(self.hp.thumb_joint_r[1], pip)
                self.move_finger_joint(self.hp.thumb_joint_r[2], dip)

    def finger_model_thumb_l(self, delta_pip):
        # joint order [MCP, PIP, DIP]
        # joint id    [80, 81, 82]
        # joint range [0.9704, 0.98506, 1.406]
        # pip is angle of PIP
        max_mcp = self.hp.thumb_joint_max[0] / self.hp.thumb_alpha_mcp
        max_dip = self.hp.thumb_joint_max[2] / self.hp.thumb_alpha_dip
        max_pip = max(max_mcp, max_dip)
        if self.finger_initial_l[4] >= max_pip:  # suppose to be value of joint 88
            self.finger_initial_l[4] = max_pip
        if self.finger_initial_l[4] < 0:
            self.finger_initial_l[4] = 0
        self.finger_initial_l[4] += delta_pip
        mcp = min(self.hp.thumb_joint_max[0], self.hp.thumb_alpha_mcp * self.finger_initial_l[4])
        dip = min(self.hp.thumb_joint_max[2], self.hp.thumb_alpha_dip * self.finger_initial_l[4])
        pip = self.finger_initial_l[4]
        mcp_contact = self.check_contact_points(self.hp.thumb_joint_l[0])
        pip_contact = self.check_contact_points(self.hp.thumb_joint_l[1])
        dip_contact = self.check_contact_points(self.hp.thumb_joint_l[2])
        if not self.check_finger_collision():
            if dip_contact:
                # Stop all joints
                pass
            elif pip_contact:
                # Move only DIP joint
                self.move_finger_joint(self.hp.thumb_joint_l[2], dip)
            elif mcp_contact:
                # Move PIP and DIP joints
                self.move_finger_joint(self.hp.thumb_joint_l[1], pip)
                self.move_finger_joint(self.hp.thumb_joint_l[2], dip)
            else:
                # Move all joints
                self.move_finger_joint(self.hp.thumb_joint_l[0], mcp)
                self.move_finger_joint(self.hp.thumb_joint_l[1], pip)
                self.move_finger_joint(self.hp.thumb_joint_l[2], dip)

    def mirror_orientation(self, quaternion):
        x, y, z, w = quaternion
        # Negate x and w components to mirror across YZ plane
        return [x, -y, z, -w]