import os
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import pybullet_data
from Helper import Helper
from Dualcontrol import Dualcontrol
from pkg_resources import parse_version

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class Dualenv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, urdf_root=pybullet_data.getDataPath(), action_repeat=1,
                 is_enable_self_collision=True, renders=False,
                 is_discrete=False, max_steps=60000):
        self.in_pos = None
        self.is_discrete = is_discrete
        self._timeStep = 1. / 240
        self._urdfRoot = urdf_root
        self._actionRepeat = action_repeat
        self._isEnableSelfCollision = is_enable_self_collision
        self._observation = []
        self._renders = renders
        self._maxSteps = max_steps
        self._width = 341
        self._height = 256
        self.terminated = 0
        self.grasp = None
        self.p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.5, 130, -50, [0.52, -0.2, 0.])
        else:
            p.connect(p.DIRECT)
        self.hp = Helper()
        action_dim = 26  # action dimension
        self._action_bound = 1.0  # action range
        action_high = np.array([self._action_bound] * action_dim, dtype=np.float32)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        # observationDim = 25
        lower_observation = [-1.0] * 1
        upper_observation = [1.0] * 1
        self.observation_space = spaces.Box(low=np.array(lower_observation, dtype=np.float64),
                                            high=np.array(upper_observation, dtype=np.float64),
                                            dtype=np.float64)
        self.viewer = None
        self.seed()
        self.reset()

    def reset(self, **kwargs):
        p.resetSimulation()
        self.terminated = 0
        self.stage = 0
        self.in_pos = -1
        self.gl_error = 0.015
        self.near_error = 0.03
        self.out_of_range = 0
        self._envStepCounter = 0
        self._graspSuccess = 0
        self.object_slip = 0
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self._timeStep)
        #p.setPhysicsEngineParameter(numSolverIterations=150)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -0.15])
        self.info = self.hp.loadInfo()  # load object info from pos_all.csv
        self.index = self.info[0]  # object id
        self.grasp = self.info[8]  # grasp topology
        self.affordance = self.info[9]  # object affordance
        self.task_id = self.info[10] #task_id
        self.fail_reason = None
        # relative pos and orn between the hand and object
        self.p_rel, self.q_rel = self.hp.relative_pno(self.hp.p_origin, self.hp.q_origin,
                                                      self.info[1], self.info[2])
        # align link pos to center mass pos
        self.h_p_rel, self.h_q_rel = self.hp.relative_pno(self.hp.p_palm_cm, self.hp.q_origin,
                                                          self.hp.p_palm_lk, self.hp.q_origin)
        self.p_obj = self.info[5]  # object pos
        self.q_obj = self.info[6]  # object orientation
        angle = self.hp.rNum(self.info[3], self.info[4])  # random rotate the object
        # rotate the object round z axis by angle degree
        self.q_obj = self.hp.rotate_object(self.q_obj, angle, "z")
        self.object = p.loadURDF(os.path.join(parent_dir, self.info[7]), self.p_obj, self.q_obj,
                                 useFixedBase=0)  # load object
        self.p_new, self.q_new = self.hp.calculate_rigid_trans(self.p_obj, self.q_obj, self.p_rel,
                                                               self.q_rel)  # calculate new hand pos and orn
        self._dual = Dualcontrol(timeStep=self._timeStep, grasp=self.grasp, object_id=self.object)
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        d_s1 = 0.01
        d_s2 = 0.1
        # right hand
        x_right = action[0] * d_s1
        y_right = action[1] * d_s1
        z_right = action[2] * d_s1
        roll_right = action[3] * d_s2
        pitch_right = action[4] * d_s2
        yaw_right = action[5] * d_s2
        j_88 = action[6] * d_s2  # index right
        j_92 = action[7] * d_s2  # mid right
        j_105 = action[8] * d_s2  # ring right
        j_100 = action[9] * d_s2  # pinky right
        j_81 = action[10] * d_s2  # thumb right
        # left hand
        x_left = action[11] * d_s1
        y_left = action[12] * d_s1
        z_left = action[13] * d_s1
        roll_left = action[14] * d_s2
        pitch_left = action[15] * d_s2
        yaw_left = action[16] * d_s2
        j_42 = action[17] * d_s2  # index left
        j_46 = action[18] * d_s2  # mid left
        j_59 = action[19] * d_s2  # ring left
        j_54 = action[20] * d_s2  # pinky left
        j_35 = action[21] * d_s2  # thumb left
        # wheels
        j_w1 = action[22]
        j_w2 = action[23]
        j_w3 = action[24]
        j_w4 = action[25]
        realAction = [x_right, y_right, z_right, roll_right, pitch_right, yaw_right, j_88, j_92, j_105, j_100, j_81,
                      x_left, y_left, z_left, roll_left, pitch_left, yaw_left, j_42, j_46, j_59, j_54, j_35,
                      j_w1, j_w2, j_w3, j_w4]
        return self.step1(realAction)

    def step1(self, action):
        for i in range(self._actionRepeat):
            self._dual.applyAction_right(action)
            p.stepSimulation()
            self._dual.applyAction_left(action)
            p.stepSimulation()
            self._dual.applyAction_move_car(action)
            p.stepSimulation()
            if self._termination(action):
                break
            self._envStepCounter += 1
            if self._renders:
                time.sleep(self._timeStep)
        reward = self._reward()
        done = self._termination(action)
        self._observation = self.getExtendedObservation()
        return np.array(self._observation), reward, done, {}

    def getExtendedObservation(self):
        ################################################################################################
        # distance between the current pos of the right hand and the target
        dist = self.hp.distant(p.getLinkState(self._dual.dualUid, self.hp.dualEndEffectorIndex_right)[4], self.p_new)
        self._observation = [dist]
        return self._observation

    def _termination(self, action):
        joints = []
        for x in self.hp.r_joint_id:
            joints.append(p.getJointState(self._dual.dualUid, x)[0])

        if self._envStepCounter > self._maxSteps:
            self._observation = self.getExtendedObservation()
            print("stop due to time out")
            self.fail_reason = "time out"
            time.sleep(1)
            return True
        return False

    def _reward(self):
        p_hand = p.getLinkState(self._dual.dualUid, self.hp.dualEndEffectorIndex_right)[4]
        # distance reward
        dist = self.hp.distant(p_hand, self.p_new)
        reward = 1 / (dist + 1)  # max 1
        return reward

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        return self.getExtendedObservation()

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step

    ################################################# helper function #####################################

    def inPos(self, error):  # the grasp location is a range
        return self.s1_x(error) and self.s1_y(error) and self.s1_z(error)

    def s1_x(self, error):
        p_hand = p.getLinkState(self._dual.dualUid, self.hp.dualEndEffectorIndex_right)[4]
        return (p_hand[0] <= self.p_new[0] + error) and (p_hand[0] >= self.p_new[0] - error)

    def s1_y(self, error):
        p_hand = p.getLinkState(self._dual.dualUid, self.hp.dualEndEffectorIndex_right)[4]
        return (p_hand[1] <= self.p_new[1] + error) and (p_hand[1] >= self.p_new[1] - error)

    def s1_z(self, error):
        p_hand = p.getLinkState(self._dual.dualUid, self.hp.dualEndEffectorIndex_right)[4]
        return (p_hand[2] <= self.p_new[2] + error) and (p_hand[2] >= self.p_new[2] - error)

    def object_inPos(self):  # the object pos range
        obPos, _ = p.getBasePositionAndOrientation(self.object)
        x = (obPos[0] >= 0.72) and (obPos[0] <= 0.84)
        y = (obPos[1] >= -0.51) and (obPos[1] <= -0.40)
        return x and y

    def sus(self):
        return self._graspSuccess

    def pickup(self):
        contact = self.contactInfo(300)
        #print(contact)
        if self.grasp is None:
            return False
        if self.grasp == "inSiAd2":
            return sum(contact) >= 2
        if self.grasp == "pPdAb2":
            return contact[1] == 1 and contact[2] == 1
        if self.grasp == "pPdAb23":
            return contact[1] == 1 and contact[2] == 1 and sum(contact) >= 3
        if self.grasp == "pPdAb25":
            return sum(contact) >= 4
        if self.grasp == "poPmAb25":
            return sum(contact) >= 4

    def contactInfo(self, threshold=500):
        # boolean value 1 for read to pick up, 0 otherwise
        # thumb and index finger contact_points object
        limitForce = threshold
        contactParts = [0, 0, 0, 0, 0, 0]  # palm, thumb, index, middle, ring, pink
        palmLinks = self.hp.palmLinks
        thumbLinks = self.hp.thumbLinks
        indexLinks = self.hp.indexLinks
        middleLinks = self.hp.middleLinks
        ringLinks = self.hp.ringLinks
        pinkyLinks = self.hp.pinkyLinks
        # get contact information
        contact_points = p.getContactPoints(self._dual.dualUid, self.object)
        contact_num = len(contact_points)

        # find contact_points point
        # fill force and dist
        if contact_num > 0:
            for i in range(contact_num):
                if contact_points[i][3] in palmLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[0] = 1

                if contact_points[i][3] in thumbLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[1] = 1
                #print("thumb", contact_points[i][9] )

                if contact_points[i][3] in indexLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[2] = 1
                #print("index", contact_points[i][9] )
                if contact_points[i][3] in middleLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[3] = 1

                if contact_points[i][3] in ringLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[4] = 1

                if contact_points[i][3] in pinkyLinks:
                    if contact_points[i][9] >= limitForce:
                        contactParts[5] = 1
        return contactParts

    def observation_relatives(self, linkId):
        # convert link pos to center mass pos (palm)
        p_link = p.getLinkState(self._dual.dualUid, linkId)[4]
        q_link = p.getLinkState(self._dual.dualUid, linkId)[5]
        # get relative position between the hand and the target
        p_r, q_r = self.hp.relative_pno(self.p_new, self.q_new, p_link, q_link)
        q_r = p.getEulerFromQuaternion(q_r)
        p_rel = [p_r[0], p_r[1], p_r[2]]
        q_rel = [q_r[0], q_r[1], q_r[2]]
        return [p_rel, q_rel]

    def setup_rays_positions(self):  # 12 rays
        # pair 2 list without duplication
        ray_from, ray_to = [], []
        # 36 cross hit
        thumb_joints = [80] * 4 + [81] * 4 + [82] * 4
        finger_joints = [87, 88, 89, 91, 92, 93, 99, 100, 101, 104, 105, 106]
        for i in range(len(thumb_joints)):
            ray_from.append(p.getLinkState(self._dual.dualUid, thumb_joints[i])[4])
            ray_to.append(p.getLinkState(self._dual.dualUid, finger_joints[i])[4])
        return ray_from, ray_to

    def inGrasp(self):
        # check if the object is in grasp
        ray_from, ray_to = self.setup_rays_positions()
        readings = p.rayTestBatch(ray_from, ray_to)
        object_contact = 0
        if len(readings) > 0:
            for i in range(len(readings)):
                if readings[i][0] == self.object:
                    object_contact += 1
            if object_contact > 0:
                return True
        return False

    def check_equilibrium(self):

        force_total = np.array([0, 0, 0], dtype=np.float64)
        torque_total = np.array([0, 0, 0], dtype=np.float64)
        contact_points = p.getContactPoints(self._dual.dualUid, self.object)
        if len(contact_points) > 0:
            for contact in contact_points:
                # position vecc for torque
                object_com_position, _ = p.getBasePositionAndOrientation(self.object)
                object_com_position = np.array(object_com_position)
                contact_pos = np.array(contact[6], dtype=np.float64)
                c_com = contact_pos - object_com_position
                # forces and directions
                norm_force = np.array(contact[9], dtype=np.float64)  # 1
                norm_vec = np.array(contact[7], dtype=np.float64)  # 3
                lateral1 = np.array(contact[10])  # 1
                lateral1_vec = np.array(contact[11], dtype=np.float64)  # 3
                lateral2 = np.array(contact[12], dtype=np.float64)  # 1
                lateral2_vec = np.array(contact[13], dtype=np.float64)  # 3
                # forces
                norm = norm_force * norm_vec
                lateral_1 = lateral1 * lateral1_vec
                lateral_2 = lateral2 * lateral2_vec
                # force balance
                force_total += norm
                force_total += lateral_1
                force_total += lateral_2
                # torque balance
                torque_total += np.cross(c_com, norm)
                torque_total += np.cross(c_com, lateral_1)
                torque_total += np.cross(c_com, lateral_2)
            return force_total, torque_total
        else:
            return (np.array([3.3, 3.3, 3.3], dtype=np.float64),
                    np.array([3.3, 3.3, 3.3], dtype=np.float64))

    def in_friction_cone(self, friction_coefficient=0.6):
        """
        Check if all contact forces between two bodies are within the friction cone.
        Parameters:
        friction_coefficient (float): Coefficient of friction at the contact points.
        Returns:
        bool: True if all contact forces are within the friction cone, False otherwise.
        """
        # Get all contact points between the two bodies
        contact_points = p.getContactPoints(self._dual.dualUid, self.object)
        for contact in contact_points:
            # Extract normal force (force along the contact normal)
            normal_force = contact[9]  # index 9 is normal force in contact information
            # Extract lateral forces
            lateral_friction1 = contact[10]  # Lateral friction force along the first direction
            lateral_friction2 = contact[12]  # Lateral friction force along the second direction

            # Compute the tangential (lateral) force magnitude
            tangential_force = np.sqrt(lateral_friction1 ** 2 + lateral_friction2 ** 2)

            # Check if the tangential force is within the friction cone
            if tangential_force > friction_coefficient * normal_force:
                # If any tangential force exceeds the friction cone limit, return False
                return False
        # If all contact points satisfy the friction cone condition, return True
        return True

