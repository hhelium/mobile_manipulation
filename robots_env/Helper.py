import ast
import math
import time
import random
import numpy as np
import pandas as pd

class Helper:

    def __init__(self):

        ########################################## robot configurations ##############################################

        self.p_origin = np.array([0, 0, 0])
        self.q_origin = np.array([0, 0, 0, 1])

        # palm mass center pos
        self.p_palm_cm = np.array([0.7002149979151765, -0.7013460020218656, 0.5994980314648282])
        # palm link center pos
        self.p_palm_lk = np.array([0.7, -0.7, 0.7])

        # position of the table, orientation is self.q_origin
        self.p_table = np.array([0.79, -0.455, -0.5])

        self.dualEndEffectorIndex_right = 90
        self.dualEndEffectorIndex_left = 44

        self.endEffectorPos_right = np.array([0.65, -0.7, 0.6])
        self.endEffectorPos_left = np.array([0.65, 0.7, 0.6])

        # max force of hand joint
        self.hand_maxForce = 1000

        # joint damp for each revolute joint of the robot
        self.joint_damp = [0] * 56

        # all revolute joints
        self.r_joint_id = [3, 4, 5, 6, 13, 14,  # husky [0,5]
                           19, 20, 21, 22, 23, 24,  # left arm [6, 11]
                           32, 34, 35, 36, 40, 41, 42, 43, 45, 46, 47, 52, 53, 54, 55, 57, 58, 59, 60,  # left hand
                           65, 66, 67, 68, 69, 70,  # right arm [31-36]
                           78, 80, 81, 82, 86, 87, 88, 89, 91, 92, 93, 98, 99, 100, 101, 103, 104, 105, 106]

        # lower limit of each revolute joint
        self.lower_limit = [0, 0, 0, 0, -2.775, -0.82, -2 * math.pi, -2 * math.pi, -math.pi, -2 * math.pi, -2 * math.pi,
                            -2 * math.pi, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, -2 * math.pi, -2 * math.pi, -math.pi, -2 * math.pi, -2 * math.pi, -2 * math.pi, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # upper limit of each revolute joint
        self.upper_limit = [-1, -1, -1, -1, 2.775, 0.52, 2 * math.pi, 2 * math.pi, math.pi, 2 * math.pi, 2 * math.pi,
                            2 * math.pi, 0.9879, 0.9704, 0.98506, 1.406, 0.28833, 0.79849, 1.334, 1.394,
                            0.79849, 1.334, 1.334, 0.5829, 0.98175, 1.334, 1.3971, 0.28833, 0.98175, 1.334, 1.395,
                            2 * math.pi, 2 * math.pi, math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi, 0.9879, 0.9704,
                            0.98506,
                            1.406, 0.28833, 0.79849, 1.334, 1.394, 0.79849, 1.334, 1.334, 0.5829, 0.98175, 1.334,
                            1.3971, 0.28833, 0.98175, 1.334, 1.395]

        # movement range of each revolute joint
        self.joint_range = [-1, -1, -1, -1, 5.5, 1.34, 2 * math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi,
                            4 * math.pi, 0.9879, 0.9704, 0.98506, 1.406, 0.28833, 0.79849, 1.334, 1.394,
                            0.79849, 1.334, 1.334, 0.5829, 0.98175, 1.334, 1.3971, 0.28833, 0.98175, 1.334, 1.395,
                            2 * math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi, 0.9879,
                            0.9704, 0.98506,
                            1.406, 0.28833, 0.79849, 1.334, 1.394, 0.79849, 1.334, 1.334, 0.5829, 0.98175, 1.334,
                            1.3971, 0.28833, 0.98175, 1.334, 1.395]

        # max force of each revolute joint
        self.max_force = [0, 0, 0, 0, 30, 30, 150, 150, 150, 28, 28, 28, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                          1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                          150, 150, 150, 28, 28, 28, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                          1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

        # self.max_force = [0, 0, 0, 0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
        #                   1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
        #                   1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
        #                   1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

        self.max_force = [0, 0, 0, 0, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000,
                          100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000,
                          100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000,
                          100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]

        # max velocity of each revolute joint
        self.max_velocity = [0, 0, 0, 0, 1, 1, math.pi, math.pi, 3.15, math.pi, math.pi, math.pi, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, math.pi, math.pi, 3.15, math.pi,
                             math.pi, math.pi, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # hand link index
        self.palmLinks = [77, 94, 95, 96, 97]  # link right_hand_[e1, e2, e3, e, e4]
        self.thumbLinks = [78, 80, 81, 82, 79]  # right_hand_[z,a,b,c] and right_hand_virtual_a
        self.indexLinks = [87, 88, 89, 86]  # right_hand_[l, p, t] and right_hand_virtual_l
        self.middleLinks = [91, 92, 93, 90]  # right_hand_[k, o, s] and right_hand_virtual_k
        self.ringLinks = [104, 105, 106, 103]  # right_hand_[j, n, r] and right_hand_virtual_j
        self.pinkyLinks = [99, 100, 101, 98]  # right_hand_[i, m, q] and right_hand_virtual_i
        self.handLinks = (self.palmLinks + self.thumbLinks + self.indexLinks + self.middleLinks +
                          self.ringLinks + self.pinkyLinks)  # 26 links

        # finger joint in [MCP, PIP, DIP]
        # index
        self.index_joint_r = [87, 88, 89]
        self.index_joint_l = [41, 42, 43]

        self.index_alpha_mcp = 0.67
        self.index_alpha_dip = 0.77
        self.index_joint_max = [0.79849, 1.334, 1.394]
        # mid
        self.mid_joint_r = [91, 92, 93]
        self.mid_joint_l = [45, 46, 47]

        self.mid_alpha_mcp = 0.67
        self.mid_alpha_dip = 0.75
        self.mid_joint_max = [0.79849, 1.334, 1.334]
        # ring
        self.ring_joint_r = [104, 105, 106]
        self.ring_joint_l = [58, 59, 60]

        self.ring_alpha_mcp = 0.67
        self.ring_alpha_dip = 0.75
        self.ring_joint_max = [0.98175, 1.334, 1.395]

        # pinky
        self.pinky_joint_r = [99, 100, 101]
        self.pinky_joint_l = [53, 54, 55]

        self.pinky_alpha_mcp = 0.67
        self.pinky_alpha_dip = 0.57
        self.pinky_joint_max = [0.98175, 1.334, 1.3971]

        # thumb
        self.thumb_joint_r = [80, 81, 82]
        self.thumb_joint_l = [34, 35, 36]

        self.thumb_alpha_mcp = 0.67
        self.thumb_alpha_dip = 0.5
        self.thumb_joint_max = [0.9704, 0.98506, 1.406]
        self.hand_joints = [78, 80, 81, 82, 86, 87, 88, 89, 91, 92, 93, 98, 99, 100, 101, 103, 104, 105, 106]
        self.finger_joints = [78, 80, 81, 82, 86, 87, 88, 89, 91, 92, 93, 98, 99, 100, 101, 103, 104, 105, 106]
        self.wheel_joints = [3, 4, 5, 6]
    ########################################## robot rest pos ################################################

        # wheels and pan and tilt
        self.car_initial_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # left arm
        self.left_arm_initial_pos = [-0.0023850635414499745, -0.5163902356343145, -1.9667822782174174,
                                     -2.046727181005138, 0.0014841842216413985, -0.6837058402550358]
        # left hand
        self.left_hand_initial_pos = [0.9879, 0.9704, 0.98506, 1.406, 0.28833, 0.79849, 1.334, 1.394, 0.79849, 1.334,
                                      1.334, 0.5829, 0.98175, 1.334, 1.3971, 0.28833, 0.98175, 1.334, 1.395]
        # right arm
        self.right_arm_initial_pos = [-0.0023891079478327083, -2.5163905456766303, 1.9667964054643494,
                                      -1.0481748092516625, 0.0014841842216413985, 1.683496311243686]

        # robot rest pos except for grasp pose of right hand
        self.arm_pos = (self.car_initial_pos + self.left_arm_initial_pos + self.left_hand_initial_pos +
                        self.right_arm_initial_pos)

        ########################################## right hand grasp topology ###################################

        # platform topology
        self.platform = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # platform topology
        self.relax = [0., 0.1, 0.1, 0.1, 0., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0., 0.1, 0.1, 0.1, 0., 0.1, 0.1, 0.1]

        self.inSiAd2 = [-5.505757020954133e-07, 0.029915555588673565, 0.04459163350382067, 0.022306418383103044,
                        -6.307837781042613e-16, 0.7403527016174615, 1.1050192214019174, 0.8508442116936221,
                        0.7613242713714279, 1.1363136564784961, 0.8522250889150719, 1.0796192504974497e-05,
                        0.6538296454750105, 0.9758946883498663, 0.5562521943865839, 7.417146554943206e-06,
                        0.6773680402706062, 1.0110018797317464, 0.758241048448454]

        self.pPdAb2 = [0.9878295412164513, 0.18050690485777726, 0.26934819527173925, 0.1346270985748516,
                       -1.024120979218551e-22, 0.5152334110080274, 0.7690326095195987, 0.5921305413696777,
                       -1.4170276315794524e-06, -1.2944166456782428e-06, -1.5440709503624593e-07,
                       2.8100407868064208e-05, -3.472058364554257e-08, 2.76893660930718e-05, 6.79686107128117e-08,
                       -2.6806090949049904e-07, -4.6707046056431425e-07, 1.7147707509608124e-05, 1.0835411670736107e-05]

        self.pPdAb23 = [0.987897860602783, 0.19656337245789682, 0.2933787826165356, 0.1466878745541722,
                        -4.367701095116939e-12, 0.4892367606320518, 0.7302312285045254, 0.5622612874078037,
                        0.4720542000455045, 0.704575956147246, 0.5284277552845956, -5.961653247187501e-08,
                        -5.5865696978680826e-09, -1.7888859403729357e-06, -1.001374718716222e-08,
                        -1.4978609557525e-07, -4.441555989749478e-07, -4.084069375413004e-09, -2.8354311515307458e-06]

        self.pPdAb25 = [0.9877861744655235, 0.1976655190467019, 0.29493803992866036, 0.14742076315131036,
                        -3.453990966221513e-08, 0.48447262419885095, 0.7231544398628342, 0.5568291348722626,
                        0.5099284322194312, 0.7611390949520193, 0.5708132673484286, 1.4260458767573854e-05,
                        0.3806314008122537, 0.5681385300824333, 0.32382839539434805, -1.3072333525717985e-07,
                        0.4849997717413712, 0.7239231319137976, 0.5429355078041577]

        self.poPmAb25 = [0.1, 0.1730438230245825, 0.25825909377621026, 0.1291313340919125,
                         -2.0667514181565684e-20, 0.4937177285699986, 0.7369358011441807, 0.5674285840962151,
                         0.5363703244448537, 0.8005861809944325, 0.6004370990822906, -1.4617310696058945e-07,
                         0.42727452647544445, 0.6377660751817943, 0.3635193758041601, -2.389287900264578e-07,
                         0.4895094674432001, 0.7306396566991542, 0.5479796554403817]

        # grasp pose dictionary
        self.grasp_pose = {"platform": self.platform, "inSiAd2": self.inSiAd2, "pPdAb2": self.pPdAb2,
                           "pPdAb23": self.pPdAb23, "pPdAb25": self.pPdAb25, "poPmAb25": self.poPmAb25,
                           "relax": self.relax}

    ############################################# helper functions #########################################

    def rNum(self, lower, upper, step=0.1):
        random_float = random.uniform(lower, upper)
        rounded_float = round(random_float / step) * step
        return rounded_float

    def loadInfo(self):
        random.seed(round(time.time()))
        i = random.randint(0, 26)  # task id
        #i = 0
        csvName = "object_info/" + "pos_all.csv"
        data = pd.read_csv(csvName)
        object_id = data.iloc[i]['Object ID']  # 0
        trans = np.array(ast.literal_eval(data.iloc[i]['Trans']))  # 1
        orientation = np.array(ast.literal_eval(data.iloc[i]['Orientation']))  # 2
        lower = data.iloc[i]['lower']  # 3
        upper = data.iloc[i]['upper']  # 4
        p_obj = np.array(ast.literal_eval(data.iloc[i]['p_obj']))  # 5
        q_obj = np.array(ast.literal_eval(data.iloc[i]['q_obj']))  # 6
        #objectPath = "objects/" + str(object_id) + "/" + "target.urdf"  # 7
        objectPath = "objects/" + str(object_id) + "/" + "target.urdf"  # 7
        #objectPath = "objects_t/" + str(object_id) + "/" + "target.urdf"  # 7
        grasp = data.iloc[i]['Topology']  # 8
        affordance = data.iloc[i]['Affordance']  # 9
        task_id = data.iloc[i]['Task ID']  # 9

        return [object_id, trans, orientation, lower, upper, p_obj, q_obj, objectPath, grasp, affordance, task_id]

    def calculate_rigid_trans(self, p_obj, q_obj, p_rel, q_rel):
        # p_obj, q_obj is the latest position and orientation of object
        # p_new = p_obj + C(q_obj)*p_rel
        # q_new = q_obj ⊗ q_rel
        # ⊗ is the quaternion multiplication operator
        # C(q_obj) is the rotation matrix formed from q_obj
        q_obj = self.quaternion_normalize(q_obj)
        c_q = self.c_function(q_obj)
        p_new = p_obj + np.dot(c_q, p_rel)
        q_new = self.quaternion_multiply(q_obj, q_rel)
        q_new = self.convert_hand(q_new)
        q_new = self.quaternion_normalize(q_new)
        return p_new, q_new

    def crt(self, p_obj, q_obj, p_rel, q_rel):
        # p_obj, q_obj is the latest position and orientation of object
        # p_new = p_obj + C(q_obj)*p_rel
        # q_new = q_obj ⊗ q_rel
        # ⊗ is the quaternion multiplication operator
        # C(q_obj) is the rotation matrix formed from q_obj
        q_obj = self.quaternion_normalize(q_obj)
        c_q = self.c_function(q_obj)
        p_new = p_obj + np.dot(c_q, p_rel)
        q_new = self.quaternion_multiply(q_obj, q_rel)
        q_new = self.quaternion_normalize(q_new)
        return p_new, q_new

    def relative_pno(self, p1, q1, p2, q2):
        # Input: ndarray
        # p1, p2 are coordinates of object 1 and 2
        # q1, q2 are quaternions of object 1 and 2
        # q in format [x, y, z, w]
        # Output: relative position and orientation
        p_rel = p2 - p1
        q_rel = self.quaternion_multiply(q2, self.conjugateQ(q1))  # q_rel = q2 dot q1*
        q_rel = self.quaternion_normalize(q_rel)
        return p_rel, q_rel

    def c_function(self, q):
        # convert quaternion to rotation matrix
        x, y, z, w = q
        xx = 2 * x * x
        xy = 2 * x * y
        xz = 2 * x * z
        yy = 2 * y * y
        yz = 2 * y * z
        zz = 2 * z * z
        wx = 2 * w * x
        wy = 2 * w * y
        wz = 2 * w * z
        rotation_matrix = np.array([[1 - yy - zz, xy - wz, xz + wy],
                                    [xy + wz, 1 - xx - zz, yz - wx],
                                    [xz - wy, yz + wx, 1 - xx - yy]])
        return rotation_matrix

    def rotate_object(self, q_initial, d, ax):
        # rotate object from q_initial orientation by d degree around ax axis
        # q_initial = [a, b, c, w], d = int, ax in ["x", "y", "z"]
        q_rotate = [0, 0, 0, 0]
        if ax == "x":
            q_rotate = [math.sin(d / 2), 0, 0, math.cos(d / 2)]
        if ax == "y":
            q_rotate = [0, math.sin(d / 2), 0, math.cos(d / 2)]
        if ax == "z":
            q_rotate = [0, 0, math.sin(d / 2), math.cos(d / 2)]
        q_new = self.quaternion_multiply(q_rotate, q_initial)
        return q_new

    def conjugateQ(self, q):
        return [-q[0], -q[1], -q[2], q[3]]

    def quaternion_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return [x, y, z, w]

    def quaternion_normalize(self, q):
        magnitude = sum(x ** 2 for x in q) ** 0.5
        return [x / magnitude for x in q]

    def convert_hand(self, q):
        # rotate the robotic hand -90 around y
        # and -180 around x to match the orientation
        # of the hand in the dataset.
        # The rotation may change
        y90 = [0, 0.7071, 0.7071, 0]
        q_hand = [0, 0, 0, 1]
        q_hand = self.quaternion_multiply(y90, q_hand)
        q_hand = self.quaternion_multiply(q, q_hand)
        return q_hand

    def distant(self, current_pos, target_pos):
        return np.linalg.norm(np.array(current_pos) - np.array(target_pos))

    def calculate_direction(self, v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        else:
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


