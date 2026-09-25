import math
from typing import TypeGuard

# Based on t2m_kinematic chain btw

vec3 = tuple[float, float, float]
def is_vec3(val: list[float]) -> TypeGuard[vec3]:
    return len(val) == 3

class Pose():
    def __init__(self) -> None:
        self.root_length = 0.1
        self.floor_plane = 0.3
        self.upperarm_length = 0.25
        self.lowerarm_length = 0.25
        self.upperleg_length = 0.35
        self.lowerleg_length = 0.35
        self.hip_offset = 0.03
        self.hip_height = -0.07
        self.clavicle_offset = 0.04
        self.clavicle_height = 0.1
        self.shoulder_offset = 0.05
        self.shoulder_height = -0.02
        self.foot_height = 0.03
        self.toe_offsetz = 0.08
        self.spine1_length = 0.1
        self.spine2_length = 0.04
        self.spine3_length = 0.2
        self.neck_length = 0.05

    def distance(self, v1, v2):
        if not is_vec3(v1) or not is_vec3(v2): raise ValueError("Parameters should both be vectors.")
        x1, y1, z1 = v1
        x2, y2, z2 = v2

        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        return distance

    def arm_chain(self, data: list[list[float]]):
        spine3 = vec3(data[9])
        l_clavicle = vec3(data[14])
        r_clavicle = vec3(data[13])

        l_shoulder = vec3(data[17])
        r_shoulder = vec3(data[16])

        l_elbow = vec3(data[19])
        r_elbow = vec3(data[18])
        
        l_wrist = vec3(data[21])
        r_wrist = vec3(data[20])

        self.clavicle_offset = round((abs(spine3[0] - l_clavicle[0]) + abs(r_clavicle[0] - spine3[0])) / 2, 2)
        self.shoulder_offset = round((abs(l_shoulder[0] - l_clavicle[0]) + abs(r_shoulder[0] - r_clavicle[0])) / 2, 2)
        self.clavicle_height = round((l_clavicle[1] - spine3[1] + r_clavicle[1] - spine3[1]) / 2, 2)
        self.shoulder_height = round((l_shoulder[1] - l_clavicle[1] + r_shoulder[1] - r_clavicle[1]) / 2, 2)

        self.upperarm_length = round((self.distance(l_shoulder, l_elbow) + self.distance(r_shoulder, r_elbow)) / 2, 2)
        self.lowerarm_length = round((self.distance(l_elbow, l_wrist) + self.distance(r_elbow, r_wrist)) / 2, 2)

    
    def leg_chain(self, data: list[list[float]]):
        root = vec3(data[0])
        l_hip = vec3(data[2])
        r_hip = vec3(data[1])

        l_knee = vec3(data[5])
        r_knee = vec3(data[4])

        l_foot = vec3(data[8])
        r_foot = vec3(data[7])
        
        l_toe = vec3(data[11])
        r_toe = vec3(data[10])

        self.floor_plane = round((l_toe[1] + r_toe[1]) / 2, 2)

        self.hip_offset = round((abs(root[0] - l_hip[0]) + abs(r_hip[0] - root[0])) / 2, 2)
        self.hip_height = round((l_hip[1] - root[1] + r_hip[1] - root[1]) / 2, 2)
        self.foot_height = round((l_foot[1] - l_toe[1] + r_foot[1] - r_toe[1]) / 2, 2)
        self.toe_offsetz = round((abs(l_toe[2] - l_foot[2]) + abs(r_toe[2] - r_foot[2])) / 2, 2)

        self.upperleg_length = round((self.distance(l_hip, l_knee) + self.distance(r_hip, r_knee)) / 2, 2)
        self.lowerleg_length = round((self.distance(l_knee, l_foot) + self.distance(r_knee, r_foot)) / 2, 2)

    def spine_chain(self, data: list[list[float]]):
        root = vec3(data[0])
        spine1 = vec3(data[3])
        spine2 = vec3(data[6])
        spine3 = vec3(data[9])
        neck = vec3(data[12])
        head = vec3(data[15])


        self.root_length = round(self.distance(root, spine1), 2)
        self.spine1_length = round(self.distance(spine1, spine2), 2)
        self.spine2_length = round(self.distance(spine2, spine3), 2)
        self.spine3_length = round(self.distance(spine3, neck), 2)
        self.neck_length = round(self.distance(neck, head), 2)


    def init_pose(self, data: list[list[float]]):
        self.spine_chain(data)
        self.arm_chain(data)
        self.leg_chain(data)
        
    def t_pose(self):
        foot_height = self.floor_plane + self.foot_height
        knee_height = foot_height + self.lowerleg_length
        root_height = knee_height + self.upperleg_length

        spine1_height = root_height + self.root_length
        spine2_height = spine1_height + self.spine1_length
        spine3_height = spine2_height + self.spine2_length

        clavicle_height = spine3_height + self.clavicle_height
        shoulder_height = clavicle_height + self.shoulder_height

        neck_height = spine3_height + self.spine3_length
        head_height = neck_height + self.neck_length


        l_toe = [-self.hip_offset, self.floor_plane, self.toe_offsetz]
        l_foot = [-self.hip_offset, foot_height, 0.0]
        l_knee = [-self.hip_offset, knee_height, 0.0]
        l_hip = [-self.hip_offset, root_height + self.hip_height, 0.0]

        root = [0.0, root_height, 0.0]

        r_toe = [self.hip_offset, self.floor_plane, self.toe_offsetz]
        r_foot = [self.hip_offset, foot_height, 0.0]
        r_knee = [self.hip_offset, knee_height, 0.0]
        r_hip = [self.hip_offset, root_height + self.hip_height, 0.0]

        spine1 = [0.0, spine1_height, 0.0]
        spine2 = [0.0, spine2_height, 0.0]
        spine3 = [0.0, spine3_height, 0.0]
        
        neck = [0.0, neck_height, 0.0]
        head = [0.0, head_height, 0.0]

        l_clavicle = [-self.clavicle_offset, clavicle_height, 0.0]
        l_shoulder = [-self.clavicle_offset + -self.shoulder_offset, shoulder_height, 0.0]
        l_elbow = [-self.clavicle_offset + -self.shoulder_offset + -self.upperarm_length, shoulder_height, 0.0]
        l_wrist = [-self.clavicle_offset + -self.shoulder_offset + -self.upperarm_length + -self.lowerarm_length, shoulder_height, 0.0]

        r_clavicle = [self.clavicle_offset, clavicle_height, 0.0]
        r_shoulder = [self.clavicle_offset + self.shoulder_offset, shoulder_height, 0.0]
        r_elbow = [self.clavicle_offset + self.shoulder_offset + self.upperarm_length, shoulder_height, 0.0]
        r_wrist = [self.clavicle_offset + self.shoulder_offset + self.upperarm_length + self.lowerarm_length, shoulder_height, 0.0]
        
        return [root, r_hip, l_hip, spine1, r_knee, l_knee, spine2, r_foot, l_foot, spine3, r_toe, l_toe, neck, r_clavicle, l_clavicle, head, r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist]
    


        



