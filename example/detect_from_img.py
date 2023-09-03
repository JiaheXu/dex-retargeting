import pickle
from pathlib import Path

import cv2
import tqdm
import tyro

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector

import mediapipe as mp

def retarget_video(retargeting: SeqRetargeting, img_path: str, output_path: str, config_path: str):

    #rgb = mp.Image.create_from_file(img_path)
    rgb = cv2.imread(img_path)

    rgb = cv2.resize( rgb, (640,360))
    cv2.imwrite("rgb_test.jpg", rgb)

    data = []
    #print("img shape: ", rgb.shape)
    detector = SingleHandDetector(hand_type="Right", selfie=False)
    length = int(1)
    with tqdm.tqdm(total=length) as pbar:
        
        num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot = detector.detect(rgb)
        plot_rgb = rgb
        print("keypoint_2d: ", keypoint_2d)
        print("joint_pos: ", joint_pos)
        img = detector.draw_skeleton_on_image(rgb, keypoint_2d, "white")
        cv2.imwrite( "./example/2d_joints.jpg", img)



def main(
    robot_name: RobotName, video_path: str, output_path: str, retargeting_type: RetargetingType, hand_type: HandType
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        video_path: The file path for the input video in .mp4 format.
        output_path: The file path for the output data in .pickle format.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
    """

    config_path = get_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = Path(__file__).parent.parent / "assets" / "robots"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    retarget_video(retargeting, video_path, output_path, str(config_path))


if __name__ == "__main__":
    tyro.cli(main)
