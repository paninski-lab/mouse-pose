from iblvideo import download_lp_models
from iblvideo.pose_lp import lightning_pose
from tests.download_test_data import _download_lp_test_data

#
# ckpts_path = download_lp_models()
# pqt = lightning_pose(
#     mp4_file='/media/mattw/poseinterface/_raw/ibl-face/session_videos/train/03d9a098-07bf-4765-88b7-85f8d8f620cc_left/videos/_iblrig_leftCamera.raw.mp4',
#     ckpts_path=ckpts_path,
#     remove_files=False,
# )
# print('parquet:', pqt)

cam = 'body'

test_data = _download_lp_test_data()
ckpts_path = download_lp_models()

mp4_file = test_data.joinpath('input', f'_iblrig_{cam}Camera.raw.mp4')
tmp_dir = test_data.joinpath('input', f'lp_tmp_iblrig_{cam}Camera.raw')

out_file = lightning_pose(
    mp4_file=str(mp4_file),
    ckpts_path=ckpts_path,
    force=True,
    create_labels=True,
    remove_files=False,
)
