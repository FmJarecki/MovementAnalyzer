import os

import pytest

from src.movements.ChinUp import ChinUp


test_data = [
    ("chinup1.mp4", 2),
    ("chinup2.mp4", 2),
    ("chinup3.mp4", 3)
]


@pytest.mark.parametrize("video, expected_pull_ups", test_data)
def test_calc_pull_ups(video, expected_pull_ups):
    class_obj = ChinUp()
    video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'videos', video)

    assert class_obj.calc_pull_ups(video_path) == expected_pull_ups
