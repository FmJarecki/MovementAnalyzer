import os

from movements.ChinUp import ChinUp

if __name__ == "__main__":
    chinup = ChinUp()
    video_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'videos', 'chinup3.mp4')
    chinup.calc_pull_ups(video_path)
