import cv2
import gymnasium as gym
import numpy as np

def preprocess(img):
    img = img[:84, 6:90]  # CarRacing-v2-specific cropping
    # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img


class ImageEnv(gym.Wrapper):
    def __init__(
            self,
            env,
            skip_frames=4,  #하나의 행동을 skip_frames만큼 반복함, state는 마지막 프레임을 사용
            stack_frames=4, #하나의 프레임으로 행동을 확인할 수 없으므로 4개의 프레임을 쌓아놓고 행동을 확인함
            initial_no_op=50, #초기화 시, 50프레임만큼 아무 행동도 하지 않음
            **kwargs
    ):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

    def reset(self):
        # Reset the original environment.
        s, info = self.env.reset()

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(0) #CarRacing-v2에서는 0은 엑셀,핸들,브레이크 모두 0임

        # 이미지 전처리 - Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # The initial observation is simply a copy of the frame `s`
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        return self.stacked_state, info

    def step(self, action):
        # We take an action for self.skip_frames steps
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info
