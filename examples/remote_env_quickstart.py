import os

from roboactions import RemoteEnv


def main():
    api_key = os.environ.get("ROBOACTIONS_API_KEY", "rk_your_api_key")
    env = RemoteEnv("CartPole-v1", render_mode="rgb_array", api_key=api_key)
    obs, info = env.reset(seed=123)
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Observation: {obs}")    
        frame = env.render()  # numpy.ndarray (H, W, 3) or None
        print(f"Frame: {frame}")
    env.close()


if __name__ == "__main__":
    main()


