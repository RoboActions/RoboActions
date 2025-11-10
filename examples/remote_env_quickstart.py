from roboactions import remotegym
from gymnasium.wrappers import RecordVideo


def main():
    env = remotegym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="videos")
    env.reset()
    for step in range(100):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        print(f"Step {step}")
        if terminated or truncated:
            env.reset()
    env.close()


if __name__ == "__main__":
    main()


