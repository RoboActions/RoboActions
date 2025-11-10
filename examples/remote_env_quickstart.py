from roboactions import remotegym


def main():
    env = remotegym.make("CartPole-v1", render_mode="rgb_array")
    obs, info = env.reset()
    print(f"Info: {info}")
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Observation: {obs}")    
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")
        if terminated or truncated:
            obs, info = env.reset()
            print(f"Reset Info: {info}")
    env.close()


if __name__ == "__main__":
    main()


