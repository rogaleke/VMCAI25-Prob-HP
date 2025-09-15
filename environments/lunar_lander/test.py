from stable_baselines3 import PPO
from LunarLander import CustomLunarLander

env = CustomLunarLander(render_mode='human')

next_state, info = env.reset()
current_state = env.get_state()

model = PPO.load("models/PPO_1/4000000_steps")

# print(next_state)
# print(current_state)
for i in range(300):
    print(env.normalize_state(env.get_state()))
    print(env.get_state())
    state = env.get_state()
    action, _states = model.predict(env.normalize_state(state))
    next_state, _, done, _, _ = env.step(action)
    env.render()
