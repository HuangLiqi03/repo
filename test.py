import gym
a = list(gym.envs.registry.all())
print(len(a))
import mani_skill2.envs
b = list(gym.envs.registry.all())
# 寻找a b的不同
for spec in b:
    if spec not in a:
        print(spec) 
