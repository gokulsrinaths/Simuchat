import os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'C:/Users/sgoku/Downloads/final_a/simuchat_rl')
os.environ['SIMUCHAT_MODEL'] = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'

from envs.social_consensus_env import SocialConsensusEnv

env = SocialConsensusEnv(config={'max_rounds': 6})
obs, info = env.reset(options={'topic': 'climate change'})
print("=== EPISODE START ===")
print("Topic: climate change | Agents: Alice, Bob, Charlie")
print()

done = False
step = 0
total_reward = 0.0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    agent = info['agent_name']
    action_name = info['action_name']
    response = info.get('response_text', '')[:90]
    print(f"Step {step+1} | {agent:8} | {action_name:16} | reward={reward:+.2f}")
    print(f"  -> {response}")
    step += 1

print()
print("=== EPISODE END ===")
print(f"Steps: {step} | Total reward: {total_reward:+.3f}")
print(f"Consensus reached: {info['consensus_reached']}")
print(f"Avg trust: {info['average_trust']:.3f}")
print(f"Polarization: {info['polarization_score']:.3f}")
print()
env.render()
