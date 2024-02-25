import gymnasium as gym
from dqn import Agent
from utils import plot_learning_curve
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                  eps_end=0.01, input_dims=[8], lr=0.001, eps_dec=1e-4)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0 
        done = False
        observation, _ = env.reset()
        while not done:
            # print(type(observation))
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            # print(observation)
            agent.store_transition(action, observation, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
        
    x = [i+1 for i in range(n_games)]
    filename = 'lundar_lander.png'
    plot_learning_curve(x, scores, eps_history, filename)