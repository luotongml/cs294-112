import gym
import numpy as np
import tensorflow as tf
import tf_util
import load_policy
from argparse import ArgumentParser
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

NUM_EPISODES_DEFAULT = 50
RENDER_DEFAULT = False

def parse_args(**kwargs):
    parser = ArgumentParser()
    for arg, (_type, default) in kwargs.items():
        parser.add_argument('--{}'.format(arg), type=_type, default=default)
    return parser.parse_args()

def generate_pdf_table(data, savepath):
    doc = SimpleDocTemplate('pdf/{}.pdf'.format(savepath), pagesize=A4)
    elements = []
    t=Table(data,5*[1.5*inch], (len(data))*[1*inch])
    t.setStyle(TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),
                          ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                          ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
                          ('BOX', (0,0), (-1,-1), 0.25, colors.black)]))
    
    elements.append(t)
    doc.build(elements)

def roll_expert(envname, num_episodes, render):
    num_episodes = num_episodes or NUM_EPISODES_DEFAULT
    render = render or RENDER_DEFAULT

    expert_policy_file = 'experts/{}.pkl'.format(envname)
    predict_fn = load_policy.load_policy(expert_policy_file)

    with tf.Session():
        tf_util.initialize()

        return simulate(envname, predict_fn, num_episodes, render)

def roll_model(envname, predict_fn, num_episodes, render):
    if num_episodes == None: num_episodes = NUM_EPISODES_DEFAULT
    if render == None: render = RENDER_DEFAULT

    return simulate(envname, predict_fn, num_episodes, render)    

def simulate(envname, predict_fn, num_episodes, render):
    print('Running {}'.format(envname))

    env = gym.make(envname)
    max_steps = env.spec.timestep_limit

    rewards = []
    observations = []
    actions = []
    for i in range(num_episodes):
        print('Episode', i+1)
        observation = env.reset()
        done = False
        total_reward = 0.
        steps = 0
        while not done and steps < max_steps:
            action = predict_fn(observation[None,:])

            observations.append(observation)
            actions.append(action)

            observation, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

            if render: env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))

        rewards.append(total_reward)

    expert_data = {'observations': np.array(observations),
                    'actions': np.array(actions)}

    return expert_data, np.mean(rewards), np.std(rewards)
