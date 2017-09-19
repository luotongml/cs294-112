import os
import pickle
import numpy as np
from BCModel import BCModel
import tensorflow as tf
import tf_util
from util import parse_args, roll_expert, roll_model
from load_policy import load_policy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = parse_args(envname=(str, None),
                      pdf_save_path=(str, None),
                      hidden_layers=(int, None),
                      units=(int, None), epochs=(int, None),
                      learning_rate=(float, None),
                      num_episodes=(int, None),
                      render=(bool, None),
                      num_iterations=(int, 10))
    
    if args.envname is None:
        env_list = os.listdir('experts')
    else:
        env_list = [args.envname]

    all_results = []
    for env in env_list:
        envname = env.split('.')[0]
        print(envname)

        with open('expert_rollouts/{}.pkl'.format(envname), 'rb') as f:
            expert_data, expert_mean, expert_std = pickle.loads(f.read())
            observations, actions = expert_data.values()
            actions = actions.reshape(-1, actions.shape[2])

        means = []
        stds = []
        for i in range(args.num_iterations):
            print('Iteration {}'.format(i+1))

            model = BCModel(args.hidden_layers, args.units, args.epochs, args.learning_rate)

            print('# Step 1: train model on human data')
            model.train(observations, actions)
            
            print('# Step 2: run model to get new dataset')
            model_data, model_mean, model_std = roll_model(envname, model.predict, args.num_episodes, args.render)
            observations_new, actions_new = model_data.values()
            means.append(model_mean)
            stds.append(model_std)

            print('# Step 3: run expert to label new observations')
            predict_fn = load_policy('experts/{}.pkl'.format(envname))
            with tf.Session():
                tf_util.initialize()
                correct_actions_new = predict_fn(observations_new)
                print(correct_actions_new.shape)

            print('# Step 4: aggregate datasets')
            observations = np.concatenate((observations, observations_new))
            actions = np.concatenate((actions, correct_actions_new))
        
        all_results.append((envname, means, stds, expert_mean, expert_std))
        
    pdf_save_path = args.pdf_save_path or ('dagger_results' if len(env_list) > 1 else '{}_dagger_results'.format(env_list[0]))
    with PdfPages('pdf/{}.pdf'.format(pdf_save_path)) as pdf:
        for envname, means, stds, expert_mean, expert_std in all_results:
            plt.figure()
            plt.errorbar(range(len(means)), means, yerr=stds)
            plt.title('Dagger results - {}'.format(envname))
            plt.ylabel('mean reward')
            plt.xlabel('dagger iterations')
            plt.text(.05, .05, 'Expert mean: {} Expert std: {}'.format(expert_mean, expert_std))
            pdf.savefig()
            plt.close()
