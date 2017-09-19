import os
import pickle
from util import parse_args, roll_expert

if __name__ == '__main__':
    args = parse_args(envname=(str, None), num_rollouts=(int, None), render=(bool, None))

    if args.envname is None:
        env_list = os.listdir('experts')
    else:
        env_list = [args.envname]
    
    for env in env_list:
        envname = env.split('.')[0]
        rollouts = roll_expert(envname, args.num_rollouts, args.render)

        save_file = 'expert_rollouts/{}.pkl'.format(envname)
        with open(save_file, 'wb') as f:
            pickle.dump(rollouts, f)
            print('Saved results to {}'.format(save_file))
