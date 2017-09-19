import os
import pickle
from util import roll_model, parse_args, generate_pdf_table
from BCModel import BCModel

if __name__ == "__main__":
    args = parse_args(envname=(str, None), pdf_save_path=(str, None), render=(bool, None), num_rollouts=(int, None))

    if args.envname is None:
        env_list = os.listdir('experts')
        pdf_save_path = args.pdf_save_path or 'model_expert_comparison'
    else:
        env_list = ['{}.pkl'.format(args.envname)]
        pdf_save_path = args.pdf_save_path or '{}_model_expert_comparison'.format(args.envname)

    results = []
    for filename in env_list:
        envname = filename.split('.')[0]
        with open('expert_rollouts/{}'.format(filename), 'rb') as f:
            _, expert_mean, expert_std = pickle.loads(f.read())

            model = BCModel.load(envname)
            _, model_mean, model_std = roll_model(envname, model.predict, args.num_rollouts, args.render)

            results.append([envname, model_mean, expert_mean, model_std, expert_std])

    data = [['', 'Trained model mean', 'Expert mean', 'Trained model std', 'Expert std'], *results]

    generate_pdf_table(data, pdf_save_path)
