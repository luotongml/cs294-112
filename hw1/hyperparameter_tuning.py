import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from util import parse_args, roll_model, generate_pdf_table
from BCModel import BCModel
from train_model import train

if __name__ == '__main__':
    args = parse_args(envname=(str, None), model_save_suffix=(str, None), pdf_save_path=(str, None), hidden_layers=(int, None), units=(int, None), epochs=(int, None), learning_rate=(float, None), num_rollouts=(int, None), render=(bool, None))

    if args.envname is None:
        env_list = os.listdir('experts')
    else:
        env_list = [args.envname]

    prev_model_losses = {env.split('.')[0]: BCModel.load(env.split('.')[0]).history['loss'] for env in env_list}
    model_losses = train(env_list, args.model_save_suffix or 'tuned', args.hidden_layers, args.units, args.epochs, args.learning_rate)

    pdf_save_path = args.pdf_save_path or ('hyperparameter_tuning' if len(env_list) > 1 else '{}_hyperparameter_tuning'.format(env_list[0]))

    with PdfPages('pdf/{}.pdf'.format(pdf_save_path)) as pdf:
        for envname in model_losses:
            plt.figure()
            plt.plot(model_losses[envname], label='new_model')
            plt.plot(prev_model_losses[envname], label='prev_model')
            plt.title('Loss on imitation policy - {}'.format(envname))
            plt.legend()
            plt.ylabel('loss')
            plt.xlabel('epochs')
            pdf.savefig()
            plt.close()

    results = []
    for filename in env_list:
        envname = filename.split('.')[0]
        with open('expert_rollouts/{}'.format(filename), 'rb') as f:
            prev_model = BCModel.load(envname)
            _, prev_model_mean, prev_model_std = roll_model(envname, prev_model.predict, args.num_rollouts, args.render)

            model = BCModel.load(envname + 'tuned')
            _, model_mean, model_std = roll_model(envname, model.predict, args.num_rollouts, args.render)

            results.append([envname, model_mean, prev_model_mean, model_std, prev_model_std])
    data = [['', 'Tuned model mean', 'Previous model mean', 'Tuned model std', 'Previous model std'], *results]

    generate_pdf_table(data, pdf_save_path + '_comparison')
