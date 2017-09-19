import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from BCModel import BCModel
from util import parse_args, roll_model

def train(env_list, model_save_suffix, hidden_layers, units, epochs, learning_rate):
    model_losses = {}
    for env in env_list:
        envname = env.split('.')[0]
        print(envname)

        with open('expert_rollouts/{}.pkl'.format(envname), 'rb') as f:
            expert_data, _, __ = pickle.load(f)

            observations, actions = expert_data.values()
            actions = actions.reshape(-1, actions.shape[2])
        
            model = BCModel(hidden_layers, units, epochs, learning_rate)

            history = model.train(observations, actions)

            model.save(envname + ("_".join([model_save_suffix] if model_save_suffix else '')))

            model_losses[envname] = history['loss']

    return model_losses 


def write_pdf(env_list, pdf_save_path, model_losses):
    pdf_save_path = pdf_save_path or ('learning_rates' if len(env_list) > 1 else '{}_learning_rate'.format(env_list[0]))

    with PdfPages('pdf/{}.pdf'.format(pdf_save_path)) as pdf:
        for envname, model_loss in model_losses.items():
            plt.figure()
            plt.plot(model_loss)
            plt.title('Loss on imitation policy - {}'.format(envname))
            plt.ylabel('loss')
            plt.xlabel('epochs')
            pdf.savefig()
            plt.close()

if __name__ == '__main__':
    args = parse_args(envname=(str, None), model_save_suffix=(str, None), pdf_save_path=(str, None), hidden_layers=(int, None), units=(int, None), epochs=(int, None), learning_rate=(float, None))
    
    if args.envname is None:
        env_list = os.listdir('experts')
    else:
        env_list = [args.envname]

    model_losses = train(env_list, args.model_save_suffix, args.hidden_layers, args.units, args.epochs, args.learning_rate)
    write_pdf(env_list, args.pdf_save_path, model_losses)
