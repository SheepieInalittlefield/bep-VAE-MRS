def main():
    # from config import custom_VAE_config, gen_parameters
    # from visualization import sample_model, show_generated_data, mse
    # from training import train_model
    # from data import load_mrs
    # from torch import mean
    #
    # parameters = gen_parameters(lr=1e-3, batch_size=16, epochs=800, dim=12)
    # trained_model, mse = train_model(custom_VAE_config, parameters)
    #
    # train_dataset, eval_dataset, ppm = load_mrs()
    # del(train_dataset)
    # gen_data = sample_model(trained_model)
    # gen_data = gen_data.cpu()
    # eval_data_mean = mean(eval_dataset, dim=0)
    # show_generated_data(gen_data, eval_data_mean, ppm)
    from data import load_mrs_real, load_mrs_simulations
    train_dataset, eval_dataset, ppm = load_mrs_simulations()
    train_dataset, eval_dataset, ppm = load_mrs_real()


def parameter_search():
    from hyperparameter import test
    runs = test()
    with open('custom_vae_search.txt', 'w') as f:
        f.write(str(runs))


if __name__ == '__main__':
    main()

# MSE: 0.0001698292866272024
# Train loss: 0.9325
# Eval loss: 2.6887
