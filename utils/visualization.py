import matplotlib.pyplot as plt

def fig_loss_acc(train_data, valid_data, loss_or_acc, savepath):
    fig = plt.figure()
    plt.plot(train_data)
    plt.plot(valid_data)
    # Whether to plot loss or validation
    if loss_or_acc == 'loss':
        plt.title("Loss")
        plt.legend(['train', 'valid'])
        plt.savefig(str(savepath) + "/loss.png", format="png", dpi=300)
    else:
        plt.title("Accuracy")
        plt.legend(['train', 'valid'])
        plt.savefig(str(savepath) + "/accuracy.png", format="png", dpi=300)
    plt.close()

