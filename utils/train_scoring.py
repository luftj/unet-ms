
def get_log_data(logfile):
    data = []
    with open(logfile) as fr:
        header = fr.readline().strip().split(",")

        for line in fr:
            line=line.strip().split(",")
            data.append({k:line[header.index(k)] for k in header})
    return data

def get_best_dice(logfile="train_log.txt"):
    data = get_log_data(logfile)
    dices = [float(x["score"]) for x in data]
    best_dice = max(dices)
    best_epoch = dices.index(best_dice)
    print("best dice: %f at epoch %d" % (best_dice, best_epoch))
    return best_dice, best_epoch

def get_best_loss(logfile="train_log.txt"):
    data = get_log_data(logfile)
    losses = [float(x["loss"]) for x in data]
    best_loss = min(losses)
    best_epoch = losses.index(best_loss)
    print("best loss: %f at epoch %d" % (best_loss, best_epoch))
    return best_loss, best_epoch

def plot_train_scores(logfile="train_log.txt", outdir="plots/"):
    from matplotlib import pyplot as plt
    data = get_log_data(logfile)
    epochs = [str(x["epoch"]) for x in data] # todo: str wont really work with many epochs. handle xticsk properly
    dices = [float(x["score"]) for x in data]
    losses = [float(x["loss"]) for x in data]
    plt.plot(epochs,dices, label="dice", color="r")
    plt.plot(epochs,losses, label="loss", color="b")
    plt.vlines([dices.index(max(dices)),losses.index(min(losses))],0,1,["r","b"])
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(outdir+"train_scores.png")
    plt.show() 

if __name__ == "__main__":
    get_best_dice()
    get_best_loss()
    plot_train_scores()