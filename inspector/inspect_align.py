import matplotlib.pyplot as plt



def inspect_align(source, target, source_aligned, out_dir):
    mse_before = np.mean(np.sum((source - target)**2, axis=1))
    mse_after = np.mean(np.sum((source_aligned - target)**2, axis=1))
    logging.info('rmse before: {}'.format(np.sqrt(mse_before)))
    logging.info('rmse after: {}'.format(np.sqrt(mse_after)))

    # plot absolute deviation
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    std_dev = np.sqrt(np.sum((source - target)**2, axis=1))
    plt.hist(std_dev, bins=100, density=True)
    plt.title('before procrustes')
    plt.xlabel('Euclidean distance')
    plt.ylabel('density')

    plt.subplot(122)
    std_dev = np.sqrt(np.sum((source_aligned - target)**2, axis=1))

    max_std_dev = np.max(std_dev)
    thres = 10 if max_std_dev > 10 else max_std_dev
    mask = std_dev <= thres
    ratio = (1. - np.sum(mask) / mask.size) * 100.
    std_dev = std_dev[mask]

    plt.hist(std_dev, bins=100, density=True)
    plt.title('after procrustes\n{:.2f}% points above thres {}'.format(ratio, thres))
    plt.xlabel('Euclidean distance (meter)')
    plt.ylabel('density')
    plt.xticks(range(0, 11))
    plt.tight_layout()

    plt.savefig(out_dir)