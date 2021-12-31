import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--std', default='1,1,1', type=str)
    parser.add_argument('--mean', default='0,0,0', type=str)
    parser.add_argument('--pt-data', default='ori_neigh', choices=['ori_rand', 'ori_neigh', 'train'], type=str)
    parser.add_argument('--pt-method', default='adv', choices=['adv', 'dir_adv', 'normal'], type=str)
    parser.add_argument('--adv-dir', default='na', choices=['na', 'pos', 'neg', 'both'], type=str)
    parser.add_argument('--neigh-method', default='untargeted', choices=['untargeted', 'targeted'], type=str)
    parser.add_argument('--pt-iter', default=50, type=int)
    parser.add_argument('--pt-lr', default=0.001, type=float)
    parser.add_argument('--att-iter', default=20, type=int)
    parser.add_argument('--att-restart', default=1, type=int)
    parser.set_defaults(blackbox=False, type=bool)
    parser.add_argument('--blackbox', dest='blackbox', action='store_true')
    parser.add_argument('--log-file', default='logs/default.log', type=str)
    args = parser.parse_args()

    # process dataset std and mean
    args.std = [float(x) for x in list(args.std.split(','))]
    args.mean = [float(x) for x in list(args.mean.split(','))]

    # check args validity
    if args.adv_dir != 'na':
        assert args.pt_method == 'dir_adv'
    if args.pt_method == 'dir_adv':
        assert args.adv_dir != 'na'
    return args