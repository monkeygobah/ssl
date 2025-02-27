import argparse




def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, required=False, default = '.')
    # parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--crop-size', type=int, default=256)

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=False, default = 'labeled_ids.txt')
    parser.add_argument('--val_path', type=str, required=False, default = 'labeled_ids_test.txt')

    parser.add_argument('--unlabeled-id-path', type=str, required=False,default = 'unlabeled_ids.txt')
    parser.add_argument('--pseudo-mask-path', type=str, required=False, default = 'pseudomasks_best_model')
    parser.add_argument('--masks_not_saved', action='store_true', help = 'if the pseudomasks havent been generated yet, turn this on to generate and store them')

    parser.add_argument('--save-path', type=str, required=False, default='st_plus_testing')

    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument

    # arguments for ST++
    parser.add_argument('--reliable_id_path', type=str,default='reliable_ids.txt')
    parser.add_argument('--plus', dest='plus', default=False, action='store_true', help='whether to use ST++')

    args = parser.parse_args()
    return args