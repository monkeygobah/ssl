from full_parameter import *
from full_trainer import Trainer
# from full_tester import Tester
from full_data_loader import Data_Loader_Split
from torch.backends import cudnn
from full_utils import make_folder
import torch
import wandb
from full_tester import Tester

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    print('starting run in main')

    cudnn.benchmark = True
    set_seed(42)
    
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")

    if config.train:
        make_folder(config.model_save_path, config.model_name)
        
        data_loader = Data_Loader_Split(config.labeled_img_path, config.labeled_mask_path, config.imsize, config.batch_size, config.train)            

        trainer = Trainer(data_loader.loader(), config, device=device)
        trainer.train()
            
    if config.test:
        tester = Tester(config, device)
        tester.test()


        # tester.generate_pseudolabels(
        #     pseudolabel_save_path='pseudolabeled_masks_all')

        # tester.generate_pseudolabels(
        #     pseudolabel_save_path='pseudolabeled_masks_08',
        #     confidence_threshold=0.8)

        # tester.generate_pseudolabels(
        #     pseudolabel_save_path='pseudolabeled_masks_05',
        #     confidence_threshold=0.5)

if __name__ == '__main__':
    config = get_parameters()
    main(config)


