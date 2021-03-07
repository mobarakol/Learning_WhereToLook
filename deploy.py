import math
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
#from model import ST_MTL_SEG
from model import SalSegNet
from dataset import SurgicalDataset
from utils import seed_everything, calculate_dice, calculate_confusion_matrix_from_arrays
from metrics_functions import AUC_Borji, NSS, SIM
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings("ignore")


def validate(valid_loader, model, args):
    confusion_matrix = np.zeros(
            (args.num_classes, args.num_classes), dtype=np.uint32)
    model.eval()
    SIM_SCORE, EMD_SCORE = [], []
    with torch.no_grad():
        for batch_idx, (inputs, labels_seg, labels_sal,_) in enumerate(valid_loader):
            inputs, labels_seg = inputs.to(device), np.array(labels_seg)
            pred_seg, output_sal = model(inputs)
            pred_seg = pred_seg.data.max(1)[1].squeeze_(1).cpu().numpy()
            confusion_matrix += calculate_confusion_matrix_from_arrays(
                pred_seg, labels_seg, args.num_classes)
            
            pred_ = output_sal.view(output_sal.size()[0], -1)
            gt_maps_ = labels_sal.view(labels_sal.size()[0], -1)
            labels_sal = np.array(labels_sal.cpu())
            output_sal = np.array(output_sal.cpu())
            for eval_idx in range(0, inputs.size()[0]):
                if(np.max(labels_sal[eval_idx])==0):
                    continue
                SIM_SCORE.append((SIM(labels_sal[eval_idx], np.squeeze(output_sal[eval_idx]))))
                EMD_SCORE.append((wasserstein_distance(pred_[eval_idx].cpu(), gt_maps_[eval_idx].cpu())))   

    confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
    dices = {'dice_{}'.format(cls + 1): dice
                for cls, dice in enumerate(calculate_dice(confusion_matrix))}
    dices_per_class = np.array(list(dices.values()))          

    return dices_per_class, SIM_SCORE, EMD_SCORE

def main():
    parser = argparse.ArgumentParser(description='Instrument Segmentation')
    parser.add_argument('--num_classes', default=8, type=int, help="num of classes")
    parser.add_argument('--data_root', default='instruments17_saliency', help="data root dir")
    parser.add_argument('--batch_size', default=2, type=int, help="num of classes")
    args = parser.parse_args()
    dataset_test = SurgicalDataset(data_root=args.data_root, seq_set=[4,7], is_train=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2,
                              drop_last=True)
    
    print('Sample size of test dataset:', dataset_test.__len__())
    model = SalSegNet(num_classes=args.num_classes).to(device)
    model = torch.nn.parallel.DataParallel(model)
    model.load_state_dict(torch.load('epoch_75.pth.tar'))
    dices_per_class, SIM_SCORE, EMD_SCORE = validate(test_loader, model, args)
    print('Mean Avg Dice:%.4f [Bipolar Forceps:%.4f, Prograsp Forceps:%.4f, Large Needle Driver:%.4f, Vessel Sealer:%.4f]'
        %(dices_per_class[:4].mean(),dices_per_class[0], dices_per_class[1],dices_per_class[2],dices_per_class[3]))
    print('Saliency Metrics: SIM:%.4f, EMD:%.4f'%(np.mean(SIM_SCORE), np.mean(EMD_SCORE)))
    
if __name__ == '__main__':
    class_names = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Vessel Sealer", "Grasping Retractor", "Monopolar Curve, Scissors", "Other"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything()
    main()