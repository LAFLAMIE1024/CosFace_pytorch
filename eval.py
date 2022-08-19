from PIL import Image
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision.transforms import functional as F

cudnn.benchmark = True
import net

# Extract deep feature of the input image by using the given model
# param "is_gray" -> True / False : Different Transformation
def extractDeepFeature(img, model, is_gray):
    
    if is_gray:
        transform = transforms.Compose([
            """
            torchvision.transforms.Grayscale() is bascially the same in both versions.
            """
            transforms.Grayscale(),
            
            
            """
            torchvision.transforms.ToTensor()
                0.13.1 : 
                Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
                if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
                In the other cases, tensors are returned without scaling.
                
                Old version:
                Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
            """
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            
            
            """
            torchvision.transforms.Normalize()
                Descriptions are the basically the same, however:
            
                0.13.1 : 
                    NOTE: This transform acts out of place, i.e., it does not mutate the input tensor.
                    Parameters:
                        mean (sequence) – Sequence of means for each channel.
                        std (sequence) – Sequence of standard deviations for each channel.
                        inplace (bool,optional) – Bool to make this operation in-place. <- we need to keep this inplace to be True to make sure this function acts like before.
                
                Old version:
                    NOTE: This transform acts in-place, i.e., it mutates the input tensor.
                    Parameters:	
                        mean (sequence) – Sequence of means for each channel.
                        std (sequence) – Sequence of standard deviations for each channel.
            """
            transforms.Normalize(mean=(0.5,), std=(0.5,))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
        
    img, img_ = transform(img), transform(F.hflip(img)) # Basically the same in both versions.
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    
    feature = torch.cat((model(img), model(img_)), 1)[0].to('cpu')
    return feature


def KFold(n=6000, n_folds=10):
    
    folds = []
    base = list(range(n))
    
    for i in range(n_folds):
        test = base[ i * n / n_folds : (i + 1) * n / n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
        
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
        
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
            
    return best_threshold

# Evaluate with the given model : checkpoint/CosFace_24_checkpoint.pth
# However, this pretrained model is not provided (?
def eval(model, model_path=None, is_gray=False):
    
    predicts = []
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    root = '/home/wangyf/dataset/lfw/lfw-112X96/'
    with open('/home/wangyf/Project/sphereface/test/data/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]

    with torch.no_grad():
        for i in range(6000):
            p = pairs_lines[i].replace('\n', '').split('\t')

            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")

            # Read in image1 and image2
            with open(root + name1, 'rb') as f:
                img1 =  Image.open(f).convert('RGB')
            with open(root + name2, 'rb') as f:
                img2 =  Image.open(f).convert('RGB')

            # Extract their features
            f1 = extractDeepFeature(img1, model, is_gray)
            f2 = extractDeepFeature(img2, model, is_gray)

            # Calculate their distances by formula
            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, distance, sameflag))

    accuracy = []
    thd = []
    
    folds = KFold(n=6000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts = np.array(map(lambda line: line.strip('\n').split(), predicts))
    
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)

    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

    return np.mean(accuracy), predicts


if __name__ == '__main__':
    _, result = eval(net.sphere().to('cuda'), model_path='checkpoint/CosFace_24_checkpoint.pth')
    np.savetxt("result.txt", result, '%s')
