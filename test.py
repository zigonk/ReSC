import os
import sys
import argparse
import time
import random
import json
import math
from distutils.version import LooseVersion
import scipy.misc
import logging
import datetime
import cv2
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from torchvision.transforms import Compose, ToTensor, Normalize

from utils.transforms import letterbox, random_affine
from dataset.data_loader import *
from model.grounding_model import *
from model.loss import *
from utils.parsing_metrics import *
from utils.utils import *
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume

def load_image(img_path):
    img = cv2.imread(img_path)
    return img

class Evaluator():
    def __init__(self, image_path, phrase, model, transform=None, imsize=256, max_query_len=128, bert_model='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.query_len = max_query_len
        self.phrase = phrase
        self.imsize = imsize
        self.model = model
        img = load_image(image_path)
        img, _, ratio, dw, dh = letterbox(img, None, self.imsize)
        if (transform is not None):
            self.img = transform(self.img).unsqueeze(0)
        self.dw = dw
        self.dh = dh
        self.ratio =ratio
    def tokenize_text(self):
        phrase = self.phrase.lower()
        examples = read_examples(phrase, 1)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        return np.array(word_id, dtype=int), np.array(word_mask, dtype=int)
    def eval(self):
        model.eval()
        word_id, word_mask = self.tokenize_text()
        img = self.img
        dw = self.dw
        dh = self.dh
        ratio = self.ratio

        image = img.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda
        with torch.no_grad():
            pred_anchor_list, attnscore_list = model(image, word_id, word_mask)
        pred_anchor = pred_anchor_list[-1]
        pred_anchor = pred_anchor.view(   \
                pred_anchor.size(0),9,5,pred_anchor.size(2),pred_anchor.size(3))
        pred_conf = pred_anchor[:,:,4,:,:].contiguous().view(1,-1)
        max_conf, max_loc = torch.max(pred_conf, dim=1)

        pred_bbox = torch.zeros(1,4)

        pred_gi, pred_gj, pred_best_n = [],[],[]
        grid, grid_size = args.size//args.gsize, args.gsize
        anchor_idxs = range(9)
        anchors = [anchors_full[i] for i in anchor_idxs]
        scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            x[1] / (args.anchor_imsize/grid)) for x in anchors]

        pred_conf = pred_anchor[:,:,4,:,:].data.cpu().numpy()
        max_conf_ii = max_conf.data.cpu().numpy()

        (best_n, gj, gi) = np.where(pred_conf[0,:,:,:] == max_conf_ii[0])
        best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
        pred_gi.append(gi)
        pred_gj.append(gj)
        pred_best_n.append(best_n)

        pred_bbox[0,0] = F.sigmoid(pred_anchor[0, best_n, 0, gj, gi]) + gi
        pred_bbox[0,1] = F.sigmoid(pred_anchor[0, best_n, 1, gj, gi]) + gj
        pred_bbox[0,2] = torch.exp(pred_anchor[0, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
        pred_bbox[0,3] = torch.exp(pred_anchor[0, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
        pred_bbox[0,:] = pred_bbox[0,:] * grid_size
        pred_bbox = xywh2xyxy(pred_bbox)
        pred_bbox[:,0], pred_bbox[:,2] = (pred_bbox[:,0]-dw)/ratio, (pred_bbox[:,2]-dw)/ratio
        pred_bbox[:,1], pred_bbox[:,3] = (pred_bbox[:,1]-dh)/ratio, (pred_bbox[:,3]-dh)/ratio

        ## convert pred, gt box to original scale with meta-info
        top, bottom = round(float(dh[0]) - 0.1), args.size - round(float(dh[0]) + 0.1)
        left, right = round(float(dw[0]) - 0.1), args.size - round(float(dw[0]) + 0.1)
        img_np = img[0,:,top:bottom,left:right].data.cpu().numpy().transpose(1,2,0)

        ratio = float(ratio)
        new_shape = (round(img_np.shape[1] / ratio), round(img_np.shape[0] / ratio))
        ## also revert image for visualization
        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)

        img_np = Variable(torch.from_numpy(img_np.transpose(2,0,1)).cuda().unsqueeze(0))

        pred_bbox[:,:2], pred_bbox[:,2], pred_bbox[:,3] = \
            torch.clamp(pred_bbox[:,:2], min=0), torch.clamp(pred_bbox[:,2], max=img_np.shape[3]), torch.clamp(pred_bbox[:,3], max=img_np.shape[2])
        return pred_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--impath', default='', help='image path')
    parser.add_argument('--text', default='', help='query sentence')
    parser.add_argument('--size', default=256, type=int, help='image size')
    parser.add_argument('--anchor_imsize', default=416, type=int,
                        help='scale used to calculate anchors defined in model cfg file')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--nflim', default=3, type=int, help='nflim')
    parser.add_argument('--mstage', dest='mstage', default=False, action='store_true', help='if mstage')
    parser.add_argument('--mstack', dest='mstack', default=False, action='store_true', help='if mstack')
    parser.add_argument('--w_div', default=0.125, type=float, help='weight of the diverge loss')
    parser.add_argument('--fusion', default='prod', type=str, help='prod/cat')
    parser.add_argument('--tunebert', dest='tunebert', default=False, action='store_true', help='if tunebert')
    parser.add_argument('--large', dest='large', default=False, action='store_true', help='if large mode: fpn16, convlstm out, size 512')

    global args, anchors_full
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))
    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    ## following anchor sizes calculated by kmeans under args.anchor_imsize=416
    anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors = [float(x) for x in anchors.split(',')]
    anchors_full = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)][::-1]

    model = grounding_model_multihop(NFilm=args.nflim, fusion=args.fusion, intmd=args.mstack, mstage=args.mstage, \
        emb_size=args.emb_size, coordmap=True, convlstm=args.large, \
        bert_model=args.bert_model, dataset=args.dataset, tunebert=args.tunebert)
    model = torch.nn.DataParallel(model).cuda()
    model=load_pretrain(model,args,logging)

    evaluator = Evaluator(args.impath, args.text, model, input_transform)
    pred_bbox = evaluator.eval()
    print(pred_bbox)
