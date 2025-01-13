# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys
from abc import ABC
from typing import Dict, List, Optional

import torch
from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict
from mmdet.models.builder import DETECTORS
from torch import Tensor

from .query_support import QuerySupportDetectorFPD
from .utils import TestMixins

from .builder import build_information_coupling#加
#from mmfewshottwo.mmfewshot.detection.models.builder import build_information_coupling#加
#from mmfewshot.detection.models.builder import build_information_coupling#加
from torchvision import transforms#加
import torch.nn.functional as F

@DETECTORS.register_module()
class FPD(QuerySupportDetectorFPD, TestMixins):
    """Implementation of `FPD.  <https://arxiv.org/abs/tbd>`_.
    Args:
        backbone (dict): Config of the backbone for query data.
        neck (dict | None): Config of the neck for query data and
            probably for support data. Default: None.
        support_backbone (dict | None): Config of the backbone for
            support data only. If None, support and query data will
            share same backbone. Default: None.
        support_neck (dict | None): Config of the neck for support
            data only. Default: None.
        rpn_head (dict | None): Config of rpn_head. Default: None.
        roi_head (dict | None): Config of roi_head. Default: None.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        pretrained (str | None): model pretrained path. Default: None.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 backbone: ConfigDict,
                 neck: Optional[ConfigDict] = None,
                 support_backbone: Optional[ConfigDict] = None,
                 support_neck: Optional[ConfigDict] = None,
                 information_coupling: Optional[ConfigDict] = None,#加
                 rpn_head: Optional[ConfigDict] = None,
                 roi_head: Optional[ConfigDict] = None,
                 train_cfg: Optional[ConfigDict] = None,
                 test_cfg: Optional[ConfigDict] = None,
                 pretrained: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None,
                 post_rpn=True,
                 with_refine=False,
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            support_backbone=support_backbone,
            support_neck=support_neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            post_rpn=post_rpn)

        #
        self.with_information_coupling = False
        if information_coupling is not None:
            self.with_information_coupling = True
            self.information_coupling = build_information_coupling(information_coupling)
        #加


        self.is_model_init = False
        # save support template features for model initialization,
        # `_forward_saved_support_dict` used in :func:`forward_model_init`.
        self._forward_saved_support_dict = {
            'gt_labels': [],
            'roi_feats': [],
        }
        # save processed support template features for inference,
        # the processed support template features are generated
        # in :func:`model_init`
        self.inference_support_dict = {}

        self._new_forward_saved_support_dict = {
            'gt_labels': [],
            'weight': [],
            'prototypes': [],  # fine-grained prototypes
            'prototypes_novel': [],
        }
        self.new_inference_support_dict = {
            'weight': {},
            'prototypes': {},
            'prototypes_novel': {},
        }

        # refine results for COCO. We do not use it for VOC.
        self.with_refine = with_refine

        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30)
        ])

    @auto_fp16(apply_to=('img',))
    def extract_support_feat(self, img):
        """Extracting features from support data.
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
        Returns:
            list[Tensor]: Features of input image, each item with shape
                (N, C, H, W).
        """
        feats = self.backbone(img, use_meta_conv=True)
        if self.support_neck is not None:
            feats = self.support_neck(feats)
        return feats

    def forward_model_init(self,
                           img: Tensor,
                           img_metas: List[Dict],
                           gt_bboxes: List[Tensor] = None,
                           gt_labels: List[Tensor] = None,
                           **kwargs):
        """extract and save support features for model initialization.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.

        Returns:
            dict: A dict contains following keys:

                - `gt_labels` (Tensor): class indices corresponding to each
                    feature.
                - `res5_rois` (list[Tensor]): roi features of res5 layer.
        """
        # `is_model_init` flag will be reset while forwarding new data.
        self.is_model_init = False
        assert len(gt_labels) == img.size(0), \
            'Support instance have more than two labels'

        augmented_img = self.augmentations(img)#加
        img = torch.cat((img, augmented_img), dim=0)
        augmented_img = self.augmentations(img)#加
        img = torch.cat((img, augmented_img), dim=0)
        n_list = []
        n_list.extend(gt_labels)
        n_list.extend(gt_labels)
        n_list.extend(gt_labels)
        n_list.extend(gt_labels)
        gt_labels = n_list
        #gt_labels_ = torch.cat(gt_labels)
        #if getattr(self.roi_head, 'prototypes_distillation', False):
        #    if self.roi_head.num_novel != 0:
        #        num_classes = self.roi_head.bbox_head.num_classes
        #        nc = torch.tensor(self.roi_head.novel_class, device='cuda')
        #        n_ids = torch.isin(gt_labels_, nc)
        #        b_ids = torch.logical_not(n_ids)
        #        b_gts = gt_labels_[b_ids]
        #        n_gts = gt_labels_[n_ids]
                #加
                #for label in n_gts:
                #    index = torch.where(gt_labels_ == label)[0]
                #    image = img[index]
                #    image = self.augmentations(image)
                #    img = torch.cat((img, image), dim=0)
                #    gt_labels.append(label.item())
                #加

        feats = self.extract_support_feat(img)#原feats

        self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
        roi_feat = self.roi_head.extract_support_feats(feats)  # (16,2048)
        self._forward_saved_support_dict['roi_feats'].extend(roi_feat)

        gt_labels_ = torch.cat(gt_labels)
        if getattr(self.roi_head, 'prototypes_distillation', False):
            if self.roi_head.num_novel != 0:
                num_classes = self.roi_head.bbox_head.num_classes
                nc = torch.tensor(self.roi_head.novel_class, device='cuda')
                n_ids = torch.isin(gt_labels_, nc)
                
                b_ids = torch.logical_not(n_ids)
                b_gts = gt_labels_[b_ids]
                n_gts = gt_labels_[n_ids]
                ###可改


                bc = torch.tensor(list(set(list(range(num_classes))) - set(self.roi_head.novel_class)), device='cuda')
                r_b_gts = torch.cat([torch.argwhere(bc == item)[0] for item in b_gts], dim=0) \
                    if len(b_gts) else b_gts  # relative gt_labels
                r_n_gts = torch.cat([torch.argwhere(nc == item)[0] for item in n_gts], dim=0) \
                    if len(n_gts) else n_gts  # relative gt_labels
                base_support_feats = feats[0][b_ids] # 形状：(len(b_ids), C, H, W)
                novel_support_feats = feats[0][n_ids]


                # 循环计算相似度并更新标签
                for i in n_ids:
                    max_sim = -float('inf')  # 初始化最大相似度为负无穷
                    max_idx = -1  # 初始化最大相似度的索引

                    # 展平新类特征 (C, H, W) 为 (C*H*W)
                    novel_feat_flat = feats[0][i]
                    
                    # 计算新类特征与所有旧类特征的余弦相似度
                    for j in b_ids:

                        # 展平旧类特征 (C, H, W) 为 (C*H*W)
                        base_feat_flat = feats[0][j]
                        # similarity = F.cosine_similarity(
                        #     novel_support_feats[i].unsqueeze(0),  # 扩展为 (1, C, H, W)
                        #     base_support_feats[j].unsqueeze(0)  # 扩展为 (1, C, H, W)
                        # )

                        # 计算余弦相似度
                        similarity = F.cosine_similarity(novel_feat_flat.squeeze(0), base_feat_flat.squeeze(0), dim=0)

                        # 如果当前相似度大于最大相似度，更新最大相似度及索引
                        if similarity.mean() > max_sim:
                            max_sim = similarity.mean()
                            max_idx = j
                    
                    # 更新标签：将最大相似度对应的旧类特征标签替换为新类特征的标签
                    #print("n_ids shape:", gt_labels_[max_idx].shape)  #  torch.Size([64])
                    #print("bbbbbb_shape:", gt_labels_[i].shape)  #  torch.Size([64])

                    if gt_labels_[max_idx].shape == gt_labels_[i].shape:
                        gt_labels_[max_idx] = gt_labels_[i]

                # n_i = 0
                # for i in b_ids:
                #     gt_labels_[i] = b_gts[n_i]
                #     n_i += 1

                num_classes = self.roi_head.bbox_head.num_classes
                nc = torch.tensor(self.roi_head.novel_class, device='cuda')
                n_ids = torch.isin(gt_labels_, nc)
                b_ids = torch.logical_not(n_ids)
                b_gts = gt_labels_[b_ids]
                n_gts = gt_labels_[n_ids]
                bc = torch.tensor(list(set(list(range(num_classes))) - set(self.roi_head.novel_class)), device='cuda')
                r_b_gts = torch.cat([torch.argwhere(bc == item)[0] for item in b_gts], dim=0) \
                    if len(b_gts) else b_gts  # relative gt_labels
                r_n_gts = torch.cat([torch.argwhere(nc == item)[0] for item in n_gts], dim=0) \
                    if len(n_gts) else n_gts  # relative gt_labels
                base_support_feats = feats[0][b_ids] # 形状：(len(b_ids), C, H, W)
                novel_support_feats = feats[0][n_ids]




                # prototypes distillation
                weight_base, prototypes_base = self.roi_head.prototypes_distillation(
                    base_support_feats, support_gt_labels=r_b_gts)
                weight_novel, prototypes_novel = self.roi_head.prototypes_distillation(
                    novel_support_feats, support_gt_labels=r_n_gts, forward_novel=True, forward_novel_test=True)
                prototypes = torch.cat([prototypes_base, prototypes_novel], dim=0)
                prototypes[b_ids] = prototypes_base
                prototypes[n_ids] = prototypes_novel
                weight = torch.cat([weight_base, weight_novel], dim=0)
                weight[b_ids] = weight_base
                weight[n_ids] = weight_novel
            else:
                weight, prototypes = self.roi_head.prototypes_distillation(feats[0], support_gt_labels=gt_labels_)
            self._new_forward_saved_support_dict['gt_labels'].extend(gt_labels)
            self._new_forward_saved_support_dict['weight'].extend([weight])
            self._new_forward_saved_support_dict['prototypes'].extend([prototypes])

        return {'gt_labels': gt_labels, 'roi_feat': roi_feat}

    def model_init(self):
        pass

    def fpd_model_init(self):
        """process the saved support features for model initialization."""
        gt_labels = torch.cat(self._forward_saved_support_dict['gt_labels'])
        class_ids = set(gt_labels.data.tolist())

        roi_feats = torch.cat(self._forward_saved_support_dict['roi_feats'])
        self.inference_support_dict.clear()
        for class_id in class_ids:
            self.inference_support_dict[class_id] = roi_feats[
                gt_labels == class_id].mean([0], True)
        self.is_model_init = True
        # reset support features buff
        for k in self._forward_saved_support_dict.keys():
            self._forward_saved_support_dict[k].clear()

        if getattr(self.roi_head, 'prototypes_distillation', False):
            weight = torch.cat(self._new_forward_saved_support_dict['weight'])
            prototypes = torch.cat(self._new_forward_saved_support_dict['prototypes'])
            for k in self.new_inference_support_dict.keys():
                self.new_inference_support_dict[k].clear()
            for class_id in class_ids:
                self.new_inference_support_dict['weight'][class_id] = weight[
                    gt_labels == class_id].mean([0], True)
                self.new_inference_support_dict['prototypes'][class_id] = prototypes[
                    gt_labels == class_id].mean([0], True)

                ws = 0.5
                weighted_sum = True  # test time natural integration
                if weighted_sum:
                    prototypes_c = prototypes[gt_labels == class_id]
                    weight_c = weight[gt_labels == class_id][:, :prototypes_c.size(1)].div(ws).softmax(0).unsqueeze(-1)
                    prototypes_c = prototypes_c.mul(weight_c).sum(0, True)
                    self.new_inference_support_dict['prototypes'][class_id] = prototypes_c

            if self.roi_head.num_novel != 0:
                nc = torch.tensor(self.roi_head.novel_class, device='cuda')
                n_ids = torch.isin(torch.arange(len(class_ids), device='cuda'), nc)
                b_ids = torch.logical_not(n_ids)

                # weight = torch.cat(
                #     [self.new_inference_support_dict['weight'][class_id] for class_id in class_ids], dim=0)
                prototypes = torch.cat(
                    [self.new_inference_support_dict['prototypes'][class_id] for class_id in class_ids], dim=0)
                prototypes_base = prototypes[b_ids][:, :self.roi_head.prototypes_distillation.num_queries, :]
                prototypes_novel = prototypes[n_ids]  # (5, 5, 1024)
                # weight_novel = weight[n_ids]  # (5, 5)
                self.new_inference_support_dict['prototypes'][0] = prototypes_base
                self.new_inference_support_dict['prototypes_novel'][0] = prototypes_novel
                # prototypes = torch.cat([prototypes_base, prototypes_novel], 0)
            else:
                prototypes = torch.cat(
                    [self.new_inference_support_dict['prototypes'][class_id] for class_id in class_ids], dim=0)
                self.new_inference_support_dict['prototypes'][0] = prototypes
            keys = list(self.new_inference_support_dict['prototypes'].keys())
            for k in keys:
                if k != 0:
                    self.new_inference_support_dict['prototypes'].pop(k)

            for k in self._new_forward_saved_support_dict.keys():
                self._new_forward_saved_support_dict[k].clear()

    def simple_test(self,
                    img: Tensor,
                    img_metas: List[Dict],
                    proposals: Optional[List[Tensor]] = None,
                    rescale: bool = False):
        """Test without augmentation.
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor] | None): override rpn proposals with
                custom proposals. Use when `with_rpn` is False. Default: None.
            rescale (bool): If True, return boxes in original image space.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) == 1, 'Only support single image inference.'
        if not self.is_model_init:
            # process the saved support features
            self.fpd_model_init()

        query_feats = self.extract_feat(img)

        proposal_list = None
        if proposals is None:
            if not self.post_rpn:
                proposal_list = self.rpn_head.simple_test(query_feats, img_metas)
        else:
            proposal_list = proposals

        bbox_results = self.roi_head.simple_test(
            query_feats,
            copy.deepcopy(self.inference_support_dict),
            copy.deepcopy(self.new_inference_support_dict),
            proposal_list,
            img_metas,
            rescale=rescale)
        if self.with_refine:
            return self.refine_test(bbox_results, img_metas)
        else:
            return bbox_results
