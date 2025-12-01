# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from typing import Any, Dict, List, Tuple, Union
from .tiny_vit_sam import TinyViT
from .image_encoder import ImageEncoderViT
from model.sam_decoder import MaskDecoder1, MaskDecoder, SegmentAnythingDecoder, StyleMaskDecoder, adaptdecoder, adaptdecoderv2
from .prompt_encoder import PromptEncoder
from model.sam.prompt_encoder import PositionEmbeddingRandom
from model.diffunet import ContextUnet
from torchvision.transforms.functional import resize, to_pil_image
import numpy as np
from copy import deepcopy
from util.util import promt_generate
import math
from timm.models.vision_transformer import Block

class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: Union[ImageEncoderViT, TinyViT],
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        img_size=1024,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.img_size = img_size
        self.pe_layer = PositionEmbeddingRandom(128)

    @property
    def device(self) -> Any:
        return self.device

    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        # batched_input = batched_input[:, 1:, :, :]
        batched_input = self.channel_conv(batched_input)
        original_size = batched_input.shape[-1]  
        input_images = self.preprocess(batched_input)      
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # if "point_coords" in image_record:
            #     points = (image_record["point_coords"], image_record["point_labels"])
            # else:
            #     points = None
            # points = None
            # sparse_embeddings, dense_embeddings = self.prompt_encoder(
            #     points=points,
            #     boxes=image_record.get("boxes", None),
            #     masks=image_record.get("mask_inputs", None),
            # )
            img_pe = self.pe_layer([64, 64]).unsqueeze(0) # [1, 256, 64, 64]
            low_res_masks, iou_predictions = self.mask_decoder(image_embeddings=curr_embedding.unsqueeze(0),
                                           image_pe=img_pe, multimask_output=True)
            # masks = self.postprocess_masks(
            #     low_res_masks,
            #     # input_size=image_record["image"].shape[-2:],
            #     original_size=original_size,
            # )
            outputs.append(low_res_masks)
            # masks = masks > self.mask_threshold
            # outputs.append(
            #     {
            #         "masks": masks,
            #         "iou_predictions": iou_predictions,
            #         "low_res_logits": low_res_masks,
            #     }
            # )
        outputs = torch.stack(outputs, dim=0)
        outputs = self.postprocess_masks(
            outputs,
            original_size=original_size,
        )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        ) 
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

class mobilesam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: Union[ImageEncoderViT, TinyViT],
        mask_decoder: SegmentAnythingDecoder,
        img_size=1024,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.img_size = img_size
        self.pe_layer = PositionEmbeddingRandom(128)

    @property
    def device(self) -> Any:
        return self.device
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return x
    # @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
    ) -> List[Dict[str, torch.Tensor]]:
        # batched_input = batched_input[:, 1:, :, :]
        batched_input = self.channel_conv(batched_input)
        input_images = self.preprocess(batched_input)      
        image_embeddings = self.image_encoder(input_images)
        outputs = self.mask_decoder(image_embeddings)
      
        return outputs


class Sam_muti_styledeco(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder_1: StyleMaskDecoder,
        mask_decoder_2: MaskDecoder1,
        img_size=1024,
    ) -> None:
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder_1 = mask_decoder_1
        self.mask_decoder_2 = mask_decoder_2
        self.img_size = img_size
        self.transform = ResizeLongestSide(img_size)
        self.pe_layer = PositionEmbeddingRandom(128)

    @property
    def device(self) -> Any:
        return self.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        labels: torch.Tensor,
        multimask_output: bool,
        args: Any      
    ) -> List[Dict[str, torch.Tensor]]:
        B, _, H, W = batched_input.size()
        # batched_input = batched_input[:, 1:, :, :]
        batched_input = self.channel_conv(batched_input)
        original_size = batched_input.shape[-1]  
        input_images = self.preprocess(batched_input)      
        image_embeddings = self.image_encoder(input_images)
        
        outputs0 = []
        outputs1 = []
        outputs = []
        # 无提示decoder
        img_pe = self.pe_layer([64, 64]).unsqueeze(0).repeat(B, 1, 1, 1)
        outputs0, outputs1, diffloss = self.mask_decoder_1(image_embeddings=image_embeddings,
                                          image_pe=img_pe, multimask_output=True)
        outputs0 = self.postprocess_masks(
            outputs0,
            original_size=original_size,
        )   
        outputs1 = self.postprocess_masks(
            outputs1,
            original_size=original_size,
        )
        diffloss = diffloss / B

        consistency_loss = self.consis_loss(outputs0, outputs1)
        if self.training:
            tgt_out_maxvalue, _ = torch.max(outputs1, dim=1)
            flattened_maxvalues = tgt_out_maxvalue.view(B, -1)
            sorted_values, _ = torch.sort(flattened_maxvalues, descending=True, dim=1)
            num_elements = flattened_maxvalues.shape[1]
            threshold_idx = int(num_elements * (1-args.threshold))
            thresholds = sorted_values[:, threshold_idx].view(-1, 1, 1)
            tgt_out = torch.where(tgt_out_maxvalue < thresholds, torch.full_like(tgt_out_maxvalue, 255), torch.full_like(tgt_out_maxvalue, 0))
            # True_mask = (outputs1.argmax(1) != labels).unsqueeze(1)
            # masks = torch.cat((True_mask.long().float().detach(),outputs1.detach()),dim=1)

            prompt = promt_generate(tgt_out, is_clu=True)
            # 有提示decoder
            for i, (image_record, curr_embedding) in enumerate(zip(prompt, image_embeddings)):
                # Transform input prompts
                if "point_coords" in image_record:
                    point_coords, point_labels = image_record["point_coords"], image_record["point_labels"]
                    point_coords = self.transform.apply_coords(point_coords, (H, W))
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=batched_input.device)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=batched_input.device)
                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                    points = (coords_torch, labels_torch)
                if "boxes" in image_record:
                    box = image_record["boxes"]
                    box = self.transform.apply_boxes(box, (H, W))
                    box = torch.as_tensor(box, dtype=torch.float, device=batched_input.device)
                    box = box[None, :]
                # if "mask_inputs" in image_record:
                # mask_inputs = masks[i].unsqueeze(0)
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=None,
                )
                
                low_res_masks, iou_predictions = self.mask_decoder_2(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                outputs.append(low_res_masks)
            outputs = torch.stack(outputs, dim=0)
            outputs = self.postprocess_masks(
                outputs,
                original_size=original_size,
            )
            return outputs0, outputs1, outputs, diffloss, consistency_loss
        return outputs0, outputs1
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        ) 
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return x
    def consis_loss(self, aug_prob, im_prob):
        aug_prob = F.softmax(aug_prob, dim=1)
        im_prob = F.softmax(im_prob, dim=1)
        aug_prob = aug_prob.permute(0,2,3,1).reshape(-1, self.mask_decoder_1.num_classes)
        im_prob = im_prob.permute(0,2,3,1).reshape(-1, self.mask_decoder_1.num_classes)
        p_mixture = torch.clamp((aug_prob + im_prob) / 2., 1e-7, 1).log()
        consistency_loss = 1* (
                            F.kl_div(p_mixture, aug_prob, reduction='batchmean') +
                            F.kl_div(p_mixture, im_prob, reduction='batchmean') 
                            ) / 2.
        return consistency_loss


# class Sam_adapt(nn.Module):
#     mask_threshold: float = 0.0
#     image_format: str = "RGB"

#     def __init__(
#         self,
#         in_ch,
#         image_encoder: TinyViT,
#         prompt_encoder: PromptEncoder,
#         mask_decoder: adaptdecoderv2,
#         img_size=1024,
#     ) -> None:
#         """
#         SAM predicts object masks from an image and input prompts.

#         Arguments:
#           image_encoder (ImageEncoderViT): The backbone used to encode the
#             image into image embeddings that allow for efficient mask prediction.
#           prompt_encoder (PromptEncoder): Encodes various types of input prompts.
#           mask_decoder (MaskDecoder): Predicts masks from the image embeddings
#             and encoded prompts.
#           pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
#           pixel_std (list(float)): Std values for normalizing pixels in the input image.
#         """
#         super().__init__()
#         self.channel_conv = nn.Sequential(
#             nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(3)
#         )
#         self.image_encoder = image_encoder
#         self.prompt_encoder = prompt_encoder
#         self.mask_decoder = mask_decoder
#         self.img_size = img_size
#         self.transform = ResizeLongestSide(img_size)
#         self.pe_layer = PositionEmbeddingRandom(128)

#     @property
#     def device(self) -> Any:
#         return self.device

#     def forward(
#         self,
#         batched_input: List[Dict[str, Any]],
#         batched_inputtd: List[Dict[str, Any]],
#         args: Any,
#     ) -> List[Dict[str, torch.Tensor]]:
#         B, _, H, W = batched_input.size()
#         batched_input_3 = self.channel_conv(batched_input)
#         original_size = batched_input_3.shape[-1]  
#         input_images = self.preprocess(batched_input_3)
#         image_embeddings = self.image_encoder(input_images)
#         img_pe = self.pe_layer([64, 64]).unsqueeze(0).repeat(B, 1, 1, 1)

#         if self.training:
#             batched_inputtd_3 = self.channel_conv(batched_inputtd)
#             input_imagestd = self.preprocess(batched_inputtd_3)      
#             image_embeddingstd = self.image_encoder(input_imagestd)         
#             outputs, recon_src, recon_tgt = self.mask_decoder(batched_input, batched_inputtd, 
#                                         src_image_embeddings=image_embeddings, 
#                                         tgt_image_embeddings=image_embeddingstd, 
#                                         image_pe=img_pe)
#             outputs = self.postprocess_masks(
#                 outputs,
#                 original_size=original_size,
#             )
#             return outputs, recon_src, recon_tgt
#         else:
#             outputs = self.mask_decoder(batched_input, None, 
#                                         src_image_embeddings=image_embeddings, 
#                                         tgt_image_embeddings=None, 
#                                         image_pe=img_pe)
#             outputs = self.postprocess_masks(
#                 outputs,
#                 original_size=original_size,
#             )
#             return outputs
    
#     def postprocess_masks(
#         self,
#         masks: torch.Tensor,
#         original_size: Tuple[int, ...],
#     ) -> torch.Tensor:
        
#         masks = F.interpolate(
#             masks,
#             (original_size, original_size),
#             mode="bilinear",
#             align_corners=False,
#         ) 
#         return masks

#     def preprocess(self, x: torch.Tensor) -> torch.Tensor:
#         """Normalize pixel values and pad to a square input."""
#         if x.shape[-1] != self.img_size:
#             x = F.interpolate(
#                 x,
#                 (self.img_size, self.img_size),
#                 mode="bilinear",
#                 align_corners=False,
#             )
#         return x

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return loss

class Sam_adapt(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        in_ch,
        image_encoder: TinyViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: adaptdecoderv2,
        img_size=1024,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.img_size = img_size
        self.transform = ResizeLongestSide(img_size)
        self.pe_layer = PositionEmbeddingRandom(128)
        self.mmd_loss = MMD_loss(kernel_type='rbf', kernel_mul=2.0, kernel_num=5)

    @property
    def device(self) -> Any:
        return self.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        batched_inputtd: List[Dict[str, Any]],
        args: Any,
    ) -> List[Dict[str, torch.Tensor]]:
        B, _, H, W = batched_input.size()
        batched_input_3 = self.channel_conv(batched_input)
        original_size = batched_input_3.shape[-1]  
        input_images = self.preprocess(batched_input_3)
        img_pe = self.pe_layer([64, 64]).unsqueeze(0).repeat(B, 1, 1, 1)

        if self.training:
            image_embeddings, A_src = self.image_encoder(input_images, sd=True)
            batched_inputtd_3 = self.channel_conv(batched_inputtd)
            input_imagestd = self.preprocess(batched_inputtd_3)      
            image_embeddingstd, A_tgt = self.image_encoder(input_imagestd, sd=False)
            outputs, recon_src, recon_tgt, src_psnr, tgt_psnr = self.mask_decoder(batched_input, batched_inputtd, 
                                        src_image_embeddings=image_embeddings, 
                                        tgt_image_embeddings=image_embeddingstd, 
                                        image_pe=img_pe)
            param_alignment_loss = self.param_loss(A_src, A_tgt)
            outputs = self.postprocess_masks(
                outputs,
                original_size=original_size,
            )
            return outputs, recon_src, recon_tgt, param_alignment_loss, src_psnr, tgt_psnr
        else:
            image_embeddings, A_src = self.image_encoder(input_images, sd=False)
            outputs = self.mask_decoder(batched_input, None, 
                                        src_image_embeddings=image_embeddings, 
                                        tgt_image_embeddings=None, 
                                        image_pe=img_pe)
            outputs = self.postprocess_masks(
                outputs,
                original_size=original_size,
            )
            return outputs
    
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        
        masks = F.interpolate(
            masks,
            (original_size, original_size),
            mode="bilinear",
            align_corners=False,
        ) 
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if x.shape[-1] != self.img_size:
            x = F.interpolate(
                x,
                (self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return x
    
    def param_loss(self, A_src, A_tgt):
        lambda_A0 = 1
        alpha_A = 0.5
        
        loss_align_A = 0 
        for i in range(len(A_src)):
            d = i  # 层的深度    
            # 计算A参数的分布对齐损失
            lambda_A = lambda_A0 * math.exp(alpha_A * d)
            loss_align_A += lambda_A * self.mmd_loss(A_src[i], A_tgt[i])    
            # loss_align_A += lambda_A * F.mse_loss(A_src[i], A_tgt[i]) 
        
        # 返回总的参数对齐损失
        return loss_align_A 
