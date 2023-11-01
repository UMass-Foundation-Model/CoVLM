import random
import torch
import torch.nn as nn
import numpy as np

from .utils import getattr_recursive, setattr_recursive


class FlamingoLayer(nn.Module):
    def __init__(self, decoder_layer):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.image_nums = None
        self.image_start_index_list = None

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x, image_nums=None, image_start_index_list=None, num_beams=None, visual_tokens=None, data_list=None):
        self.vis_x = vis_x
        self.image_nums = image_nums
        self.image_start_index_list = image_start_index_list
        self.num_beams = num_beams
        self.visual_tokens = visual_tokens
        self.data_list = data_list

    def forward(
        self,
        hidden_states,  # alignment with hugging face name
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        if self.vis_x is not None:
            if self.training:
                single_length = self.vis_x.shape[-2]
                image_nums = self.image_nums
                image_start_index_list = self.image_start_index_list
                image_nums = [0] + np.cumsum(image_nums).tolist()
                for i, (image_num_begin, image_num_end, start_indices) in enumerate(zip(image_nums[:-1], image_nums[1:], image_start_index_list)):
                    for index in start_indices:
                        if image_num_begin < image_num_end:
                            hidden_states[i, index:index+single_length] = self.vis_x[image_num_begin]
                            image_num_begin += 1

                if self.visual_tokens is not None and len(self.visual_tokens) != 0:
                    for i, (x, y) in enumerate(self.data_list):
                        if len(self.visual_tokens[i].shape) > 1:
                            # print(self.visual_tokens[i].shape[0], "embedding")
                            hidden_states[x, y+1-self.visual_tokens[i].shape[0]:y+1] = self.visual_tokens[i]
                        else:
                            # print(self.visual_tokens[i].shape[0], "embedding")
                            hidden_states[x, y] = self.visual_tokens[i]
            elif not self.training:
                if (
                    ("past_key_value" in decoder_layer_kwargs and decoder_layer_kwargs["past_key_value"] is None) or
                    ("layer_past" in decoder_layer_kwargs and decoder_layer_kwargs["layer_past"] is None)
                ):
                    single_length = self.vis_x.shape[-2]
                    image_nums = self.image_nums
                    image_start_index_list = self.image_start_index_list
                    image_nums = [0] + np.cumsum(image_nums).tolist()
                    for i, (image_num_begin, image_num_end, start_indices) in enumerate(zip(image_nums[:-1], image_nums[1:], image_start_index_list)):
                        for index in start_indices:
                            if image_num_begin < image_num_end:
                                hidden_states[i, index:index+single_length] = self.vis_x[image_num_begin]
                                image_num_begin += 1
                    if self.visual_tokens is not None and len(self.visual_tokens) != 0:
                        for i, (x, y) in enumerate(self.data_list):
                            # import pdb; pdb.set_trace()
                            # print(x, y, self.visual_tokens[i].shape)
                            if len(self.visual_tokens[i].shape) > 1:
                                # print(self.visual_tokens[i].shape[0], "embedding")
                                hidden_states[x, y+1-self.visual_tokens[i].shape[0]:y+1] = self.visual_tokens[i]
                            else:
                                # print(self.visual_tokens[i].shape[0], "embedding")
                                hidden_states[x, y] = self.visual_tokens[i]
        hidden_states = self.decoder_layer(
            hidden_states, attention_mask=attention_mask, **decoder_layer_kwargs
        )
        return hidden_states


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(
        self,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        self._set_decoder_layers(
            nn.ModuleList(
                [FlamingoLayer(decoder_layer) for decoder_layer in self._get_decoder_layers()]
            )
        )
        self.initialized_flamingo = True

    def forward(self, *input, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )
        return super().forward(
            *input, **kwargs
        )  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether only the first decoder layer is conditioned."""
        return self._get_decoder_layers()[0].is_conditioned() and all(not layer.is_conditioned() for layer in self._get_decoder_layers()[1:])

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
