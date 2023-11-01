from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip
import torch
import os

from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from .utils import extend_instance
import logging


LLAVA_TO_VICUNA = {
    "liuhaotian/llava-v1.5-7b": "lmsys/vicuna-7b-v1.5",
}


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    checkpoint_activations: bool = False,
    freeze_vision_encoder: bool = False,
    load_detection_head_weight: str = None,
    **flamingo_kwargs,
):
    is_llava = "llava" in lang_encoder_path
    if is_llava:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        model_path = lang_encoder_path
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        text_tokenizer, llava_model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
        mm_projector = llava_model.model.mm_projector
        vision_encoder = llava_model.model.vision_tower
        del llava_model.model.layers
        del llava_model.lm_head
    else:
        mm_projector = None
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
        )
        # set the vision encoder to output the visual features
        vision_encoder.visual.output_tokens = True
        # delete text encoder part
        del vision_encoder.transformer
        del vision_encoder.text_projection
        del vision_encoder.token_embedding
        del vision_encoder.ln_final
        del vision_encoder.positional_embedding
        del vision_encoder.logit_scale
        vision_encoder.visual.proj = None
        vision_encoder.visual.ln_post = torch.nn.Identity()
        text_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=use_local_files
        )

    # add Flamingo special tokens to the tokenizer
    additional_special_tokens = ["<|#image#|>", "<|#endofimage#|>", "<|#visual#|>", "<|#object#|>", "<|#box#|>", "<|#endofobject#|>", "<|#attr#|>", "<|#endofattr#|>", "<|#previsual#|>", "<|#prebox#|>"]
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    if is_llava:
        vicuna_path = LLAVA_TO_VICUNA[lang_encoder_path]
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            vicuna_path, local_files_only=use_local_files
        )
    else:
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path, local_files_only=use_local_files
        )
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))
    lang_encoder_name = lang_encoder.__class__.__name__.lower()
    if checkpoint_activations:
        from fairscale.nn.checkpoint import checkpoint_wrapper
        vision_encoder_layers = vision_encoder.vision_tower.vision_model.encoder.layers if is_llava else vision_encoder.visual.transformer.resblocks
        for i in range(len(vision_encoder_layers)):
            vision_encoder_layers[i] = checkpoint_wrapper(
                vision_encoder_layers[i],
                offload_to_cpu=False,
            )
        if "opt" in lang_encoder_name:
            lang_encoder_layers = lang_encoder.model.decoder.layers
        elif "codegen" in lang_encoder_name:
            lang_encoder_layers = lang_encoder.transformer.h
        elif "llama" in lang_encoder_name:
            lang_encoder_layers = lang_encoder.model.layers
        elif "gptneo" in lang_encoder_name:
            lang_encoder_layers = lang_encoder.gpt_neox.layers
        else:
            raise ValueError(f"unknown model {lang_encoder_name}")
        for i in range(len(lang_encoder_layers)):
            lang_encoder_layers[i] = checkpoint_wrapper(
                lang_encoder_layers[i],
                offload_to_cpu=False,
            )
    if is_llava:
        vis_dim = vision_encoder.config.hidden_size
        image_size = vision_encoder.config.image_size
        patch_size = vision_encoder.config.patch_size
    else:
        vis_dim = open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["width"]
        image_size = open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["image_size"]
        patch_size = open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["patch_size"]
    assert image_size % patch_size == 0
    vis_embed_size = (image_size // patch_size) ** 2

    lang_dim = int(lang_encoder.config.hidden_size)
    if hasattr(lang_encoder.config, "word_embed_proj_dim"):
        hidden_state_dim = lang_encoder.config.word_embed_proj_dim
    else:
        hidden_state_dim = lang_encoder.config.hidden_size
    model = Flamingo(
        vision_encoder=vision_encoder,
        lang_encoder=lang_encoder,
        eoc_token_id=text_tokenizer.encode(text_tokenizer.eos_token)[-1],
        media_token_id=text_tokenizer.encode("<|#image#|>")[-1],
        image_end_token_id=text_tokenizer.encode("<|#endofimage#|>")[-1],
        visual_token_id=text_tokenizer.encode("<|#visual#|>")[-1],
        previsual_token_id=text_tokenizer.encode("<|#previsual#|>")[-1],
        box_token_id=text_tokenizer.encode("<|#box#|>")[-1],
        prebox_token_id=text_tokenizer.encode("<|#prebox#|>")[-1],
        endofobject_token_id=text_tokenizer.encode("<|#endofobject#|>")[-1],
        vis_dim=vis_dim,
        vis_embed_size=vis_embed_size,
        lang_dim=lang_dim,
        image_size=image_size,
        patch_size=patch_size,
        hidden_state_dim=hidden_state_dim,
        mm_projector=mm_projector,
        **flamingo_kwargs,
    )

    if is_llava and load_detection_head_weight is not None:
        temp = torch.load(load_detection_head_weight, map_location="cpu")
        detection_head_checkpoint = {}
        for key in temp["model_state_dict"]:
            if key.startswith("detection_head"):
                detection_head_checkpoint[key.replace("detection_head.", "")] = temp["model_state_dict"][key]
        model.detection_head.yolox_head.load_state_dict(detection_head_checkpoint, strict=True)
        logging.info(f"load detection_head weights from: {load_detection_head_weight}")
        del temp

    if freeze_vision_encoder:
        logging.info("freeze vision encoder")
        model.vision_encoder.requires_grad_(False)

    logging.info(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer, vis_embed_size


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        "We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    # "gptneo": "transformer.h",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "gptneox": "gpt_neox.layers",
    "llama": "model.layers",
    "llamaforcausallm": "model.layers",
    "gpt2": "transformer.h",
    "codegen": "transformer.h",
}
