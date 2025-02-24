import numpy as np
import torch

from lizrd.core.initialization import get_init_weight


FREEZE_PARAMS_REGULES = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",

    ".block.residual_attention.layer.attention.input_projection.input_projection.weight", #ATT
    ".block.residual_attention.layer.attention.output_projection.output_projection.weight",

    "embedding_layer.layers.0.embedding.weight", #TE
    "embedding_layer.layers.1.projected_layer.pe_layer.weight", #PE

    "head.head.weight", #Head
]

FF_PARAMS_BLACKLIST = [
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu.", #FF
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu.",
]

def freeze_projected_params(model, unprojected_ff):
    frozen_modules = []
    for name, param in model.named_parameters():
        if unprojected_ff and any([reg in name for reg in FF_PARAMS_BLACKLIST]):  # Check if the parameter belongs to layer1
            continue
        if any([reg in name for reg in FREEZE_PARAMS_REGULES]):  # Check if the parameter belongs to layer1
            param.requires_grad = False
            frozen_modules.append(param)
    return frozen_modules

FREEZE_LN_REGULES = [
    ".pre_norm.", # Layer norm
]

def freeze_ln_params(model):
    frozen_modules = []
    for name, param in model.named_parameters():
        if any([reg in name for reg in FREEZE_LN_REGULES]):  # Check if the parameter belongs to layer1
            param.requires_grad = False
            frozen_modules.append(param)
    return frozen_modules 

PROJECTIONS_1_1 = [
    ".block.residual_attention.layer.attention.input_projection.input_projection_p11.weight",
    ".block.residual_attention.layer.attention.output_projection.output_projection_p21.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight",
    "head.head_p.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight", #FF in - 1ff configuration
]

PROJECTIONS_1_1_T = [
    "embedding_layer.layers.0.embedding_p.weight",
    "embedding_layer.layers.1.projected_layer.pe_layer_p.weight",
    ".block.residual_attention.layer.attention.output_projection.output_projection_p22.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight",
    ".block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight", #FF out - 1ff configuration
    ".block.residual_attention.layer.attention.input_projection_out_projection_q.input_projection_p12_q.weight", #Attention in_projection_out_projection
    ".block.residual_attention.layer.attention.input_projection_out_projection_k.input_projection_p12_k.weight", 
    ".block.residual_attention.layer.attention.input_projection_out_projection_v.input_projection_p12_v.weight", 
]



def is_in_partial_list(elemen_name:str, partials_list:list[str]):
    for weight_name in partials_list:
        if weight_name in elemen_name:
            return True
    return False

def print_dict_hierarchy(d, indent=0): #dev debug
    """Recursively print dictionary keys hierarchically."""
    for key, value in d.items():
        print(' ' * indent + str(key))
        if isinstance(value, dict):
            print_dict_hierarchy(value, indent + 2)

def svd_init_truncated_sv(weight:torch.Tensor, dm, projected_dm):
    u, s, v = torch.svd(weight)
    assert u.shape[0] == u.shape[1] == s.shape[0] == s.shape[1] == v.shape[0] == v.shape[1] == projected_dm
    
    projection_in = u[:, :dm] # f.e. projected_dm=512, than shape=[512, 256]
    projection_out = v[:dm, :] # f.e. projected_dm=512, than shape=[256, 512]
    projected_weight = s # f.e. projected_dm=512, than shape=[512, 512]
    
    return projection_in, projection_out, projected_weight

def add_projections(parameters:dict[str, torch.Tensor], projection, projection_t, projection_subnames, projection_t_subnames):
    for name, params in parameters.items():
        if is_in_partial_list(name, projection_subnames):
            # projection
            print(f"projection: {name}, {params.shape}, {params.requires_grad}")
            params.data.copy_(projection)
            # params.data = projection #dev coupled 
        elif is_in_partial_list(name, projection_t_subnames):
            # projection_T
            print(f"projection_T: {name}, {params.shape}, {params.requires_grad}")
            params.data.copy_(projection_t)
            # params.data = projection.T #dev coupled 
            # params.data.copy_(torch.inverse(projection).T) #dev inverted_test
            # params.data.copy_(torch.inverse(projection)) #dev inverted_test
        else:
            print(f"Not projection: {name}, {params.shape}, {params.requires_grad}")

def initialize_projections(model:torch.nn.Module, dmodel:int, projected_dmodel:int, projection:torch.Tensor):
    weight_dependent_projections = None

    if projection is None:
        print("No projection initialization")
        return
    elif projection == "svd":
        weight_dependent_projections = projection
        projection = None
    
    embedding_layer_tag = "embedding_layer."
    head_tag = "head."
    encode_block_tag = "encoder.blocks.block_"
    model_grouped = {
        embedding_layer_tag: {},
        head_tag: {},
        encode_block_tag: {

        }
    }
    
    for name, params in model.named_parameters():
        if embedding_layer_tag == name[:len(embedding_layer_tag)]:
            # print(f"embedding_layer_tag {name}") #dev
            model_grouped[embedding_layer_tag][name[len(embedding_layer_tag):]] = params
            continue
        if head_tag == name[:len(head_tag)]:
            # print(f"head_tag {name}") #dev
            model_grouped[head_tag][name[len(head_tag):]] = params
            continue
        if encode_block_tag == name[:len(encode_block_tag)]:
            # print(f"encode_block_tag {name}") #dev
            parsed_name = name[len(encode_block_tag):].split('.')
            block_number = int(parsed_name[0])
            block_component_name = ".".join(parsed_name[1:])
            if model_grouped[encode_block_tag].get(str(block_number)) is None:
                model_grouped[encode_block_tag][str(block_number)] = {}
            model_grouped[encode_block_tag][str(block_number)][block_component_name] = params
            continue
        raise Exception(f"Could not parse model into expected template, unexpected name: name")
        
    print_dict_hierarchy(model_grouped, 3) #dev

    # raise Exception(f"Good ending") #dev

    print("------------------------------init projections------------------------") #dev

    EMBEDDING_P = [
    ]
    EMBEDDING_P_T = [
        "layers.0.embedding_p.weight", 
        "layers.1.projected_layer.pe_layer_p.weight", 
    ]
    add_projections(model_grouped[embedding_layer_tag], projection,  projection.T, EMBEDDING_P, EMBEDDING_P_T)
    
    DEEMBEDDING_P = [
        "head_p.weight",
    ]
    DEEMBEDDING_P_T = [
    ]
    add_projections(model_grouped[head_tag], projection,  projection.T, DEEMBEDDING_P, DEEMBEDDING_P_T)

    BLOCK_P = [
        "block.residual_attention.layer.attention.input_projection.input_projection_p11.weight",
        "block.residual_attention.layer.attention.output_projection.output_projection_p21.weight",
        "block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p11.weight",
        "block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p21.weight",

    ]
    BLOCK_P_T = [
        "block.residual_attention.layer.attention.input_projection_out_projection_q.input_projection_p12_q.weight",
        "block.residual_attention.layer.attention.input_projection_out_projection_k.input_projection_p12_k.weight",
        "block.residual_attention.layer.attention.input_projection_out_projection_v.input_projection_p12_v.weight",
        "block.residual_attention.layer.attention.output_projection.output_projection_p22.weight",
        "block.residual_feedforward.layer.feedforward.logging_ff_pre_relu_p12.weight",
        "block.residual_feedforward.layer.feedforward.logging_ff_post_relu_p22.weight",
    ]
    for block_id, block_params in model_grouped[encode_block_tag].items():
        print(f"Block: {block_id}")
        add_projections(block_params, projection,  projection.T, BLOCK_P, BLOCK_P_T)

    print("------------------------------init projections end------------------------") #dev

    