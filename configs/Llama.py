
def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")

def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")

def get_num_key_value_heads(model_params):
    return getattr(model_params, "num_key_value_heads")

def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]

def get_collective_layers(tp_size, use_sequence_parallelism):
    if tp_size > 1:
        if use_sequence_parallelism:
            return ["attn_all_gather", "attn_reduce_scatter", "mlp_all_gather", "mlp_reduce_scatter"]
        else:
            return ["attn_all_reduce", "mlp_all_reduce"]
    else:
        return []

def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")

def get_intermediate_size(model_params):
    return getattr(model_params, "intermediate_size")

def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")

def post_process(model_params,args):
    hiddensize=get_hidden_size(model_params)
    vocab_size=get_vocab_size(model_params)
    layers=[]
    for stage in ["prefill", "decode"]:
        layers.append({
            'name': 'lm_head',
            'stage':stage,
            'OPs':args['batchsize']*hiddensize*vocab_size*1,
            'load_weight':hiddensize*vocab_size *args['w_byte'],
            'load_act':hiddensize*args['a_byte'],
            'store_act':vocab_size*args['a_byte'],
        })
    return layers

def get_linear_layers(model_params, tp_size: int):
    hidden_size=get_hidden_size(model_params)
    intermediate_size=get_intermediate_size(model_params)
    key_value_heads=get_num_key_value_heads(model_params)
    attention_heads=get_num_attention_heads(model_params)
    
    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0
        #assert key_value_heads % tp_size == 0
    
    return {
        "q_proj":[hidden_size, hidden_size // tp_size],
        "k_proj":[hidden_size, hidden_size * key_value_heads // attention_heads // tp_size],
        "v_proj":[hidden_size, hidden_size * key_value_heads // attention_heads // tp_size],
        "out_proj":[hidden_size // tp_size, hidden_size],
        "gate_proj":[hidden_size, intermediate_size // tp_size],
        "up_proj":[hidden_size,intermediate_size // tp_size],
        "down_proj":[intermediate_size // tp_size, hidden_size],
    }


def build_transformer_layer_graph(tp_size: int = 1, use_sequence_parallelism: bool = False, use_flashattention: bool = False):
    """Build transformer layer graph with configurable attention implementation.
    
    Args:
        use_flashattention: Whether to use fused flash attention or standard attention
        
    Returns:
        Dict mapping node names to their input dependencies
    """
    # Common nodes for both implementations
    graph = {
        "input": [],
        "attn_norm": ["input"]
    }

    if tp_size > 1 and use_sequence_parallelism:
        graph.update({
            "attn_all_gather": ["attn_norm"],
            "q_proj": ["attn_all_gather"],
            "k_proj": ["attn_all_gather"],
            "v_proj": ["attn_all_gather"],
        })
    else:
        graph.update({
            "q_proj": ["attn_norm"],
            "k_proj": ["attn_norm"],
            "v_proj": ["attn_norm"],
        })

    # Attention implementation specific nodes
    if use_flashattention:
        graph.update({
            "fused_attention": ["q_proj", "k_proj", "v_proj"],
            "out_proj": ["fused_attention"],
        })
    else:
        graph.update({
            "qk_matmul": ["q_proj", "k_proj"],
            "softmax": ["qk_matmul"],
            "sv_matmul": ["softmax", "v_proj"],
            "out_proj": ["sv_matmul"],
        })

    if tp_size > 1:
        if use_sequence_parallelism:
            graph.update({
                "attn_reduce_scatter": ["out_proj"],
                "attn_add": ["input", "attn_reduce_scatter"],
                "mlp_norm": ["attn_add"],
                "mlp_all_gather": ["mlp_norm"],
                "gate_proj": ["mlp_all_gather"],
                "up_proj": ["mlp_all_gather"],                
                "mlp_act": ["up_proj", "gate_proj"],
                "down_proj": ["mlp_act"],
                "mlp_reduce_scatter": ["down_proj"],
                "mlp_add": ["attn_add", "mlp_reduce_scatter"],
                "output": ["mlp_add"],
            })
        else:
            graph.update({
                "attn_all_reduce": ["out_proj"],
                "attn_add": ["input", "attn_all_reduce"],
                "mlp_norm": ["attn_add"],
                "gate_proj": ["mlp_norm"],
                "up_proj": ["mlp_norm"],
                "mlp_act": ["up_proj", "gate_proj"],
                "down_proj": ["mlp_act"],
                "mlp_all_reduce": ["down_proj"],
                "mlp_add": ["attn_add", "mlp_all_reduce"],
                "output": ["mlp_add"],
            })
    else:
        graph.update({
            "attn_add": ["input", "out_proj"],
            "mlp_norm": ["attn_add"],
            "gate_proj": ["mlp_norm"],
            "up_proj": ["mlp_norm"],
            "mlp_act": ["up_proj", "gate_proj"],
            "down_proj": ["mlp_act"],
            "mlp_add": ["attn_add", "down_proj"],
            "output": ["mlp_add"],
        })

    return graph
