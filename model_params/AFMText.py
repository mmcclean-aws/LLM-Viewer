from easydict import EasyDict


model_params={
    "AFMText-30B":EasyDict(
        num_hidden_layers=48, hidden_size=7168, num_attention_heads=56, intermediate_size=23296, num_key_value_heads=8,vocab_size=102400 
    )
}