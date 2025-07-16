from nemo.collections import llm


class EKA_DeepSeekV3Config(llm.DeepSeekV3Config):
    pass 
    # currently there are no new functions or attributes for this class, this is a copy of nemo deepseekV3 config.
    # However, made this class to add modularity to the code in case in future we want to change something.