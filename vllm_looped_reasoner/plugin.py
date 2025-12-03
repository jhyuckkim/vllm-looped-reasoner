def register():
    from vllm import ModelRegistry
    from vllm.transformers_utils import config as vllm_config_mod
    from vllm_looped_reasoner.configurations import ReasonerConfig, LoopedReasonerConfig

    if "reasoner" not in vllm_config_mod._CONFIG_REGISTRY:
        vllm_config_mod._CONFIG_REGISTRY["reasoner"] = ReasonerConfig
    
    if "looped_reasoner" not in vllm_config_mod._CONFIG_REGISTRY:
        vllm_config_mod._CONFIG_REGISTRY["looped_reasoner"] = LoopedReasonerConfig

    if "LoopedReasonerForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "LoopedReasonerForCausalLM",
            "vllm_looped_reasoner.looped_reasoner:LoopedReasonerForCausalLM",
        )
    
    if "ReasonerForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "ReasonerForCausalLM",
            "vllm_looped_reasoner.reasoner:ReasonerForCausalLM",
        )