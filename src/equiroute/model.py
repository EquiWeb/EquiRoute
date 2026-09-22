"""The sole base-model contract supported by the initial release."""

FUNCTIONGEMMA_MODEL_ID = "google/functiongemma-270m-it"
FUNCTIONGEMMA_REVISION = "39eccb091651513a5dfb56892d3714c1b5b8276c"

FUNCTIONGEMMA_TEMPLATE_ID = "stage2-functiongemma-native-v1"
FUNCTIONGEMMA_LORA_TARGET_MODULES = ("q_proj", "v_proj")
FUNCTIONGEMMA_LORA_DROPOUT = 0.0
FUNCTIONGEMMA_LORA_BIAS = "none"
