from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.acfql_gru_ablation_online import ACFQLAgent_GRUAblationOnline
from agents.acfql_transformer_ablation_online import ACFQLAgent_TransformerAblationOnline
from agents.acfql_ablation_online import ACFQLAgent_AblationOnline
from agents.acfql_gru_ablation_online_sac import ACFQLAgent_GRUAblationOnlineSAC
from agents.acfql_transformer_ablation_online_sac import ACFQLAgent_TransformerAblationOnlineSAC

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    acfql_gru_ablation_online = ACFQLAgent_GRUAblationOnline,
    acfql_transformer_ablation_online = ACFQLAgent_TransformerAblationOnline,
    acfql_ablation_online = ACFQLAgent_AblationOnline,
    acfql_gru_ablation_online_sac = ACFQLAgent_GRUAblationOnlineSAC,
    acfql_transformer_ablation_online_sac = ACFQLAgent_TransformerAblationOnlineSAC
)
