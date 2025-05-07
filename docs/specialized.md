Model (repo) | Size | Good for | Why it helps
epfl-llm/meditron-70b | 70 B | KE / RG | SOTA clinical reasoning; already beats GPT-3.5 on MedQA, so a perfect teacher or inference backbone. Hugging Face
epfl-llm/meditron-7b | 7 B | KE / DST | Same corpus, 16-bit or 4-bit variants run on a single A100. Hugging Face
Flmc/DISC-MedLLM | 13 B | RG | Conversation-tuned for patient–doctor dialogue; drops straight into Purpose's inference module. Hugging Face
stanford-crfm/BioMedLM-2.7B | 2.7 B | DST | Lightweight GPT-style model trained only on PubMed; great for browser/mobile deployment. Hugging Face
Simonlee711/Clinical ModernBERT | 110 M | KM | Fast clinical NER or sentence-embeddings layer when you just need high-recall entity coverage. Hugging Face
microsoft/BioGPT-Large | 1.5 B | QG | Generative specialist for biomedical text; good at crafting exam-style QA pairs for distillation. Hugging Face


Model | Size | Stage | Why
MathLLMs/MathCoder-L-13B (and 7B) | 7-13 B | RG | Code-augmented math solver—great for "explain + derive" answers. Hugging Face
MathLLMs/MathCoder-CL-34B | 34 B | KE | Larger contextual window ( 16 k ) for theorem-heavy corpora. Hugging Face

# Embedding Reranker 

Model | Type | Tokens | Notes
BAAI/bge-large-en-v1.5 | dense embed | 8 k | SOTA on MTEB-retrieval; easy drop-in for KM. Hugging Face
BAAI/bge-m3 | multi-function | 8 k | One model for dense, sparse and multi-vector retrieval; > 100 languages. Hugging Face
BAAI/bge-reranker-v2-m3 | cross-encoder | 4 k | Lightweight reranker for high-recall pipelines. Hugging Face
NeuML/pubmedbert-base-embeddings-matryoshka | domain embed | 128-768 | Dynamic-dim biomedical embeddings—great when space matters. Hugging Face


# Legal Domain Models

Model | Size | Good for | Why it helps
lexlms/legal-roberta-base | 125 M | KM | Legal domain specialized encoder; excellent for document classification. Hugging Face
lexlms/legal-longformer-base | 149 M | KM | Long-context legal text encoder supporting 4096 tokens; ideal for contracts and long legal documents. Hugging Face
nile/legal-bert-base | 110 M | KE | Fine-tuned on US case law; excellent for legal entity recognition and citation linking. Hugging Face
CaseLawBERT/CaseLawBERT | 340 M | KE | Specialized for case law understanding with high precision on legal precedent identification. Hugging Face
IBM/Legal-Universe-Llama-2-7b | 7 B | RG | Legal reasoning and compliance analysis; trained on regulatory documents. Hugging Face

# Finance Domain Models

Model | Size | Good for | Why it helps
yiyanghkust/finbert-tone | 110 M | KM | Sentiment analysis for financial text; valuable for market sentiment extraction. Hugging Face
ProsusAI/finbert | 110 M | KM | Financial domain BERT with strong entity recognition for financial instruments. Hugging Face
FinGPT/fingpt-mt_llama2-7b | 7 B | RG | Multi-task financial LLM; specialized for market analysis and financial forecasting. Hugging Face
microsoft/phi-2-finance | 2.7 B | DST | Compact financial model with strong performance on specialized fiscal knowledge. Hugging Face
NVIDIA/NeMo-Megatron-Fin | 20 B | KE | Large financial model with strong regulatory knowledge and compliance capability. NGC

# Code & Technical Models

Model | Size | Good for | Why it helps
facebook/incoder-6B | 6 B | RG | Specialized for code infilling and completion; excellent for developer assistance. Hugging Face
WizardLM/WizardCoder-Python-34B | 34 B | RG | Expert-level Python code generation and explanation; outperforms many larger models. Hugging Face
codellama/CodeLlama-7b-hf | 7 B | DST | Base model for fine-tuning domain-specific code generators; supports multiple languages. Hugging Face
bigcode/starcoder2-15b | 15 B | RG | Trained on permissively licensed code; strong for enterprise integrations. Hugging Face


`# example snippet for ModelHub overrides
from purpose.utils.model_hub import PurposeAPIClient

client = PurposeAPIClient(api_token="YOUR_HF_TOKEN")
client.task_model_map.update({
    "knowledge_extraction": ["epfl-llm/meditron-7b", "FinGPT/FinGPT-Chat"],
    "knowledge_mapping": ["BAAI/bge-m3"],
    "response_generation": ["Equall/Saul-7B-Instruct-v1"],
    "distillation_target": ["microsoft/Phi-3-mini-4k-instruct"]
})
`