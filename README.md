# Preference Aligned Distillation

**Efficient Preference Distillation for Personalized Text–Image Retrieval**

## Overview

**Preference Aligned Distillation (PAD)** is a framework for efficient, personalized text–image retrieval. Traditional embedding-based approaches like CLIP enable large-scale retrieval via vector similarity search but often struggle to capture abstract or persona-driven attributes (e.g., *“a gift for a mother who loves gardening”*).  

PAD distills the nuanced preference rankings from a powerful vision–language model (vLLM) into a scalable embedding-based system, combining the strengths of both approaches:  

- **Nuanced alignment:** Learns from the rich context of vLLMs.  
- **Scalable inference:** Maintains efficiency for large catalog retrieval.  

Experiments show PAD significantly outperforms traditional embedding-based baselines in persona-driven product recommendation tasks.

## Key Features

- Distills preferences from vLLMs into embeddings for efficient retrieval  
- Supports personalized and abstract text–image queries  
- Maintains scalability for large datasets  
- Demonstrated performance improvements on product recommendation tasks  

## Installation

```bash
# Clone the repository
git clone https://github.com/ericyh/Preference_Aligned_Distillation.git
cd Preference_Aligned_Distillation

# Install dependencies
pip install -r requirements.txt
