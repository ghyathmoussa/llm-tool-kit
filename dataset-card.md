---
language:
- ar
language_bcp47:
- ar-Arab
license: mit
multilinguality:
- monolingual
size_categories:
- 1M<n<10M
source_datasets:
- original
task_categories:
- text-generation
- text-classification
- token-classification
- fill-mask
- question-answering
- summarization
- translation
- other
task_ids:
- language-modeling
- text-generation
- diacritization
- arabic-nlp
---

# Arabic NLP Toolkit Dataset

## Dataset Description

- **Repository:** [LLM-Tool-Kit](https://github.com/ghyathmoussa/llm-tool-kit)
- **Paper:** [Tashkeela: Novel corpus of Arabic vocalized texts](https://doi.org/10.1016/j.dib.2017.01.011)
- **Point of Contact:** [ghyathmoussa](https://github.com/ghyathmoussa)

### Dataset Summary

This dataset contains processed Arabic text data from the LLM-Tool-Kit project, which focuses on Natural Language Processing (NLP) for Arabic text. The dataset is derived from the Tashkeela corpus, a comprehensive collection of Arabic vocalized texts, and includes both classical and modern Arabic texts that have been processed and prepared for various NLP tasks.

### Supported Tasks and Leaderboards

This dataset supports multiple Arabic NLP tasks:

- **Text Generation**: Training language models for Arabic text generation
- **Diacritization**: Adding Arabic diacritical marks to text
- **Token Classification**: Named entity recognition and part-of-speech tagging
- **Text Classification**: Document classification and sentiment analysis
- **Question Answering**: Arabic Q&A systems
- **Summarization**: Arabic text summarization
- **Translation**: Arabic translation tasks

### Languages

The dataset contains Arabic text in both:
- **Classical Arabic**: Historical and religious texts
- **Modern Standard Arabic**: Contemporary texts and web content

## Dataset Structure

### Data Instances

The dataset contains processed text data in JSONL format with the following structure:

```json
{
    "text": "Arabic text content with diacritical marks",
    "source_file": "original source information",
    "chunk_id": "id of chunk from the page",
    "chunk_id": "Token Count"
}
```

### Data Fields

- **text**: The main Arabic text content with diacritical marks
- **metadata**: Additional information about the text source and properties

## Dataset Creation

### Source Data

The dataset is based on the Tashkeela corpus, which includes:

**Classical Arabic Sources:**
- Arabic al-Shamela library (97 books from 7079 total books)
- Historical and religious texts
- Classical literature

**Modern Standard Arabic Sources:**
- 20 modern books
- Web-crawled content from:
  - learning.aljazeera.net
  - al-kalema.org
  - enfal.de
- Manually diacritized texts

### Annotations

The original Tashkeela corpus contains:
- **Total Words**: 75,629,921
- **Arabic Vocalized Words**: 67,287,202
- **Punctuation and Non-Arabic Words**: 8,342,719
- **Unique Unvocalized Words**: 486,524
- **Unique Vocalized Words**: 998,538
- **Unique Semi-vocalized Words**: 770,702
- **Average Vocalizations per Word**: 2.05

### Personal and Sensitive Information

This dataset contains classical and modern Arabic texts. Users should be aware that:
- Classical texts may contain religious content
- Modern texts may include contemporary political or social content
- All texts are publicly available sources

## Additional Information

### Dataset Curators

- **Original Tashkeela Corpus**: Taha Zerrouki
- **Processing and Toolkit**: ghyathmoussa

### Licensing Information

- **Dataset License**: MIT License
- **Original Tashkeela License**: See COPYING.txt in the original corpus

### Citation Information

```bibtex
@article{zerrouki2017tashkeela,
  title={Tashkeela: Novel corpus of Arabic vocalized texts, data for auto-diacritization systems},
  author={Zerrouki, Taha and Balla, Amar},
  journal={Data in Brief},
  volume={10},
  pages={147--151},
  year={2017},
  publisher={Elsevier},
  doi={10.1016/j.dib.2017.01.011}
}
```

### Contributions

Contributions to the LLM-Tool-Kit project are welcome! Please see the [GitHub repository](https://github.com/ghyathmoussa/llm-tool-kit) for contribution guidelines.

### Known Limitations

- The dataset focuses on formal Arabic text (Classical and Modern Standard Arabic)
- Dialectal Arabic is not included
- Text quality depends on the original sources
- Processing may introduce artifacts in very long texts

### Social Impact of Dataset

This dataset supports:
- Arabic language preservation and digitization
- Development of Arabic NLP tools and applications
- Research in Arabic computational linguistics
- Educational applications for Arabic language learning
- Cultural heritage preservation through digital means

### Discussion of Biases

- The dataset primarily contains formal Arabic texts
- Classical texts may reflect historical perspectives
- Modern texts may have contemporary biases
- Users should be aware of the cultural and historical context of the texts

## Additional Resources

- **Project Repository**: [LLM-Tool-Kit](https://github.com/ghyathmoussa/llm-tool-kit)
- **Original Tashkeela Corpus**: [SourceForge](https://sourceforge.net/projects/tashkeela/)
- **Documentation**: See the project README for detailed usage instructions
- **Tools**: The toolkit includes utilities for tokenization, embedding training, and model fine-tuning.