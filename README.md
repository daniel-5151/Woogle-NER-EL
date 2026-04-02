# Automatic co-occurrence generation using Named Entity Recognition and Entity Linking in Dutch Woo Document

## Abstract
Co-occurrence networks are used to represent (potential) relationships between terms, such as named entities, within a domain. Automatically generating such networks from text remains a challenging task. There already exists robust techniques for doing Named Entity Recognition (NER) and Entity Linking (EL) in text documents. This study proposes a full pipeline for recognizing persons, organization, locations, and laws in Woogle, a search engine for open Dutch government documents, to automatically generate co-occurrence networks for these entity types. The findings demonstrate that the developed pipeline can recognize entities with an f1-score of 0.7, while the EL component achieves a micro f1-score of 0.74, leaving further room for improvements. This paper also showed some evaluations of the co-occurrence networks that have been created.

## Web Application
Interactive web application where users can paste (Dutch) text into the textbox and recieve Linked entities from the Knowledge Base. The models and Knowledge Base are stored externally on Hugging Face

### Run with Docker
**1. Build image**
docker build -t ner-el-app .

**2. Run container**
docker run -p 7860:7860 ner-el-app

**3. Open in browser**
http://127.0.0.1:7860/


## Tech Used
**Front end**: HTML, CSS, JavaScript

**Back end**: Python, FastAPI

**Models**: Spacy, sententence_transformers

**Database**: SQlite3

**Additional packages**: Pandas, huggingface_hub. Numpy

