# Projeto Final de SEM5952 Redes Neurais e Aprendizagem de MÁquina: "Aplicação e Avaliação de Redes Neurais para Reconhecimento de Células Sanguíneas com a Base de Dados BloodMNIST".

## Resumo do Projeto  
Este projeto envolve a aplicação e avaliação de redes neurais para o reconhecimento de células sanguíneas periféricas usando a base de dados BloodMNIST, com 17.092 imagens coloridas classificadas em oito categorias. A primeira tarefa é implementar uma MLP com uma camada intermediária e avaliar sua acurácia e matriz de confusão. A segunda tarefa requer a construção de uma CNN simples e a avaliação da acurácia em função da quantidade e do tamanho dos kernels. Na terceira tarefa, a melhor configuração da CNN é re-treinada, e sua matriz de confusão, acurácia global e erros de classificação são analisados. Por fim, a quarta tarefa explora uma CNN mais profunda, avaliando seu desempenho e comparando os modelos estudados. A atividade exige justificativas detalhadas das escolhas feitas para garantir a reprodutibilidade da metodologia.

## Descriçãodo Projeto  

### Introdução
Nesta atividade, vamos abordar o problema de reconhecimento de células sanguíneas periféricas utilizando a base de dados BloodMNIST [Acevedo et al., 2020, Yang et al., 2021] (https://medmnist.com/), a qual possui 17.092 imagens microscópicas coloridas (3 canais de cor).  A Figura 1 exibe uma amostra de cada classe existente na base de dados considerando a versão com resolução de 28 × 28 pixels.  O mapeamento entre os identificadores das classes e os rótulos está indicado na Tabela 1.

<img width="576" height="288" alt="image" src="https://github.com/user-attachments/assets/3e5a56ff-085f-40da-8314-74a0a5c6bed2" />

**Figura 1:** Amostras da base de dados BloodMNIST.

Os arquivos que pessam mais de 25MB estao neste link de Google Drive para descarregar se precisar:
https://drive.google.com/drive/folders/16ZttcLSuQrm79rESBFPmRGaPbxXLRKX7?usp=drive_link

Os 4 arquivos para assistir as corridas dos 4 items do projeto também estao no link Google Drive de acima, sao:
- item a (10x).mp4   a velocidade do video real foi aumentada 10 vezes, ou seja que na realidade demorou 210 segundos.
- item b (100x).mp4  a velocidade do video real foi aumentada 10 vezes, ou seja que na realidade demorou 250 segundos.
- item c (10x).mp4   a velocidade do video real foi aumentada 10 vezes, ou seja que na realidade demorou 410 segundos.
- item d (100x).mp4  a velocidade do video real foi aumentada 10 vezes, ou seja que na realidade demorou 560 segundos.

Se vai executar o projeto no Google Colab, cada um dos 5 arquivos de Python .ipynb que vai correr devem ter estas 3 linhas de codigo primeiro:
```
!pip install medmnist
from google.colab import drive
drive.mount('/content/drive')
```


## 1. Objetivo do Projeto  
Desenvolver um sistema capaz de realizar a sumarização automática de múltiplos artigos científicos nas áreas de Inteligência Artificial (IA), Processamento de Linguagem Natural (PLN) e Aprendizado de Máquina (ML), utilizando o dataset SurveySum. A proposta é criar uma solução que gere uma seção de revisão de literatura baseada em um conjunto de artigos científicos relacionados a um tema específico, como "Mitigação de Viés em Modelos de Linguagem" ou "Modelos Transformers para PLN".

## 2. Escopo do Projeto  

O projeto será conduzido em etapas práticas e estruturadas, abrangendo desde a exploração dos dados até a avaliação dos resultados:

### Exploração do Dataset SurveySum
- Analisar e entender a estrutura do dataset SurveySum, selecionando um conjunto específico de seções para treinar e avaliar o modelo.
- Concentrar-se em temas delimitados para a sumarização, como "Transformers em PLN" ou "Redes Neurais Recorrentes em IA", com foco na relevância e coerência temática.

### Modelo de Sumarização com Pegasus-X
- Utilizar o modelo de linguagem pré-treinado Pegasus-X, que suporta contextos extensos (até 16k tokens), permitindo a sumarização de textos longos sem necessidade de truncamento excessivo.
- Implementar um pipeline de processamento que inclua:
  - **Extração de Chunks Relevantes**: Filtragem das partes mais informativas dos artigos científicos com base em similaridade semântica (SciBERT).
  - **Sumarização Abstrativa**: Geração do resumo final utilizando o Pegasus-X, integrando as referências (BIBREFs) dos artigos para formar um survey coeso.

### Avaliação da Qualidade
- Avaliar a qualidade das sumarizações geradas utilizando métricas automáticas como F1-Score com embeddings (SciBERT), G-Eval e Check-Eval semântico, garantindo uma análise mais abrangente da fidelidade semântica e da cobertura dos conceitos chave.
- Comparar o desempenho do sistema com as seções do survey original do dataset para verificar a precisão, consistência e relevância dos resumos.

### Relatório Final
- Documentar o pipeline implementado e os resultados obtidos, discutindo as limitações e propondo melhorias para trabalhos futuros.
- Analisar a eficácia do modelo em gerar resumos coerentes e relevantes, com base na consolidação de múltiplos documentos científicos, e apresentar possibilidades de expansão para tarefas mais complexas.
