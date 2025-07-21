# **Projeto Final de SEM5952 Redes Neurais e Aprendizagem de MÁquina:** _"Aplicação e Avaliação de Redes Neurais para Reconhecimento de Células Sanguíneas com a Base de Dados BloodMNIST"._

## Resumo do Projeto  
Este projeto envolve a aplicação e avaliação de redes neurais para o reconhecimento de células sanguíneas periféricas usando a base de dados BloodMNIST, com 17.092 imagens coloridas classificadas em oito categorias. A primeira tarefa é implementar uma MLP com uma camada intermediária e avaliar sua acurácia e matriz de confusão. A segunda tarefa requer a construção de uma CNN simples e a avaliação da acurácia em função da quantidade e do tamanho dos kernels. Na terceira tarefa, a melhor configuração da CNN é re-treinada, e sua matriz de confusão, acurácia global e erros de classificação são analisados. Por fim, a quarta tarefa explora uma CNN mais profunda, avaliando seu desempenho e comparando os modelos estudados. A atividade exige justificativas detalhadas das escolhas feitas para garantir a reprodutibilidade da metodologia.

## Declaração do problema a ser resolvido  

### Introdução
Nesta atividade, vamos abordar o problema de reconhecimento de células sanguíneas periféricas utilizando a base de dados BloodMNIST [Acevedo et al., 2020, Yang et al., 2021] (https://medmnist.com/), a qual possui 17.092 imagens microscópicas coloridas (3 canais de cor).  A Figura 1 exibe uma amostra de cada classe existente na base de dados considerando a versão com resolução de 28 × 28 pixels.  O mapeamento entre os identificadores das classes e os rótulos está indicado na Tabela 1.

<img width="576" height="288" alt="image" src="https://github.com/user-attachments/assets/3e5a56ff-085f-40da-8314-74a0a5c6bed2" />

**Figura 1:** Amostras da base de dados BloodMNIST. 

<br/><br/>

- ID=0 --> Basófilos
- ID=1 --> Eosinófilos
- ID=2 --> Eritroblastos
- ID=3 --> Granulocitos imaturos
- ID=4 --> Linfocitos
- ID=5 --> Monocitos
- ID=6 --> Neutrófilos
- ID=7 --> Plaquetas.
  
 **Tabela 1:** Correspondência entre os identificadores numéricos das classes e os tipos de células sanguíneas.

### Atividades
- **ITEM A)** Aplique uma rede MLP com uma camada intermediária e análise (1) a acurácia e (2) a matriz de confusão para os dados de teste obtidas pela melhor versão desta rede.  Descreva a metodologia e a arquitetura empregada, bem como todas as escolhas feitas.
- **ITEM B)** Monte uma CNN simples contendo: _(i)_ uma camada convolucional com função de ativação não-linear; _(ii)_ uma camada de pooling; _(iii)_ uma camada de saída do tipo softmax. Avalie a progressão da acurácia junto aos dados de validação em função ao:
Da quantidade de kernels utilizados na camada convolucional;
Do tamanho do kernel de convolucão.
- **ITEM C)** Escolhendo, então, a melhor configuração para a CNN simples, refaça o treinamento do modelo e apresente:
  +  A matriz de confusão para os dados de teste;
  +  A acurácia global; 
  +  Cinco padrões de teste que foram classificados incorretamente, indicando a classe esperada e as probabilidades estimadas pela rede.
  Discuta os resultados obtidos.
- **ITEM D)** Explorar, agora, uma CNN um pouco mais profunda. Descrever a arquitetura utilizada e apresente os mesmos resultados solicitados no ITEM (C) para o conjunto de teste. Por fim, façer uma breve comparação entre os modelos estudados neste exercício. Pode ser interessante explorar ideias ou elementos característicos de algumas CNNs famosas (como as ResNets ou as DenseNets).

### Referências
- [Yang et al., 2021] J. Yang, R. Shi, D. Wei, z. Liu, L. Zhao, B. Ke, H. Pfister, B. Ni, _MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification._ arXiv preprint arXiv:2110.14795, 2021.
- [Acevedo et al., 2020] A. Acevedo, A. Merino, S. Alférez, A. Molina, L. Boldú, J. Rodellar, _A dataset of microscopic peripheral blood cell images for development of automatic recognition systems_, Data in Brief, vol. 30, 2020.

<br/><br/><br/><br/>

> [!NOTA]
> A continuacao, se mostram informações úteis que os usuários devem saber, mesmo ao ler o conteúdo rapidamente.

## Indicações pela descarga e execução dos arquivos deste repositório 
Os arquivos que pessam mais de 25MB estao neste link de Google Drive para descarregar se precisar:
https://drive.google.com/drive/folders/16ZttcLSuQrm79rESBFPmRGaPbxXLRKX7?usp=drive_link

Os 4 arquivos para assistir as corridas dos 4 items do projeto também estao no link Google Drive de acima, sao:
1. item a (10x).mp4   a velocidade do video real foi aumentada 10 vezes, ou seja que na realidade demorou 210 segundos.
2. item b (100x).mp4  a velocidade do video real foi aumentada 10 vezes, ou seja que na realidade demorou 250 segundos.
3. item c (10x).mp4   a velocidade do video real foi aumentada 10 vezes, ou seja que na realidade demorou 410 segundos.
4. item d (100x).mp4  a velocidade do video real foi aumentada 10 vezes, ou seja que na realidade demorou 560 segundos.

Se vai executar o projeto no Google Colab, cada um dos 5 arquivos de Python .ipynb que vai correr devem ter estas 3 linhas de codigo primeiro:
```
!pip install medmnist
from google.colab import drive
drive.mount('/content/drive')
```
