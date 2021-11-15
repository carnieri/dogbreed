# Relatório de Teste Prático de ML

## Parte 0 - Análise exploratória

Comecei o projeto escrevendo um script para fazer download do dataset e extraí-lo: `download_dataset.py`, para facilitar a reprodução dos treinos e testes.

Em seguida, fiz uma análise exploratória dos dados. Vi a quantidade de arquivos em cada diretório do dataset. Pelo file explorer do meu sistema operacional, visualizei várias imagens para ter uma ideia do conteúdo. Percebi que havia algumas imagens bem escuras, que sob análise detalhada percebi serem completamente pretas. Criei um script para remover essas imagens da pasta train. Percebi também que havia imagens duplicadas, não só dentro de um diretório, como também entre diretórios. Algumas das duplicatas eram porque havia múltiplos cachorros na imagem, de raças diferentes; essas imagens mantive. Já as imagens duplicadas dentro de um mesmo diretório, removi. Criei um arquivo `clean_dataset.py` para juntar essa funcionalidade e a de remover imagens pretas.

Em seguida, examinei a quantidade de amostras por classe no train. A classe com menos amostras tem 145 amostras, e a com mais amostras, 252. Julguei que estão relativamente balanceadas, e portanto não usei nenhuma técnica para tratar class imbalance.

Continuando a análise exploratória, gerei um histograma de *aspect ratios* das imagens, que vão de aproximadamente 0.5 a 1.5. Por ser uma faixa de valores razoavelmente pequena, julguei não ser necessário fazer nada especial para tratar *aspect ratios* muito diferentes.

## Parte 1

Escolhi usar a biblioteca de deep learning [fastai](https://github.com/fastai/fastai), que usa o PyTorch, devido a algumas facilidades que ela proporciona, como bons defaults para fine-tuning de redes e facilidade de encontrar um learning rate perto do ótimo (método lr_find do objeto a ser treinado).

Para a Parte 1 do Teste Prático, treinei alguns classificadores para as 100 raças, usando redes pré-treinadas no ImageNet. As imagens do ImageNet tem características parecidas com as do dataset de cachorros; por exemplo, o objeto de interesse geralmente está no centro da imagem. Julguei então que uma rede pré-treinada no ImageNet seria um bom ponto de partida. Treinei algumas redes diferentes, como Resnet50, alguns tamanhos de DenseNet, e por fim a rede 'resnext50_32x4d'. Essa foi a que obteve o melhor desempenho, com cerca de 93.6% de acurácia no validation set com 5 épocas de treinamento, com train/validation split de 0.8/0.2. Julguei ser um bom resultado, considerando que há 100 classes e muitas raças de cachorros são parecidas entre si. Caso fosse necessário aumentar a acurácia, eu tentaria os seguintes métodos: treinar por mais épocas; usar data augmentation; usar uma rede com maior resolução de entrada; usar test-time augmentation; usar *ensemble* de diferentes modelos de classificação; observar as imagens com maior loss visando encontrar problemas de labels errados ou de outros tipos no dataset de treinamento.

O processo de treinamento e resultados da Parte 1 estão disponíveis no notebook [train.ipynb](https://github.com/carnieri/dogbreed/blob/master/train.ipynb).

## Parte 2

Na Parte 2, identifiquei que seria interessante usar a rede treinada na Parte 1 para calcular um embedding (vetor descritor, ou vetor de features) de cada imagem, e usar esses embeddings para fazer o enroll de novas imagens e classificação das imagens de teste. A justificativa é que a rede da Parte 1 deve ter aprendido features que conseguem distinguir entre as 100 raças de cachorros, e portanto essas mesmas features podem ser usadas na Parte 2. Removi as últimas camadas da rede resnext50 treinada na Parte 1, logo depois de uma operação AdaptiveAvgPool2d, obtendo assim um descritor com 2048 dimensões para cada imagem. Julguei o número de dimensões um pouco grande, porém adequado.

Uma alternativa para a Parte 2 seria modelar como um problema de aprender a similaridade entre amostras. Poderia ser treinada uma nova rede, utilizando as amostras das 100 raças, mas usando triplet loss ou constrastive loss. A ideia dessas funções de loss é fazer a rede aprender uma métrica, dando uma distância pequena para pares de amostras "parecidas" (da mesma raça) e uma distância grande para pares de amostras diferentes (raças diferentes). Porém, optei por usar embedding com a rede da Parte 1, por já estar treinada e ser uma solução mais simples ("always start with the simplest thing that could possibly work".)

Minha primeira ideia para fazer a classificação dos descritores foi usar um SVM linear. Porém, seria necessário retreiná-lo para cada nova imagem cadastrada na base de busca. O treinamento de um SVM linear é um processo rápido, e possivelmente ficaria abaixo de 1 segundo, mas julguei que não estaria seguindo o espírito do projeto, que pede uma solução que possa ser treinada online, ou seja, que permita indexação incremental.

Mesmo assim, experimentei treinar um [SVM linear](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) com a biblioteca [scikit-learn](https://scikit-learn.org/stable/index.html), com todas as imagens da pasta enroll, e calculei a acurácia para a pasta teste, obtendo acima de 98% de acurácia. Para a Parte 3, tratei o problema de imagens de raças desconhecidas aplicando um limiar na saída de probabilidade do SVM linear. Para tanto, foi necessário treinar o SVM para que emitisse probabilidades na saída (o que não é uma funcionalidade "nativa" do SVM, e sim um processo de calibração de probabilidade feito via cross-validation).

Por fim, julgando que o aspecto de indexação incremental da Parte 2 é uma parte fundamental do problema, abandonei a solução com SVM linear e escolhi modelar como um problema de busca de similaridade de vetores, através da medição de distância entre vetores de embeddings, e o uso do algoritmo de k-nearest neighbors (k-NN) para determinar a classe vencedora. É um algoritmo simples que pode ser facilmente implementado do zero, mas visando uma solução que pudesse ser usada em escala, escolhi a biblioteca [faiss](https://github.com/facebookresearch/faiss) do Facebook, usada para busca de similaridade eficiente e clustering de vetores densos. O [faiss](https://github.com/facebookresearch/faiss) permite indexar vetores de forma incremental, como desejado. Além de busca exata de k-NN, implementa também algoritmos aproximados, que podem ser mais apropriados para uso em escala muito grande. Escolhi usar busca exata, por ser boa o suficiente para um problema desse tamanho; e escolhi usar distância cosseno ao invés de distância euclidiana, pois em um projeto anterior obtive bons resultados dessa forma.

Há uma escolha interessante a ser feita que é o valor de k, o número de vizinhos a serem considerados na busca. Valores pequenos de k podem resultar em um sistema um pouco ruidoso, pois a classe vencedora acaba dependendo de uma única amostra; em outras palavras, a fronteira de decisão fica complexa. Porém, valores grandes de k significam que o usuário terá que adicionar no mínimo floor(k/2)+1 amostras de cada classe para poder classificar novas imagens dessa classe. Por fim, escolhi usar k = 5, por ser o menor valor de k que retornou a acurácia máxima no conjunto de teste.

A acurácia de teste foi medida da seguinte forma. Primeiro, todas as imagens da pasta `enroll` foram indexadas (adicionadas ao sistema de busca). Em seguida, foi obtida a classe de predição para cada imagem da pasta `test` e comparada com a classe verdadeira. A acurácia resultante para k = 5 foi de 0.985.

O processo de treinamento e resultados da Parte 2 e 3 estão disponíveis no notebook [search.ipynb](https://github.com/carnieri/dogbreed/blob/master/search.ipynb).

## Parte 3

A minha maior dúvida em todo o Teste Prático foi como detectar anomalias (amostras de classe desconhecida) com o modelo de k-NN. Acabei utilizando duas formas em conjunto. A primeira consiste em, no processo de busca, indicar que não foi possível classificar a imagem, caso não haja maioria nas k amostras mais próximas. A segunda forma foi estabelecer um limiar de distância para que cada vizinho seja considerado próximo o suficiente; o limiar foi determinado empiricamente. Uma desvantagem de usar detecção de anomalia é que inevitavelmente aumenta-se a chance de rejeitar classificações corretas de amostras com classe conhecida. De fato, observei que a acurácia de testes caiu ao implementar as duas formas de rejeição: para k = 5 e threshold = 0.78, a acurácia de teste caiu de 0.985 para 0.952. A escolha de um limiar que represente um bom tradeoff entre recall e precision é uma parte importante de projetos como esse.

Para medir quão bem o método de rejeição funciona, criei um dataset de imagens com raça desconhecida formado pelas amostras de validação da Parte 1 (20% das imagens da pasta `train`). Em seguida, medi a "acurácia de rejeição": a classificação de uma imagem desse dataset é considerada correta se retornar "unknown", caso contrário é considerada incorreta. Para k = 5 e threshold = 0.78 (limiar de distância), a acurácia de rejeição foi de 0.82. Isso significa que a probabilidade de uma imagem de raça desconhecida ser coretamente classificada como "unknown" é de 82%.

O processo de treinamento e resultados da Parte 2 e 3 estão disponíveis no notebook [search.ipynb](https://github.com/carnieri/dogbreed/blob/master/search.ipynb).

## Repositório e web apps

Fiz upload de todo o código escrito durante o Teste Prático para https://github.com/carnieri/dogbreed .

Escolhi o web framework [streamlit](https://streamlit.io/) para demonstrar o funcionamento da Parte 1 (`webapp_part1.py`) e Parte 2 + Parte 3 (`webapp_part2.py`).
