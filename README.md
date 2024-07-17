# Projetos de Ciência de Dados com Python

<img src="https://github.com/msftro/MLFlow/assets/145237548/f8848c13-34e4-4141-8593-b753a131eb5f">

## Resumo

Este repositório contém os códigos e materiais complementares para o livro "Projetos de Ciência de Dados com Python" de Stephen Klosterman.

## Capítulo 1: Exploração e limpeza de Dados

Objetivos do Aprendizado:

- Descrever o contexto empresarial dos dados do estudo de caso e sua adequação à tarefa.
- Executar operações de limpeza de dados.
- Examinar sínteses estatísticas e visualizar os dados do estudo de caso.
- Implementar a codificação one-hot (one-hot encoding) em variáveis categóricas.

## Capítulo 2: Scikit-Learn e avaliação de modelo

Objetivos do Aprendizado:

- Explicar a variável resposta.
- Descrever as implicações de dados desbalanceados na classificação binária.
- Dividir os dados em conjuntos de treinamento e teste.
- Descrever o ajuste do modelo no scikit-learn.
- Derivar várias métricas para a classificação binária.
- Criar uma curva ROC e uma curva precision-recall.

## Capítulo 3: Detalhes da regressão logística e exploração de características

Objetivos do Aprendizado:

- Criar *list comprehensions* em Python
- Descrever como funciona a regressão logística
- Formular as versões sigmóide e logit da regressão logística
- Utilizar a seleção univariada para encontrar características importantes
- Definir o limite de decisão linear de uma regressão logística

## Capítulo 4: O *trade-off* entre viés e variância

Objetivos do Aprendizado:

- Descrever a função custo de perda logarítimica da regressão logística
- Implementar o procesimento de gradiente descendente para estimar parâmetros do modelo
- Articular suposições estatísticas formais do modelo de regressão logística
- Caracterizar o *trade-off* entre viés e variância e usá-lo para melhorar os modelos
- Formular o lasso e a regularização ridge e usá-los no scikit-learn
- Projetar uma função para selecionar hiperparâmetros de regularização cruzada
- Criar características de interação por engenharia para melhorar um modelo de subajuste

## Capítulo 5: Árvores de decisão e florestas aleatórias

Objetivos do Aprendizado:

- Treinar um modelo de árvore de decisão no scikit-learn
- Usar o Graphviz para visualizar um modelo de árvore de decisão treinado
- Formular as funções custo usadas para dividir nós em uma árvore de decisão
- Executar uma busca de hiperparâmetro em grade usando a validação cruzada com funções do scikit-learn
- Treinar um modelo de floresta aleatória no scikit-learn
- Avaliar as características mais importantes em um modelo de floresta aleatória
  
## Bibliotecas Requeridas

Crie um arquivo chamado `requirements.txt` no repositório com o seguinte conteúdo:

```plaintext
pandas
feature_engine
scikit-learn
```

Isso garantirá que todas as bibliotecas necessárias sejam instaladas corretamente quando o usuário executar o comando:

```
pip install -r requirements.txt
```

## Contribuição

Se você deseja contribuir com este projeto, por favor, faça um *fork* do repositório e envie um *pull request* com suas alterações.

## Contato

Se você tiver alguma dúvida ou sugestão, sinta-se à vontade para abrir uma issue ou entrar em contato:

- LinkedIn: [Márcio Ferreira](https://www.linkedin.com/in/ms-ferreira)
