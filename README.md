# Modelos de Estudo e testes - Machine Learning

## 📌 Visão Geral
Este repositório contém um projeto de Rede Neural Recorrente (RNN) com unidades GRU (Gated Recurrent Unit) implementado em Python usando Jupyter Notebook. O modelo foi treinado com uma quantidade limitada de dados devido a restrições computacionais, e incorpora a métrica de Masked Loss para lidar com sequências de comprimento variável.

## 🧠 Tecnologias Utilizadas
Python 3.x
Jupyter Notebook
TensorFlow/Keras
Bibliotecas padrão para ML (NumPy, Pandas)

## 🚀 Como Executar o Modelo
Pré-requisitos
Python 3.7 ou superior
TensorFlow 2.x instalado

## Opção 1: Carregar o modelo para inferência
```from tensorflow.keras.models import load_model

### Carregar o modelo pré-treinado
model = load_model('caminho/para/seu/modelo.h5', compile=False)

### Configurar o modelo para inferência (ajuste conforme sua arquitetura)
encoder_model = Model(inputs=model.input[0], outputs=model.get_layer('encoder_outputs').output)
decoder_model = ... # Configurar o decoder para inferência```



## Opção 2: Continuar o treinamento
```from tensorflow.keras.models import load_model

### Carregar o modelo com opção para continuar treinamento
model = load_model('caminho/para/seu/modelo.h5')

### Compilar o modelo (especifique seu optimizer e loss)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

### Continuar treinamento
model.fit(novos_dados, novos_rotulos, epochs=10, batch_size=32)```

## 📝 Notas Importantes
O arquivo .h5 contém:
Arquitetura do modelo Seq2Seq com atenção
Pesos treinados
Configurações do otimizador (se salvou com compile=True)

Para usar corretamente:
Certifique-se de pré-processar seus dados da mesma forma que durante o treinamento
O mecanismo de atenção requer configuração específica para inferência

Se encontrar erros ao carregar:

```# Pode ser necessário custom_objects se usou layers customizados
model = load_model('modelo.h5', custom_objects={'AttentionLayer': AttentionLayer})```