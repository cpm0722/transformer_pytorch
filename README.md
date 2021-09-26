### [Paper Link](https://arxiv.org/pdf/1706.03762.pdf)

# Why Transformer?

 Transformer는 2017년에 등장해 NLP 분야에서 혁신적인 성과를 이끌어낸 논문이다. 비단 NLP뿐만이 아니라 다른 ML Domain 내에서도 수없이 활용되고 있다.

 Transformer의 가장 큰 contribution은 이전의 RNN(Recurrent Neural Network) model이 불가능했던 병렬 처리를 가능케 했다는 점이다. GPU를 사용함으로써 얻는 가장 큰 이점은 병렬 처리를 한다는 것인데, RNN과 같은 model은 GPU 발전의 혜택을 제대로 누리지 못했다. 앞으로 GPU의 발전은 더욱 가속화될 것이기에, Recurrent network의 한계는 점점 더 두드러질 것이다. Recurrent network를 사용하는 이유는 텍스트, 음성 등의 sequential한 data를 처리하기 위함인데, sequential하다는 것은 등장 시점(또는 위치)을 정보로 취급한다는 의미이다. 따라서 context vector를 앞에서부터 순차적으로 생성해내고, 그 context vector를 이후 시점에서 활용하는 방식으로 구현한다. 즉, 이후 시점의 연산은 앞 시점의 연산에 의존적이다. 따라서 앞 시점의 연산이 끝나지 않을 경우, 그 뒤의 연산을 수행할 수 없다. 이러한 이유로 RNN 계열의 model은 병렬 처리를 제대로 수행할 수 없다.

 Transformer는 이를 극복했다. Attention 개념을 도입해 어떤 특정 시점에 집중하고, Positional Encoding을 사용해 sequential한 위치 정보를 보존했으며, 이후 시점에 대해 masking을 적용해 이전 시점의 값만이 이후에 영향을 미치도록 제한했다. 그러면서도 모든 과정을 병렬처리 가능하도록 구현했다. Transformer를 직접 pytorch를 사용해 구현하고, 학습시키며 이러한 특징들을 이해해보자. 본 포스트의 모든 code는 [Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html)를 참조해 작성했다.
# Prerequisite

Machine Learning에 대한 기본적인 지식(Back Propagation, Activation Function, Optimizer, Softmax, KL Divergence, Drop-out, Normalization, Regularization, RNN 등)과 NLP의 기본적인 지식(tokenizing, word embedding, vocabulary, Machine Translation, BLEU Score 등)을 안다고 가정한다. 또한 Python, pytorch를 사용해 간단한 model을 만들어낼 수 있다는 것을 전제로 한다.

# Model of Transformer

## Transformer의 개괄적인 구조

Transformer는 input sentence를 넣어 output sentence를 생성해내는 model이다. input과 동일한 sentence를 만들어낼 수도, input의 역방향 sentence를 만들어낼 수도, 같은 의미의 다른 언어로 된 sentence를 만들어낼 수도 있다. 이는 model의 train 과정에서 정해지는 것으로, label을 어떤 sentence로 정할 것인가에 따라 달라진다. 결국 Transformer는 sentence 형태의 input을 사용해 sentence 형태의 output을 만들어내는 함수로 이해할 수 있다.

![transformer_simple.png](/images/transformer_simple.png)

$$y=\text{Transformer}(x)\\x,\ y\text{ : sentence}$$

전체적인 생김새를 살펴보자.

![transformer_structure_in_paper.png](/images/transformer_structure_in_paper.png)

출처: Attention is All You Need [[https://arxiv.org/pdf/1706.03762.pdf](https://arxiv.org/pdf/1706.03762.pdf)]

Transformer는 크게 Encoder와 Decoder로 구분된다. 부수적인 다른 구성 요소들이 있으나, Encoder와 Decoder가 가장 핵심이다. Encoder는 위 그림에서 좌측, Decoder는 위 그림에서 우측을 의미한다.

Encoder와 Decoder를 자세히 분석하기 이전에, 각각을 함수 형태로 이해해보자. Encoder는 sentence를 input으로 받아 하나의 vector를 생성해는 함수이다. 이러한 과정을 Encoding이라고 한다. Encoding으로 생성된 vector는 context라고 부르는데, 말그대로 문장의 '문맥'을 함축해 담은 vector이다. Encoder는 이러한 context를 제대로 생성(문장의 정보들을 빠뜨리지 않고 제대로 압축)해내는  것을 목표로 학습된다.

![encoder_simple.png](/images/encoder_simple.png)

$$c=\text{Encoder}(x)\\x\text{ : sentence}\\c\text{ : context}$$

Decoder는 Encoder와 방향이 반대이다. context를 input으로 받아 sentence를 output으로 생성해낸다. 이러한 과정을 Decoding이라고 한다. 사실 Decoder는 input으로 context만을 받지는 않고, output으로 생성해내는 sentence를 right shift한 sentence도 함께 입력받지만, 자세한 것은 당장 이해할 필요 없이 단순히 어떤 sentence도 함께 input으로 받는 다는 개념만 잡고 넘어가자. 정리하자면, Decoder는 sentence, context를 input으로 받아 sentence를 만들어내는 함수이다.

![decoder_simple.png](/images/decoder_simple.png)

$$y=\text{Decoder}(c,z)\\y,\ z\text{ : sentence}\\c\text{ : context}$$

Encoder와 Decoder에 모두 context vector가 등장하는데, Encoder는 context를 생성해내고, Decoder는 context를 사용한다. 이러한 흐름으로 Encoder와 Decoder가 연결되어 전체 Transformer를 구성하는 것이다.

지금까지의 개념을 바탕으로 아주 간단한 Transformer model을 pytorch로 구현해보자. encoder와 decoder가 각각 완성되어 있다고 가정하고, 이를 class 생성자의 인자로 받는다.

```python
class Transformer(nn.Module):

	def __init__(self, encoder, decoder):
		super(Transformer, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x, z):
		c = self.encoder(x)
		y = self.decoder(z, c)
		return y
```

## Encoder

![encoder.png](/images/encoder.png)

 Encoder는 위와 같은 구조로 이루어져 있다. Encoder Layer가 $$N$$개 쌓여진 형태이다. 논문에서는 $$N=6$$을 사용했다. Encoder Layer는 input과 output의 형태가 동일하다. 어떤 matrix를 input으로 받는다고 했을 때, Encoder Layer가 도출해내는 output은 input과 완전히 동일한 shape를 갖는 matrix가 된다. Encoder Layer $$N$$개가 쌓여 Encoder를 이룬다고 했을 때, 첫번째 Encoder Layer의 input은 전체 Encoder의 input으로 들어오는 문장 embdding이 된다. 첫번째 layer가 output을 생성해내면 이를 두번째 layer가 input으로 사용하고, 또 그 output을 세번째 layer가 사용하는 식으로 연결되며, 가장 마지막 $$N$$번째 layer의 output이 전체 Encoder의 output, 즉, context가 된다. 이러한 방식으로 layer들이 연결되기 때문에, Encoder Layer의 input과 output의 shape는 필연적으로 반드시 동일해야만 한다. 여기서 주목해야 하는 지점은 위에서 계속 언급했던 context 역시 Encoder의 input sentence와 동일한 shape를 가진다는 것이다. 즉, 어떤 matrix가 Encoder를 거쳐간다 하더라도 최종 matrix의 shape는 처음의 것과 반드시 같다.

 Encoder는 왜 여러 개의 layer를 겹쳐 쌓는 것일까? 각 Encoder Layer의 역할은 무엇일까? 결론부터 말하자면, 각 Encoder Layer는 input으로 들어오는 vector에 대해 더 높은 차원(넓은 관점)에서의 context를 담는다. 높은 차원에서의 context라는 것은 더 추상적인 정보라는 의미이다. Encoder Layer는 내부적으로 어떠한 Mechanism을 사용해 context를 담아내는데, Encoder Layer가 겹겹이 쌓이다 보니 처음에는 원본 문장에 대한 낮은 수준의 context였겠지만 이후 context에 대한 context, context의 context에 대한 context ... 와 같은 식으로 점차 높은 차원의 context가 저장되게 된다. Encoder Layer의 내부적인 작동 방식은 곧 살펴볼 것이기에, 여기서는 직관적으로 Encoder Layer의 역할, Encoder 내부의 전체적인 구조만 이해하고 넘어가자.

지금까지의 개념을 바탕으로 Encoder를 간단하게 code로 작성해보자.

```python
class Encoder(nn.Module):

	def __init__(self, encoder_layer, n_layer):  # n_layer: Encoder Layer의 개수
		super(Encoder, self).__init__()
		self.layers = []
		for i in range(n_layer):
			self.layers.append(copy.deepcopy(encoder_layer))

	def forward(self, x):
		out = x
		for layer in self.layers:
			out = layer(out)
		return out
```

forward()를 주목해보자. Encoder Layer들을 순서대로 실행하면서, 이전 layer의 output을 이후 layer의 input으로 넣는다. 첫 layer의 input은 Encoder 전체의 input인 x가 된다. 이후 가장 마지막 layer의 output (context)를 return한다.

### Encoder Layer

![encoder_layer.png](/images/encoder_layer.png)

Encoder Layer는 크게 Multi-Head Attention Layer, Position-wise Feed-Forward Layer로 구성된다. 각각의 layer에 대한 자세한 설명은 아래에서 살펴보도록 하고, 우선은 Encoder Layer의 큰 구조만을 사용해 간단하게 구현해보자.

```python
class EncoderLayer(nn.Module):

	def __init__(self, multi_head_attention_layer, position_wise_feed_forward_layer):
		super(EncoderLayer, self).__init__()
		self.multi_head_attention_layer = multi_head_attention_layer
		self.position_wise_feed_forward_layer = position_wise_feed_forward_layer

	def forward(self, x):
		out = self.multi_head_attention_layer(x)
		out = self.position_wise_feed_forward_layer(out)
		return out
```

### What is Self-Attention?

Multi-Head Attention은 Self-Attention을 병렬적으로 여러 개 수행하는 layer이다. 때문에 Multi-Head Attention을 이해하기 위해서는 Self-Attention에 대해 먼저 알아야만 한다. Attention이라는 것은 넓은 범위의 전체 data에서 특정한 부분에 집중한다는 의미이다. 다음의 문장을 통해 Attention의 개념을 이해해보자.

> The animal didn't cross the street, because it was too tired.

 위 문장에서 'it'은 무엇을 지칭하는 것일까? 사람이라면 직관적으로 'animal'과 연결지을 수 있지만, 컴퓨터는 'it'이 'animal'을 가리키는지, 'street'를 가리키는지 알지 못한다. Self-Attention은 이러한 문제를 해결하기 위해 같은 문장 내에서 두 token 사이의 연관성을 찾아내는 방법론이다. Self가 붙는 이유는 문장 내에서 (같은 문장 내의 다른 token에 대한) Attention을 구하기 때문이다.

#### RNN vs Self-Attention

 Transformer에서 벗어나, 이전 RNN의 개념을 다시 생각해보자. RNN은 이전 시점까지 나온 token들에 대한 hidden state 내부에 이전 정보들을 저장했다. RNN의 경우 hidden state를 활용해 이번에 등장한 'it'이 이전의 'The Animal'을 가리킨다는 것을 알아낼 것이다. Self-Attention 역시 동일한 효과를 내는 것을 목적으로 하나, Recurrent Network에 비해 크게 아래와 같은 2가지 장점을 갖는다.

1. Recurrent Network는 $$i$$시점의 hidden state $$h_i$$를 구하기 위해서는 $$h_{i-1}$$가 필요했다. 결국, 앞에서부터 순차 계산을 해나가 $$h_0, h_1, ... , h_n$$을 구하는 방법밖에 없었기에 병렬 처리가 불가능했다. 하지만 Self-Attention은 모든 token 쌍 사이의 attention을 한 번의 행렬 곱으로 구해내기 때문에 손쉽게 병렬 처리가 가능하다.
2. Recurrent Network는 시간이 진행될수록 오래된 시점의 token에 대한 정보가 점차 희미해져간다. 위 문장의 예시에서 현재 'didn't'의 시점에서 hidden state를 구한다고 했을 때, 바로 직전의 token인 'animal'에 대한 정보는 뚜렷하게 남아있다. 하지만 점차 앞으로 나아갈수록, 'because'나 'it'의 시점에서는 'didn't' 시점보다는 'animal'에 대한 정보가 희미하게 남게 된다. 결국, 서로 거리가 먼 token 사이의 관계에 대한 정보가 제대로 반영되지 못하는 것이다. 반면, Self-Attention은 문장에 token이 $$n$$개 있다고 가정할 경우, $$n \times n$$ 번 연산을 수행해 모든 token들 사이의 관계를 직접 구해낸다. 중간의 다른 token들을 거치지 않고 바로 direct한 관계를 구하는 것이기 때문에 Recurrent Network에 비해 더 명확하게 관계를 잡아낼 수 있다.

#### Query, Key, Value

지금까지는 추상적으로 Self-Attention에 대한 개념 및 장단점을 살펴봤다. 이제 구체적으로 어떤 방식으로 행렬 곱셈을 사용해 Self-Attention이 수행되는지 알아보자. 우선은 matrix-level이 아닌 token 단위의 vector-level에서 이해해보자.

Self-Attention에서는 총 3개의 vector가 새로 등장한다. Query, Key, Value이다. 각각의 역할은 다음과 같다.

1. Query: 현재 시점의 token을 의미
2. Key: attention을 구하고자 하는 대상 token을 의미
3. Value: attention을 구하고자 하는 대상 token을 의미 (Key와 동일한 token)

 위의 예시 문장으로 다시 되돌아가보자. 'it'이 어느 것을 지칭하는지 알아내고자 한다. 그렇다면 'it' token과 문장 내 다른 모든 token들에 대해 attention을 구해야 한다. 이 경우에는 Query는 'it'으로 고정이다. Key, Value는 서로 완전히 같은 token을 가리키는데, 문장의 시작부터 끝까지 모든 token들 중 하나가 될 것이다. Key와 Value가 'The'를 가리킬 경우 'it'과 'The' 사이의 attention을 구하는 것이고, Key와 Value가 마지막 'tired'를 가리킬 경우 'it'과 'tired' 사이의 attention을 구하는 것이 된다. 즉, Key와 Value는 문장의 처음부터 끝까지 탐색한다고 이해하면 된다. Query는 고정되어 하나의 token을 가리키고, Query와 가장 부합하는 (Attention이 높은) token을 찾기 위해서 Key, Value를 문장의 처음부터 끝까지 탐색시키는 것이다. 각각의 의미는 이해했으나, Key와 Value가 완전히 같은 token을 가리킨다면 왜 두 개가 따로 존재하는지 의문이 들 수 있다. 이는 이후에 다룰 것이나, 결론부터 말하자면 Key와 Value의 실제 값은 다르지만 의미적으로는 여전히 같은 token을 의미한다. Key와 Value는 이후 Attention 계산 과정에서 별개로 사용된다.

 Query, Key, Value가 각각 어떤 token을 가리키는지는 이해가 됐을 것이다. 하지만, 그래서 Query, Key, Value라는 세 vector의 구체적인 값은 어떻게 만들어지는지는 우리는 아직 알지 못한다. 정말 간단하게도, input으로 들어오는 token embedding vector를 fully connected layer에 넣어 세 vector를 만들어낸다. 세 vector를 생성해내는 FC layer는 모두 다르기 때문에, 결국 self-attention에서는 Query, Key, Value를 구하기 위해 3개의 서로 다른 FC layer가 존재한다. 이 FC layer들은 모두 같은 input dimension, output dimension을 갖는다. input dimension이 같은 이유는 당연하게도 모두 다 token embedding vector를 input으로 받기 때문이다. 한편, 세 FC layer의 output dimension이 같다는 것을 통해 각각 별개의 FC layer로 구해진 Query, Key, Value가 구체적인 값은 다를지언정 같은 dimension을 갖는 vector가 된다는 것을 알 수 있다. 즉, **Query, Key, Value의 shape는 모두 동일**하다. 앞으로 이 세 vector의 dimension을 $$d_k$$로 명명한다. 여기서 $$k$$는 Key를 의미하는데, 굳이 Query, Key, Value 중 Key를 이름으로 채택한 이유는 특별히 있지 않고, 단지 논문의 notation에서 이를 채택했기 때문이다. 정리하자면, Query, Key, Value는 모두 $$d_k$$의 dimension을 갖는 vector이다. 이제 위에서 얘기했던 Key, Value가 다른 값을 갖는 이유를 이해할 수 있다. input은 같은 token embedding vector였을지라도 서로 다른 FC layer를 통해서 각각 Key, Value가 구해지기 때문에 같은 token을 가리키면서 다른 값을 갖는 것이다.

#### How to Calculate?

이제 Query, Key, Value를 활용해 Attention을 계산해보자. Attention이라고 한다면 어떤 것에 대한 Attention인지 불명확하다. 구체적으로, Query에 대한 Attention이다. 이 점을 꼭 인지하고 넘어가자. 이후부터는 Query, Key, Value를 각각 $$Q$$, $$K$$, $$V$$로 축약해 부른다. Query의 Attention은 다음과 같은 수식으로 계산된다.

$$\text{Query's Attention}\left( Q, K, V \right) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

그림으로 계산의 흐름을 표현하면 다음과 같다.

![scaled_dot_production_in_paper.png](/images/scaled_dot_production_in_paper.png){: width="50%"}

출처: Attention is All You Need [[https://arxiv.org/pdf/1706.03762.pd](https://arxiv.org/pdf/1706.03762.pdf)f]

 $$Q$$는 현재 시점의 token을, $$K$$와 $$V$$는 Attention을 구하고자 하는 대상 token을 의미했다. 우선은 빠른 이해를 돕기 위해 $$Q$$, $$K$$, $$V$$가 모두 구해졌다고 가정한다. 위의 예시 문장을 다시 가져와 'it'과 'animal' 사이의 Attention을 구한다고 해보자. $$d_k=3$$이라고 한다면, 아래와 같은 모양일 것이다.

![qkv_vector.png](/images/qkv_vector.png)

그렇다면 $$Q$$와 $$K$$를 MatMul(행렬곱)한다는 의미는 어떤 의미일까? 이 둘을 곱한다는 것은 둘의 Attention Score를 구한다는 것이다. $$Q$$와 $$K$$의 shape를 생각해보면, 둘 모두 $$d_k$$를 dimension으로 갖는 vector이다. 이 둘을 곱한다고 했을 때(정확히는 $$K$$를 transpose한 뒤 곱함, 즉 두 vector의 내적), 결과값은 어떤 scalar 값이 나오게 될 것이다. 이 값을 Attention Score라고 한다. 이후 scaling을 수행하는데, 값의 크기가 너무 커지지 않도록 $$\sqrt{d_k}$$로 나눠준다. 값이 너무 클 경우 gradient vanishing이 발생할 수 있기 때문이다. scaling을 제외한 연산 과정은 아래와 같다.

![attention_score_scalar.png](/images/attention_score_scalar.png)

 지금까지는 $$1:1$$ Attention을 구했다면, 이를 확장시켜  $$1:N$$ Attention을 구해보자. 그 전에 $$Q$$, $$K$$, $$V$$에 대한 개념을 다시 되짚어보자. $$Q$$는 고정된 token을 가리키고, $$Q$$가 가리키는 token과 가장 높은 Attention을 갖는 token을 찾기 위해 $$K$$, $$V$$를 문장의 첫 token부터 마지막 token까지 탐색시키게 된다. 즉, Attention을 구하는 연산이 $$Q$$ 1개에 대해서 수행된다고 가정했을 때, $$K$$, $$V$$는 문장의 길이 $$n$$만큼 반복되게 된다. $$Q$$ vector 1개에 대해서 Attention을 계산한다고 했을 때, $$K$$와 $$V$$는 각각 $$n$$개의 vector가 되는 것이다. 이 때 $$Q$$, $$K$$, $$V$$ vector의 dimension은 모두 $$d_k$$로 동일할 것이다. 위의 예시 문장을 다시 갖고 와 'it'에 대한 Attention을 구하고자 할 때에는 $$Q$$는 'it', $$K$$, $$V$$는 문장 전체이다. $$K$$와 $$V$$를 각각 $$n$$개의 vector가 아닌 1개의 matrix로 표현한다고 하면 vector들을 concatenate해 $$n \times d_k$$의 matrix로 변환하면 된다. 그 결과 아래와 같은 shape가 된다.

![qkv_matrix_1.png](/images/qkv_matrix_1.png)

 그렇다면 이들의 Attention Score는 아래와 같이 계산될 것이다.

![attention_score_vector.png](/images/attention_score_vector.png)

그 결과 Attention Score는 $$1 \times n$$의 matrix가 되는데, 이는 $$Q$$의 token과 문장 내 모든 token들 사이의 Attention Score를 각각 계산한 뒤 concatenate한 것과 동일하다. 이를 행렬곱 1회로 수행한 것이다.

 이렇게 구한 Attention Score는 softmax를 사용해 확률값으로 변환하게 된다. 그 결과 각 Attention Score는 모두 더하면 1인 확률값이 된다. 이 값들의 의미는 $$Q$$의 token과 해당 token이 얼마나 Attention을 갖는지(얼마나 연관성이 짙은지)에 대한 비율(확률값)이 된다. 임의로 Attention Probability라고 부른다(논문에서 사용하는 표현은 아니고, 이해를 돕기 위해 임의로 붙인 명칭이다). 이후 Attention Probability를 최종적으로 $$V$$와 곱하게 되는데, $$V$$(Attention을 구하고자 하는 대상 token, 다시 한 번 강조하지만 $$K$$와 $$V$$는 같은 token을 의미한다.)를 각각 Attention Probability만큼만 반영하겠다는 의미이다. 연산은 다음과 같이 이루어진다.

![attention_vector.png](/images/attention_vector.png)

이렇게 구해진 최종 result는 기존의 $$Q$$, $$K$$, $$V$$와 같은 dimension($$d_k$$)를 갖는 vector 1개임을 주목하자. 즉, input으로 $$Q$$ vector 1개를 받았는데, 연산의 최종 output이 input과 같은 shape를 갖는 것이다.

지금까지의 Attention 연산은 'it'이라는 한 token에 대한 Attention을 구한 것이다. 그러나 우리는 문장 내에서 'it'에 대한 Attention만 구하고자 하는 것이 아니다. 모든 token에 대한 Attention을 구해내야만 한다. 따라서 Query 역시 1개의 vector가 아닌 모든 token에 대한 matrix로 확장시켜야 한다.

![qkv_matrix_2.png](/images/qkv_matrix_2.png)

그렇다면 Attention을 구하는 연산은 아래와 같이 진행된다.

![attention_score_matrix.png](/images/attention_score_matrix.png)

![attention_matrix.png](/images/attention_matrix.png)

 이제 여기까지 왔으면 $$Q$$, $$K$$, $$V$$가 주어졌을 때에 어떻게 Attention이 계산되는지 이해했을 것이다. 주목해야 할 점은, Attention 계산에서 input($$Q$$)와 output($$Q$$'s Attention)이 같은 shape라는 것이다. 즉, Self-Attention 계산을 하나의 함수로 본다면, Self-Attention 함수는 input의 shape를 보존한다(Attention을 함수라고 했을 때 syntax 측면에서 엄밀히 따지자면 input은 $$Q$$, $$K$$, $$V$$ 총 3개이다. 하지만 개념 상으로는 $$Q$$에 대한 Attention을 의미하는 것이므로 semantic 측면에서 input은 $$Q$$라고 볼 수 있다).

![self_attention.png](/images/self_attention.png)

 그렇다면 $$Q$$, $$K$$, $$V$$는 어떻게 구해지는 것일까? Self-Attention 개념 이전에 설명했듯이, 각각 서로 다른 FC layer에 의해 구해진다. FC layer의 input은 word embedding vector들이고, output은 각각 $$Q$$, $$K$$, $$V$$이다. word embedding의 dimension이 $$d_{embed}$$라고 한다면, input의 shape는 $$n \times d_{embed}$$이고, output의 shape는 $$n \times d_k$$이다. 각각의 FC layer는 서로 다른 weight matrix ($$d_{embed} \times d_k$$)를 갖고 있기 때문에 output의 shape는 모두 동일할지라도, $$Q$$, $$K$$, $$V$$의 실제 값들은 모두 다르다.

![qkv_fc_layer.png](/images/qkv_fc_layer.png)

#### Pad Masking

 뜬금없이 masking이 왜 나오는 것일까? 사실 논문의 figure에 따르면 Attention 계산에는 masking 과정이 포함되어 있다.

![scaled_dot_production_in_paper.png](/images/scaled_dot_production_in_paper.png){: width="50%"}

출처: Attention is All You Need [[https://arxiv.org/pdf/1706.03762.pd](https://arxiv.org/pdf/1706.03762.pdf)f]

pad는 무엇을 의미하는 것일까? 예시 문장을 다시 가져와보자.

> The animal didn't cross the street, because it was too tired.

문장을 word 단위로 tokenize(단순히 python의 split() 사용)한다면 token의 개수는 총 11개이다. 만약의 각 token의 embedding dimension이 $$d_{embed}$$라고 한다면, 문장 전체의 embedding matrix는 ($$11 \times d_{embed}$$)일 것이다. 그런데 문장의 길이가 더 길거나 짧다면 그 때마다 input의 shape는 바뀌게 된다. 실제 model 학습 과정에서는 한 문장 씩이 아닌 mini-batch씩 여러 문장와야 하는데 각 문장 마다의 length가 다를 경우 batch를 만들어낼 수 없다. 이러한 문제를 해결하기 위해 $$\text{seq_len}$$(해당 mini-batch 내 token 개수의 최대 값)을 지정하게 되는데, 만약 $$\text{seq_len}$$이 20이라고 한다면 위 문장에서는 9개의 빈 token이 있게 된다. 이러한 빈 token을 pad token이라고 한다. 그런데, 이러한 pad token에는 attention이 부여되어서는 안된다. 실제로는 존재하지도 않는 token과 다른 token 사이의 attention을 찾아서 계산하고, 이를 반영하는 것은 직관적으로도 말이 안된다는 것을 알 수 있다. 따라서 이러한 pad token들에 대해 attention이 부여되지 않도록 처리하는 것이 pad masking이다. masking은 $$(\text{seq_len} \times \text{seq_len})$$ shape의 mask matrix를 곱하는 방식으로 이뤄지는데 mask matrix에서 pad token에 해당하는 row, column의 모든 값은 $$-\inf$$이다. 그 외에는 모두 1이다. 이러한 연산은 scaling과 softmax 사이에 수행하게 되는데, 사실은 scaling 이전, 이후 언제 적용하든 차이는 없다. scaling은 단순히 모든 값을 $$d_k$$로 일괄 나누는 작업이기 때문이다. 대신 반드시 $$Q$$와 $$K$$의 행렬곱 이후, softmax 이전에 적용되어야 한다. mask matrix와 같은 shape는 $$Q$$와 $$K$$의 행렬곱 연산 이후에나 등장하기 때문이다. 또한 softmax는 등장하는 모든 값들을 반영해 확률값을 계산하게 되는데, 이 때 pad token의 값이 반영되어서는 안되므로 softmax 이전에는 반드시 masking이 수행되어야 한다.

#### Self-Attention Code in Pytorch

 Self-Attention을 pytorch code로 구현해보자. Self-Attention은 Transformer에서의 가장 핵심적인 code이므로 반드시 이해하고 넘어가자. 여기서 주의해야 할 점은 실제 model에 들어오는 input은 한 개의 문장이 아니라 mini-batch이기 때문에 $$Q$$, $$K$$, $$V$$의 shape가 $$\text{n_batch} \times \text{seq_len} \times d_k$$라는 것이다.

```python
def calculate_attention(self, query, key, value, mask):
	# query, key, value's shape: (n_batch, seq_len, d_k)
	d_k = key.size(-1) # get d_k
	attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, attention_score's shape: (n_batch, seq_len, seq_len)
	attention_score = attention_score / math.sqrt(d_k) # scaling
	if mask is not None:
		attention_score = score.masked_fill(mask==0, -1e9) # masking
	attention_prob = F.softmax(score, dim=-1) # softmax, attention_prob's shape: (n_batch, seq_len, seq_len)
	out = torch.matmul(attention_prob, value) # Attention_Prob x V, out's shape: (n_batch, seq_len, d_k)
	return out
```

calculate_attention()의 인자로 query, key, value, mask를 받는다. mask는 pad mask matrix일 것이다. pad mask matrix는 Transformer 외부 (대개 Batch class)에서 생성되어 Transformer에 인자로 들어오게 된다. query, key, value는 서로 다른 FC Layer를 거쳐 $$\text{n_batch} \times \text{max_seq_len} \times d_k$$로 변형되었다.

### Multi-Head Attention Layer

![multi_head_attention_in_paper.png](/images/multi_head_attention_in_paper.png){: width="50%"}

출처: Attention is All You Need [[https://arxiv.org/pdf/1706.03762.pdf](https://arxiv.org/pdf/1706.03762.pdf)]

 지금까지의 Self-Attention에 대한 개념은 모두 Multi-Head Attention Layer를 이해하기 위한 것이었다. Attention 계산을 논문에서는 Scaled Dot-Product Attention이라고 명명한다. Transformer는 Scaled Dot Attention을 한 Encoder Layer마다 1회씩 수행하는 것이 아니라 $$h$$회 수행한 뒤, 그 결과를 종합해 사용한다. 이 것이 Multi-Head Attention Layer이다. 이러한 작업을 수행하는 이유는 여러 Attention을 잘 반영하기 위해서이다. 만약 하나의 Attention만 반영한다고 했을 때, 예시 문장에서 'it'의 Attention에는 'animal'의 것이 대부분을 차지하게 될 것이다. 하지만 여러 종류의 attention을 반영한다고 했을 때 'tired'에 집중한 Attention까지 반영된다면, 최종적인 'it'의 Attention에는 'animal'을 지칭한다는 정보, 'tired' 상태라는 정보까지 모두 담기게 될 것이다. 이 것이 Multi-Head Attention을 사용하는 이유이다.

 구체적인 연산 방법을 살펴보자. 논문에서는 $$h=8$$을 채택했다. Scaled Dot-Product Attention에서는 $$Q$$, $$K$$, $$V$$를 위해 FC layer가 총 3개 필요했었는데, 이를 $$h$$회 수행한다고 했으므로 $$3*h$$개의 FC layer가 필요하게 된다. 각각 연산의 최종 output은 $$n \times d_k$$의 shape(실제로는 $$n$$은 $$\text{max_seq_len}$$이지만 notation의 통일성을 위해 $$n$$을 그대로 사용한다)인데, 총 $$h$$개의 $$n \times d_k$$ matrix를 모두 concatenate해서 $$n \times (d_k*h)$$의 shape를 갖는 matrix를 만들어낸다. 이 때 $$d_k*h$$를 $$d_{model}$$로 명명한다. $$d_{model}=d_k*h$$ 수식은 실제 코드 구현에서 매우 중요한 개념이므로 꼭 기억하고 넘어가자.

![multi_head_attention_concat.png](/images/multi_head_attention_concat.png)

 사실 위의 설명은 개념 상의 이해를 돕기 위한 것이고, 실제 연산은 병렬 처리를 위해 더 효율적인 방식으로 수행된다. 기존의 설명에서 $$Q$$, $$K$$, $$V$$를 구하기 위한 FC layer는 $$d_{embed}$$를 $$d_k$$로 변환했다. 이렇게 구해낸 $$Q$$, $$K$$, $$V$$로 각각의 Self-Attention을 계산해 concatenate하는 방식은 별개의 Self-Attention 연산을 총 $$h$$회 수행해야 한다는 점에서 매우 비효율적이다. 따라서 실제로는 $$Q$$, $$K$$, $$V$$ 자체를 $$n \times d_k$$가 아닌, $$n \times d_{model}$$로 생성해내서 한 번의 Self-Attention 계산으로 $$n \times d_{model}$$의 output을 만들어내게 된다. 때문에 $$Q$$, $$K$$, $$V$$를 생성해내기 위한 $$d_{embed} \times d_k$$의 weight matrix를 갖는 FC layer를 $$3*h$$개 운용할 필요 없이 $$d_{embed} \times d_{model}$$의 weight matrix를 갖는 FC layer를 $$3$$개만 운용하면 된다.

 여기서 우리가 알 수 있는 것은 여러 Attention을 반영한다는 Multi-Head Attention Layer의 개념적인 의미는 사실 단지 $$d_k$$의 크기를 $$d_{model}$$로 확장시키는 단순한 구현으로 끝난다는 점이다. $$Q$$, $$K$$, $$V$$ vector에는 담을 수 있는 정보의 양이 $$d_k$$의 dimension으로는 작기 때문에 더 많은 정보(Attention)을 담아내기 위해 $$Q$$, $$K$$, $$V$$ vector의 dimension을 늘린 것으로 이해하면 된다.

 다시 본론으로 되돌아와서 최종적으로 생성해된 matrix를 FC layer에 넣어 multi-head attention의 input과 같은 shape($$n \times d_{embed}$$)의 matrix로 변환하는 과정이 필요하다. 따라서 마지막 FC layer의 input dimension은 $$d_{model}$$, output dimension은 $$d_{embed}$$가 된다. 이는 multi-head attention layer도 하나의 함수라고 생각했을 때, input의 shape와 output의 shape가 동일하게 하기 위함이다.

![multi_head_attention_fc_layer.png](/images/multi_head_attention_fc_layer.png)

![multi_head_attention.png](/images/multi_head_attention.png)

#### Multi-Head Attention Code in Pytorch

 Multi-Head Attention Layer를 실제 code로 구현해보자. 위에서 구현했던 calculate_attention()을 사용한다.

```python
class MultiHeadAttentionLayer(nn.Module):
	def __init__(self, d_model, h, qkv_fc_layer, fc_layer):
		# qkv_fc_layer's shape: (d_embed, d_model)
		# fc_layer's shape: (d_model, d_embed)
		super(MultiHeadAttentionLayer, self).__init__()
		self.d_model = d_model
		self.h = h
		self.query_fc_layer = copy.deepcopy(qkv_fc_layer)
		self.key_fc_layer = copy.deepcopy(qkv_fc_layer)
		self.value_fc_layer = copy.deepcopy(qkv_fc_layer)
		self.fc_layer = fc_layer

		...
```

우선 생성자를 살펴보자. qkv_fc_layer 인자로 $$d_{embed} \times d_{model}$$의 weight matrix를 갖는 FC Layer를 받아 멤버 변수로 $$Q$$, $$K$$, $$V$$에 대해 각각 copy.deepcopy를 호출해 저장한다. deepcopy를 호출하는 이유는 실제로는 서로 다른 weight를 갖고 별개로 운용되게 하기 위함이다. copy 없이 하나의 FC Layer로 $$Q$$, $$K$$, $$V$$를 모두 구하게 되면 $$Q$$, $$K$$, $$V$$가 모두 같은 값일 것이다. fc_layer는 attention 계산 이후 거쳐가는 FC Layer로, $$d_{model} \times d_{embed}$$의 weight matrix를 갖는다.

가장 중요한 forward()이다. Transformer 구현에서 가장 핵심적인 부분이므로 반드시 이해하고 넘어가자.

```python
 class MultiHeadAttentionLayer(nn.Module):

		...

	def forward(self, query, key, value, mask=None):
		# query, key, value's shape: (n_batch, seq_len, d_embed)
		# mask's shape: (n_batch, seq_len, seq_len)
		n_batch = query.shape[0] # get n_batch

		def transform(x, fc_layer): # reshape (n_batch, seq_len, d_embed) to (n_batch, h, seq_len, d_k)
			out = fc_layer(x) # out's shape: (n_batch, seq_len, d_model)
			out = out.view(n_batch, -1, self.h, self.d_model//self.h) # out's shape: (n_batch, seq_len, h, d_k)
			out = out.transpose(1, 2) # out's shape: (n_batch, h, seq_len, d_k)
			return out

		query = transform(query, self.query_fc_layer) # query, key, value's shape: (n_batch, h, seq_len ,d_k)
		key = transform(key, self.key_fc_layer)
		value = transform(value, self.value_fc_layer)

		if mask is not None:
			mask = mask.unsqueeze(1) # mask's shape: (n_batch, 1, seq_len, seq_len)

		out = self.calculate_attention(query, key, value, mask) # out's shape: (n_batch, h, seq_len, d_k)
		out = out.transpose(1, 2) # out's shape: (n_batch, seq_len, h, d_k)
		out = contiguous().view(n_batch, -1, self.d_model) # out's shape: (n_batch, seq_len, d_model)
		out = self.fc_layer(out) # out's shape: (n_batch, seq_len, d_embed)
		return out
```

 인자로 받은 query, key, value는 실제 $$Q$$, $$K$$, $$V$$ matrix가 아니다. $$Q$$, $$K$$, $$V$$ 계산을 위해서는 각각 FC Layer 에 input으로 sentence(실제로는 mini-batch이므로 다수의 sentence)를 넣어줘야 하는데, 이 sentence를 의미하는 것이다. Self-Attention이기에 당연히 $$Q$$, $$K$$, $$V$$는 같은 sentence에서 나오게 되는데 왜 별개의 인자로 받는지 의문일 수 있다. 이는 Decoder의 작동 원리를 알고 나면 이해할 수 있을 것이다. 인자로 받은 query, key, value는 sentence이므로 shape는 ($$\text{n_batch} \times \text{seq_len} \times d_{embed}$$)이다. mask matrix는 기본적으로 한 문장에 대해 ($$\text{seq_len} \times \text{seq_len}$$)의 shape를 갖는데, mini-batch이므로 ($$\text{n_batch} \times \text{seq_len} \times \text{seq_len}$$)의 shape를 갖는다.

 transform()은 $$Q$$, $$K$$, $$V$$를 구해주는 함수이다. 따라서 input shape는 ($$\text{n_batch} \times \text{seq_len} \times d_{embed}$$)이고, output shape는 ($$\text{n_batch} \times \text{seq_len} \times d_{model}$$)이어야 한다. 하지만 실제로는 단순히 FC Layer만 거쳐가는 것이 아닌 추가적인 변형이 일어난다. 우선 $$d_{model}$$을 $$h$$와 $$d_k$$로 분리하고, 각각을 하나의 dimension으로 분리한다(```transform()```). 따라서 shape는 ($$\text{n_batch} \times \text{seq_len} \times h \times d_k$$)가 된다. 이후 이를 transpose해 ($$\text{n_batch} \times h \times \text{seq_len} \times d_k$$)로 변환한다. 이러한 작업을 수행하는 이유는 위에서 작성했던 calculate_attention()이 input으로 받고자 하는 shape가 ($$\text{n_batch} \times ... \times \text{seq_len} \times d_k$$)이기 때문이다. 아래에서 calculate_attention()의 code를 다시 살펴보자.

```python
def calculate_attention(self, query, key, value, mask):
	# query, key, value's shape: (n_batch, seq_len, d_k)
	d_k = key.size(-1) # get d_k
	attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, attention_score's shape: (n_batch, seq_len, seq_len)
	attention_score = attention_score / math.sqrt(d_k) # scaling
	if mask is not None:
		attention_score = score.masked_fill(mask==0, -1e9) # masking
	attention_prob = F.softmax(score, dim=-1) # softmax, attention_prob's shape: (n_batch, seq_len, seq_len)
	out = torch.matmul(attention_prob, value) # Attention_Prob x V, out's shape: (n_batch, seq_len, d_k)
	return out
```

 우선 $$d_k$$를 중심으로 $$Q$$와 $$K$$ 사이 행렬곱 연산을 수행하기 때문에 $$Q$$, $$K$$, $$V$$의 마지막 dimension은 반드시 $$d_k$$여야만 한다. 또한 attention_score의 shape는 마지막 두 dimension이 반드시 ($$\text{seq_len} \times \text{seq_len}$$)이어야만 masking이 적용될 수 있기 때문에 $$Q$$, $$K$$, $$V$$의 마지막 직전 dimension(```.shape[-2]```)은 반드시 $$\text{seq_len}$$이어야만 한다.

 다시 forward()로 되돌아와서, transform()을 사용해 query, key, value를 구하고 나면 mask 역시 변형을 가해야 한다. ($$\text{n_batch} \times \text{seq_len} \times \text{seq_len}$$) 형태를 ($$\text{n_batch} \times 1 \times \text{seq_len} \times \text{seq_len}$$)로 변경하게 된다. 이는 calculate_attention() 내에서 masking을 수행할 때 broadcasting이 제대로 수행되게 하기 위함이다.

 calculate_attention()을 사용해 attention을 계산하고 나면 그 shape는 ($$\text{n_batch} \times h \times \text{seq_len} \times d_k$$)이다. Multi-Head Attention Layer의 최종 output은 input의 것과 같은 ($$\text{n_batch} \times \text{seq_len} \times d_{embed}$$)여야만 하기 때문에 shape를 맞춰줘야 한다. 이를 위해 $$h$$와 $$\text{seq_len}$$의 순서를 뒤바꾸고(```.transpose(1, 2)```) 다시 $$h$$와 $$d_k$$를 $$d_{model}$$로 결합한다. 이후 FC Layer를 거쳐 $$d_{model}$$을 $$d_{embed}$$로 변환하게 된다.

 Encoder Layer로 다시 되돌아가보자. pad mask는 Transformer model 외부에서(mini-batch 생성할 때) 생성되게 될 것이므로 EncoderLayer의 forward()에서 인자로 받는다. 따라서 forward()의 최종 인자는 ```x, mask```가 된다. 한편, 이전에는 Multi-Head Attention Layer의 forward()의 인자가 1개(```x```)일 것으로 가정하고 code를 작성했는데, 실제로는 query, key, value를 받아야 하므로 이를 수정해준다. 이에 더해 mask 역시 인자로 받게 될 것이다. 따라서 multi_head_attention_layer의 forward()의 인자는 최종적으로 ```x, x, x, mask```가 된다.

```python
class EncoderLayer(nn.Module):

	def __init__(self, multi_head_attention_layer, position_wise_feed_forward_layer):
		super(EncoderLayer, self).__init__()
		self.multi_head_attention_layer = multi_head_attention_layer
		self.position_wise_feed_forward_layer = position_wise_feed_forward_layer

	def forward(self, x, mask):
		out = self.multi_head_attention_layer(query=x, key=x, value=x, mask=mask)
		out = self.position_wise_feed_forward_layer(out)
		return out
```

mask 인자를 받기 위해 Encoder 역시 수정이 가해진다. forward()의 인자에 ```mask```를 추가하고, 이를 sublayer의 forward()에 넘겨준다 (```out, mask```).

```python
class Encoder(nn.Module):

	def __init__(self, encoder_layer, n_layer):  # n_layer: Encoder Layer의 개수
		super(Encoder, self).__init__()
		self.layers = []
		for i in range(n_layer):
			self.layers.append(copy.deepcopy(encoder_layer))

	def forward(self, x, mask):
		out = x
		for layer in self.layers:
			out = layer(out, mask)
		return out
```

Transformer 역시 수정해야 한다. forward()의 인자에 ```mask```를 추가하고, encoder의 forward()에 넘겨준다(```src, mask```).

```python
class Transformer(nn.Module):

	def __init__(self, encoder, decoder):
		super(Transformer, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, src, trg, mask):
		encoder_output = self.encoder(src, mask)
		out = self.decoder(trg, encoder_output)
		return out
```

### Position-wise Feed Forward Layer

 단순하게 2개의 FC Layer를 갖는 Layer이다. 각 FC Layer는 ($$d_{embed} \times d_{ff}$$), ($$d_{ff} \times d_{embed}$$)의 weight matrix를 갖는다. 즉, Feed Forward Layer 역시 input의 shape를 그대로 유지한다. 다음 Encoder Layer에 shape를 유지한 채 넘겨줘야 하기 때문이다. 정리하자면, Feed Forward Layer는 Multi-Head Attention Layer의 output을 input으로 받아 연산을 수행하고, 다음 Encoder에게 output을 넘겨준다. 논문에서의 수식을 참고하면 첫번째 FC Layer의 output에 ReLU를 적용하게 된다.

$$\text{FFN}(x)=\text{max}(0, xW_1+b_1)W_2 + b_2$$

code로 구현하면 다음과 같다.

```python
class PositionWiseFeedForwardLayer(nn.Module):
	def __init__(self, first_fc_layer, second_fc_layer):
		self.first_fc_layer = first_fc_layer
		self.second_fc_layer = second_fc_layer
	
	def forward(self, x):
		out = self.first_fc_layer(x)
		out = F.relu(out)
		out = self.dropout(out)
		out = self.second_fc_layer(out)
		return out
```

생성자의 인자로 받는 두 FC Layer는 ($$d_{embed} \times d_{ff}$$), ($$d_{ff} \times d_{embed}$$)의 shape를 가져야만 한다.

### Norm Layer(Residual Connection)

 Encoder Layer의 구조를 다시 가져와 살펴보자.

![encoder_layer.png](/images/encoder_layer.png)

Encoder Layer는 위에서 다뤘던 Multi-Head Attention Layer와 Feed-Forwad Layer로 구성된다. 그러나 사실은 Encoder Layer를 구성하는 두 layer는 Residual Connection으로 둘러싸여 있다. Residual Connection이라는 것은 정말 단순하다. $$y = f(x)$$를 $$y=f(x)+x$$로 변경하는 것이다. 즉, output을 그대로 사용하지 않고, output에 input을 추가적으로 더한 값을 사용하게 된다. 이로 인해 얻을 수 있는 이점은 명확하다. Back Propagation 도중 발생할 수 있는 Gradient Vanishing을 방지할 수 있다. 개념적으로는 이 것이 전부이다. 여기에 더해 논문에서 채택한 Layer Normalization까지 추가한다. 간단하게 코드로 구현해보자.

```python
class ResidualConnectionLayer(nn.Module):
	def __init__(self, norm_layer):
		super(ResidualConnectionLayer, self).__init__()
		self.norm_layer = norm_layer

	def forward(self, x, sub_layer):
		out = sub_layer(x) + x
		out = self.norm_layer(out)
		return out
```

forward()에서 sub_layer까지 인자로 받는다.

따라서 EncoderLayer의 코드가 아래와 같이 변경되게 된다. residual_connection_layers에 ResidualConnectionLayer를 2개 생성해 저장하고, 0번째는 multi_head_attention_layer를 감싸고, 1번째는 position_wise_feed_forward_layer를 감싸게 된다.

```python
class EncoderLayer(nn.Module):

	def __init__(self, multi_head_attention_layer, position_wise_feed_forward_layer, norm_layer):
		super(EncoderLayer, self).__init__()
		self.multi_head_attention_layer = multi_head_attention_layer
		self.position_wise_feed_forward_layer = position_wise_feed_forward_layer
		self.residual_connection_layers = [ResidualConnectionLayer(copy.deepcopy(norm_layer)) for i in range(2)]

	def forward(self, x, mask):
		out = self.residual_connection_layers[0](x, lambda x: self.multi_head_attention_layer(x, x, x, mask))
		out = self.residual_connection_layers[1](x, lambda x: self.position_wise_feed_forward_layer(x))
		return out
```

ResidualConnectionLayer의 forward()에 sub_layer를 전달할 때에는 lambda 식의 형태로 전달한다.

## Decoder

Transformer의 Decoder는 Encoder를 완벽히 이해했다면 큰 무리없이 이해할 수 있다. Encoder의 Layer를 그대로 가져와 사용하고, 몇몇 변경만 가해주는 정도이기 때문이다. 우선 Decoder를 이해하기 위해서는 Decoder의 input과 output이 무엇인지부터 명확히 해 무엇을 하는 module인지 파악하는 것이 최우선이다.

![decoder_simple.png](/images/decoder_simple.png)

$$y=\text{Decoder}(c,z)\\y,\ z\text{ : sentence}\\c\text{ : context}$$

가장 처음에 Transformer의 전체 구조를 이야기할 때 봤던 Decoder의 구조이다. Context와 Some Sentence를 input으로 받아 Output Sentence를 출력한다. Context는 Encoder의 output이라는 것은 이해했다. Transformer model의 목적을 다시 상기시켜 보자. input sentence를 받아와 output sentence를 만들어내는 model이다. 대표적으로 번역과 같은 task를 처리할 수 있을 것이다. 번역이라고 가정한다면, Encoder는 Context를 생성해내는 것, 즉 input sentence의 정보를 압축해 담아내는 것을 목적으로 하고, Decoder는 Context를 활용해 output sentence를 만들어내는 것을 목적으로 한다. 그렇다면 Decoder는 input으로 Context만 받아야 하지, 왜 다른 추가적인 sentence를 받을까? 또 이 sentence는 도대체 무엇일까? 이에 대해 알아보자.

### Decoder's Input

#### Context

 위에서 언급했듯이, Decoder의 input으로는 Context와 sentence가 있다. Context는 Encoder에서 생성된 것이다. Encoder 내부에서 Multi-Head Attention Layer나 Position-wise Feed-Forward Layer 모두 input의 shape를 보존했음을 주목하자. 때문에 Encoder Layer 자체도 input의 shape를 보존할 것이고, Encoder Layer가 쌓인 Encoder 전체도 input의 shape를 보존한다. 따라서 Encoder의 output인 Context는 Encoder의 input인 sentence와 동일한 shape를 갖는다. 이 점만 기억하고 넘어가면, 이후 Decoder에서 Context를 사용할 때 이해가 훨씬 수월하다. 이제 Decoder input 중 Context가 아닌 추가적인 sentence에 대해서 알아보자.

#### Teacher Forcing

 Decoder의 input에 추가적으로 들어오는 sentence를 이해하기 위해서는 Teacher Forcing라는 개념에 대해 알고 있어야 한다. RNN 계열이든, Transformer 계얼이든 번역 model이 있다고 생각해보자. 결국에는 새로운 sentence를 생성해내야만 한다. 힘들게 만들어낸 model이 초창기 학습을 진행하는 상황이다. random하게 초기화된 parameter들의 값 때문에 엉터리 결과가 나올 것이다. RNN으로 생각을 해봤을 때, 첫번째 token을 생성해내고 이를 다음 token을 생성할 때의 input으로 활용하게 된다. 즉, 현재 token을 생성할 때 이전에 생성한 token들을 활용하는 것이다. 그런데 model의 학습 초반 성능은 말그대로 엉터리 결과일 것이기 떄문에, model이 도출해낸 엉터리 token을 이후 학습에 사용하게 되면 점점 결과물은 미궁으로 빠질 것이다. 이러한 현상을 방지하기 위해서 Teacher Forcing을 사용하게 된다. Teacher Forcing이란, Supervised Learning에서 label data를 input으로 활용하는 것이다. RNN으로 번역 model을 만든다고 할 때, 학습 과정에서 model이 생성해낸 token을 다음 token 생성 때 사용하는 것이 아닌, 실제 label data의 token을 사용하게 되는 것이다. 우선 정확도 100%를 달성하는 이상적인 model의 경우를 생각해보자.

![teacher_forcing_ideal.png](/images/teacher_forcing_ideal.png)

우리의 예상대로 RNN 이전 cell의 output을 활용해 다음 cell에서 token을 정상적으로 생성해낼 수 있다. 그런데 이런 100%의 성능을 갖는 model은 실존하지 않는다.

![teacher_forcing_incorrect.png](/images/teacher_forcing_incorrect.png)

현실에서는, 특히나 model 학습 초창기에는 위처럼 잘못된 token을 생성해내고, 그 이후 계속적으로 잘못된 token이 생성될 것이다. 초반에 하나의 token이 잘못 도출되었다고 이후 token이 모두 다 잘못되게 나온다면 제대로 된 학습이 진행되기 힘들 것이다. 따라서 이를 위해 Teacher Forcing을 사용한다.

![teacher_forcing_correct.png](/images/teacher_forcing_correct.png)

Teacher Forcing은 실제 labeled data(Ground Truth)를 RNN cell의 input으로 사용하는 것이다. 정확히는 Ground Truth의 [:-1]로 slicing을 한 것이다(마지막 token인 EOS token을 제외하는 것이다). 이를 통해서 model이 잘못된 token을 생성해내더라도 이후 제대로 된 token을 생성해내도록 유도할 수 있다.

하지만 이는 model 학습 과정에서 Ground Truth를 포함한 dataset을 갖고 있을 때에나 가능한 것이기에 Test나 실제로 Real-World에 Deliever될 때에는 model이 생성해낸 이전 token을 사용하게 된다.

이처럼 학습 과정과 실제 사용에서의 괴리가 발생하기는 하지만, model의 학습 성능을 비약적으로 향상시킬 수 있다는 점에서 대부분의 NLP model에서 필수적으로 사용하는 기법이다.

#### Teacher Forcing in Transformer (Subsequent Masking)

 Teacher Forcing 개념을 이해하고 나면 Transformer Decoder에 input으로 들어오는 sentence가 어떤 것인지 이해할 수 있다. ground truth[:-1]의 sentence일 것이다. 하지만 이러한 방식으로 Teacher Forcing이 Transformer에 그대로 적용될 수 있을까? 결론부터 말하자면 그래서는 안된다. 위에서 Teacher Forcing에서 예시를 든 RNN Model은 이전 cell의 output을 이후 cell에서 사용할 수 있었다. 앞에서부터 순서대로 RNN cell이 실행되기 때문에 이러한 방식이 가능했다. 하지만 Transformer가 RNN에 비해 갖는 가장 큰 장점은 병렬 연산이 가능하다는 것이었다. 병렬 연산을 위해 ground truth의 embedding을 matrix로 만들어 input으로 그대로 사용하게 되면, Decoder에서 Self-Attention 연산을 수행하게 될 때 현재 출력해내야 하는 token의 정답까지 알고 있는 상황이 발생한다. 따라서 masking을 적용해야 한다. $$i$$번째 token을 생성해낼 때, $$1 \thicksim i-1$$의 token은 보이지 않도록 처리를 해야 하는 것이다. 이러한 masking 기법을 subsequent masking이라고 한다. pytorch code로 구현해보자.

```python
def subsequent_mask(size):
	atten_shape = (1, size, size)
	mask = np.triu(np.ones(atatn_shape), k=1).astype('uint8') # masking with upper triangle matrix
	return torch.from_numpy(mask)==0 # reverse (masking=False, non-masking=True)

def make_std_mask(tgt, pad):
	tgt_mask = (tgt != pad) # pad masking
	tgt_mask = tgt_mask.unsqueeze(-2) # reshape (n_batch, seq_len) -> (n_batch, 1, seq_len)
	tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) # pad_masking & subsequent_masking
	return tgt_mask
```

 make_std_mask()는 subsequent_mask()를 호출해 subsequent mask을 생성하고, 이를 pad mask와 결합한다. 위의 code는 Transformer 내부가 아닌 Batch class 내에서 실행되는 것이 바람직할 것이다. mask 생성은 Transformer 내부 작업이 아닌 전처리 과정에 포함되기 때문이다. 따라서 Encoder에 적용되는 pad mask와 동일하게 Batch class 내에서 생성될 것이다. 이는 결국 Transformer 외부에서 넘어와야 하기 때문에 Transformer code가 수정되어야 한다. 기존에는 Encoder에서 사용하는 pad mask(```src_mask```)만이 forward()의 인자로 들어왔다면, 이제는 Decoder에서 사용할 subsequent mask (```trg_mask```)도 함께 주어진다. 따라서 forward()의 최종 인자는 ```src, trg, src_mask, trg_mask```이다. 각각 Encoder의 input, Decoder의 input, Encoder의 mask, Decoder의 mask이다. forward() 내부에서 decoder의 forward()를 호출할 때 역시 변경되는데, ```trg_mask```가 추가적으로 인자로 넘어가게 된다.

```python
class Transformer(nn.Module):

	def __init__(self, encoder, decoder):
		super(Transformer, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, src, trg, src_mask, trg_mask):
		encoder_output = self.encoder(src, src_mask)
		out = self.decoder(trg, trg_mask, encoder_output)
		return out
```

### Decoder Layer

 Decoder 역시 Encoder와 마찬가지로 $$N$$개의 Decoder Layer가 겹겹이 쌓인 구조이다. 이 때 주목해야 하는 점은 Encoder에서 넘어오는 Context가 각 Decoder Layer마다 input으로 들어간다는 것이다. 그 외에는 Encoder와 차이가 전혀 없다.

![decoder.png](/images/decoder.png)

그렇다면 각 Decoder Layer는 어떻게 생겼을까?

![decoder_layer.png](/images/decoder_layer.png)

 Decoder Layer는 Encoder Layer와 달리 Multi-Head Attention Layer가 2개가 존재한다. 첫번째 layer는 Masked Multi-Head Attention Layer라고 부르는데, 이는 위에서 언급했던 subsequent masking이 적용되기 떄문이다. 두번째 layer는 특징이 Encoder에서 넘어온 Context를 input으로 받아 사용한다는 것이다. 즉, Encoder의 Context는 Decoder 내 각 Decoder Layer의 두번째 Multi-Head Attention Layer에서 사용되게 된다. 마지막 Position-wise Feed-Forward Layer는 Encoder Layer의 것과 완전히 동일하므로 설명을 생략한다. 이제 두 Multi-Head Attention Layer에 대해서 Encoder의 것과 비교하며 특징을 살펴보자.

### Masked Multi-Head Attention Layer

 Masked Multi-Head Attention Layer에 대한 설명은 특별한 것이 없다. Encoder의 것과 완전히 동일한데 다만 mask로 들어오는 인자가 일반적인 pad masking에 더해 subsequent masking까지 적용되어 있다는 점만이 차이일 뿐이다. 즉, 이 layer는 Self-Attention을 수행하는 layer이다. '**Self**'에 주목하자. **같은** sentence 내 token들 사이의 attention을 찾는 것이다. 이는 다음 Multi-Head Attention Layer와 가장 큰 차이점이다.

### Multi-Head Attention Layer

 Decoder의 가장 핵심적인 부분이다. Decoder Layer 내 이전 Masked Multi-Head Attention Layer에서 넘어온 output을 input으로 받는다. 여기에 추가적으로 Encoder에서 도출된 Context도 input으로 받는다. 두 input의 사용 용도는 완전히 다르다. **Decoder Layer 내부에서 전달된 input**(teacher forcing으로 넘어온 input)**은 Query로써 사용**하고, Encoder에서 넘어온 **Context는 Key, Value로써 사용**하게 된다. 이 점을 반드시 기억하고 넘어가자. 정리하자면 Decoder Layer의 2번째 layer는 Decoder에서 넘어온 input과 Encoder에서 넘어온 input 사이의 Attention을 계산하는 것이다. 따라서 Self-Attention이 아니다. 우리가 Decoder에서 도출해내고자 하는 최종 output은 teacher forcing으로 넘어온 sentence와 최대한 유사한 sentence이다. 따라서 Decoder Layer 내 이전 layer에서 넘어오는 input이 Query가 되고, 이에 상응하는 Encoder에서의 Attention을 찾기 위해 Context를 Key, Value로 두게 된다. 번역 task를 생각했을 때 가장 직관적으로 와닿는다. 만약 English를 French로 번역하고자 한다면, Encoder의 input은 English sentence일 것이고, Encoder가 도출해낸 Context는 English에 대한 Context일 것이다. Decoder의 input(teacher forcing)과 output은 French sentence일 것이다. 따라서 이 경우에는 Query가 French, Key와 Value는 English가 되어야 한다.

 지금 와서 Multi-Head Attention Layer를 보면 이전과 다르게 보일 것이다. query, key, value를 굳이 각각 별개의 인자로 받은 이유는 Decoder Layer 내에서 query와 key, value는 서로 다른 embedding에서 넘어오기 때문이다.

```python
 class MultiHeadAttentionLayer(nn.Module):

		...

	def forward(self, query, key, value, mask=None):
		
		...
```

### Decoder Code in Pytorch

이제 Decoder와 Decoder Layer에 대한 code를 완성해보자. Position-wise Feed Forward Network와 Residual Connection Layer는 모두 동일하게 사용한다.

```python
class Decoder(nn.Module):
	def __init__(self, sub_layer, n_layer):
		super(Decoder, self).__init__()
		self.layers = []
		for i in range(n_layer):
			self.layers.append(copy.deepcopy(sub_layer))

	def forward(self, x, mask, encoder_output, encoder_mask):
		out = x
		for layer in self.layers:
			out = layer(x, mask, encoder_output, encoder_mask)
		return out
```

 주목해야 할 점은 encoder_mask (encoder에서 사용한 pad masking)는 decoder에서도 사용된다는 것이다. 결국 Decoder의 forward()는 Decoder의 input sentence (```x```), Decoder의 subsequent mask(```mask```), encoder의 Context (```encoder_output```), encoder의 mask (```encoder_mask```)를 모두 인자로 받아야 한다.

```python
class DecoderLayer(nn.Module):
	def __init__(self, masked_multi_head_attention_layer, multi_head_attention_layer, position_wise_feed_forward_layer, norm_layer):
		super(DecoderLayer, self).__init__()
		self.masked_multi_head_attention_layer = ResidualConnectionLayer(masked_multi_head_attention_layer, copy.deepcopy(norm_layer))
		self.multi_head_attention_layer = ResidualConnectionLayer(multi_head_attention_layer, copy.deepcopy(norm_layer))
		self.position_wise_feed_forward_layer = ResidualConnectionLayer(position_wise_feed_forward_layer, copy.deepcopy(norm_layer))

	def forward(self, x, mask, encoder_output, encoder_mask):
		out = self.masked_multi_head_attention_layer(query=x, key=x, value=x, mask=mask)
		out = self.multi_head_attention_layer(query=out, key=encoder_output, value=encoder_output, mask=encoder_mask)
		out = self.position_wise_feed_forward_layer(x=out)
		return out
```

 DecoderLayer에서 masked_multi_head_attention_layer에서는 query, key, value를 모두 decoder의 input sentence (```x```)로 사용한다. 반면 multi_head_attention_layer에서는 query는 masked_multi_head_attention_layer에서의 결과(```out```)를, key, value는 Context(```encoder_output```)으로 사용한다. 이 때 mask는 subsequent_mask가 아닌 일반 pad mask(```encoder_mask```)를 넘겨준다.

Transformer도 다음과 같이 수정된다. forward() 내부에서 Decoder의 forward()를 호출할 때에 encoder의 mask까지 함께 넘겨준다. 따라서 Decoder forward()의 최종 인자는 ```trg, trg_mask, encoder_output, src_mask```이다. 

```python
class Transformer(nn.Module):

	def __init__(self, encoder, decoder):
		super(Transformer, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, src, trg, src_mask, trg_mask):
		encoder_output = self.encoder(src, src_mask)
		out = self.decoder(trg, trg_mask, encoder_output, src_mask)
		return out
```

## Transformer's Input (Positional Encoding)

지금까지 Encoder와 Decoder의 내부 구조가 어떻게 이루어져 있는지 분석하고 code로 구현까지 마쳤다. 이번에는 Encoder와 Decoder의 input으로 들어오는 sentence는 어떤 형태인지 알아보자. Transformer의 input은 단순한 sentence(token embedding sequence)에 더해 Positional Encoding이 추가되게 된다. 전체 TransformerEmbedding은 단순 Embedding과 PositionalEncoding의 sequential이다. code는 단순하다.

```python
class TransformerEmbedding(nn.Module):
	def __init__(self, embedding, positional_encoding):
		super(TransformerEmbedding, self).__init__()
		self.embedding = nn.Sequential(embedding, positional_encoding)

	def forward(self, x):
		out = self.embedding(x)
		return out
```

Embedding 역시 단순하다.

```python
class Embedding(nn.Module):
	def __init__(self, d_embed, vocab):
		super(Embedding, self).__init__()
		self.embedding = nn.Embedding(len(vocab), d_embed)
		self.vocab = vocab
		self.d_embed = d_embed

	def forward(self, x):
		out = self.embedding(x) * math.sqrt(self.d_embed)
		return out
```

vocabulary와 $$d_{embed}$$를 사용해 embedding을 생성해낸다. 주목할 점은 embedding에도 scaling을 적용한다는 점이다. forward()에서 $$\sqrt{d_{embed}}$$를 곱해주게 된다.

마지막으로 PositionalEncoding을 살펴보자.

```python
class PositionalEncoding(nn.Module):
	def __init__(self, d_embed, max_seq_len=5000):
		super(PositionalEncoding, self).__init__()
		encoding = torch.zeros(max_seq_len, d_embed)
		position = torch.arange(0, max_seq_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
		encoding[:, 0::2] = torch.sin(position * div_term)
		encoding[:, 1::2] = torch.cos(position * div_term)
		self.encoding = encoding
	
	def forward(self, x):
		out = x + Variable(self.encoding[:, :x.size(1)], requires_grad=False)
		out = self.dropout(out)
		return out
```

code가 난해한데, 직관적으로 작동 원리만 이해하고 넘어가도 충분하다. PositionalEncoding의 목적은 positional정보(대표적으로 token의 순서, 즉 index number)를 정규화시키기 위한 것이다. 단순하게 index number를 positionalEncoding으로 사용하게 될 경우, 만약 training data에서는 최대 문장의 길이가 30이었는데 test data에서 길이 50인 문장이 나오게 된다면 30~49의 index는 model이 학습한 적이 없는 정보가 된다. 이는 제대로 된 성능을 기대하기 어려우므로, positonal 정보를 일정한 범위 안의 실수로 제약해두는 것이다. 여기서 $$sin$$함수와 $$cos$$함수를 사용하는데, 짝수 index에는 $$sin$$함수를, 홀수 index에는 $$cos$$함수를 사용하게 된다. 이를 사용할 경우 항상 -1에서 1 사이의 값만이 positional 정보로 사용되게 된다.

구현 상에서 주의할 점은 forward() 내에서 생성하는 ```Variable```이 학습이 되지 않도록 ```requires_grad=False``` 옵션을 부여해야 한다는 것이다. PositionalEncoding은 학습되는 parameter가 아니기 때문이다.

이렇게 생성해낸 embedding을 Transformer에 추가해주자. code를 수정한다. ```src_embed```와 ```trg_embed```를 Transformer의 생성자 인자로 추가한다. forward() 내부에서 Encoder와 Decoder의 forward()를 호출할 때 각각 ```src_embed(src)```, ```trg_embed(trg)```와 같이 input을 TransformerEmbedding으로 감싸 변환해준다.

```python
class Transformer(nn.Module):

	def __init__(self, src_embed, trg_embed, encoder, decoder):
		super(Transformer, self).__init__()
		self.src_embed = src_embed
		self.trg_embed = trg_embed
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, src, trg, src_mask, trg_mask):
		encoder_output = self.encoder(self.src_embed(src), src_mask)
		out = self.decoder(self.trg_embed(trg), trg_mask, encoder_output, src_mask)
		return out
```

## After Decoder (Generator)

Decoder의 output이 그대로 Transformer의 최종 output이 되는 것은 아니다. 추가적인 layer를 거쳐간다. 이 layer들을 generator라고 부른다.

우리가 결국 해내고자 하는 목표는 Decoder의 output이 sentence, 즉 token의 sequence가 되는 것이다. 그런데 Decoder의 output은 그저 ($$\text{n_batch} \times \text{seq_len} \times \text{d_model}$$)의 shape를 갖는 matrix일 뿐이다. 이를 vocabulary를 사용해 실제 token으로 변환할 수 있도록 차원을 수정해야 한다. 따라서 FC Layer를 거쳐 마지막 dimension을 $$\text{d_model}$$에서 $$\text{len(vocab)}$$으로 변경한다. 그래야 실제 vocabulary 내 token에 대응시킬 수 있는 값이 되기 때문이다. 이후 softmax 함수를 사용해 각 vocabulary에 대한 확률값으로 변환하게 되는데, 이 때 log_softmax를 사용해 성능을 향상시킨다.

 Generator를 직접 Transformer code에 추가해보자.

```python
class Transformer(nn.Module):

	def __init__(self, src_embed, trg_embed, encoder, decoder, fc_layer):
		super(Transformer, self).__init__()
		self.src_embed = src_embed
		self.trg_embed = trg_embed
		self.encoder = encoder
		self.decoder = decoder
		self.fc_layer = fc_layer

	def forward(self, src, trg, src_mask, trg_mask):
		encoder_output = self.encoder(self.src_embed(src), src_mask)
		out = self.decoder(self.trg_embed(trg), trg_mask, encoder_output, src_mask)
		out = self.fc_layer(out)
		out = F.log_softmax(out, dim=-1)
		return out
```

log_softmax에서는 ```dim=-1```이 되는데, 마지막 dimension인 len(vocab)에 대한 확률값을 구해야 하기 때문이다.

## Make Model

Transformer를 생성하는 예제 함수 make_model()은 다음과 같이 작성할 수 있다. 실제 논문 상에서는 $$d_{embed}$$와 $$d_{model}$$을 구분하지 않고 통합해서 $$d_{model}$$로 사용했지만, 이해를 돕기 위해 지금까지 분리해 사용했다.

```python
def make_model(
    src_vocab, 
    trg_vocab, 
    d_embed = 512, 
    n_layer = 6, 
    d_model = 512, 
    h = 8, 
    d_ff = 2048):

    cp = lambda x: copy.deepcopy(x)

    # multi_head_attention_layer 생성한 뒤 copy해 사용
    multi_head_attention_layer = MultiHeadAttentionLayer(
                                    d_model = d_model,
                                    h = h,
                                    qkv_fc_layer = nn.Linear(d_embed, d_model),
                                    fc_layer = nn.Linear(d_model, d_embed))

    # position_wise_feed_forward_layer 생성한 뒤 copy해 사용    
    position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(
                                        first_fc_layer = nn.Linear(d_embed, d_ff),
                                        second_fc_layer = nn.Linear(d_ff, d_embed))
    
    # norm_layer 생성한 뒤 copy해 사용
    norm_layer = nn.LayerNorm(d_embed, eps=1e-6)

    # 실제 model 생성
    model = Transformer(
                src_embed = TransformerEmbedding(    # SRC embedding 생성
                                embedding = Embedding(
                                                d_embed = d_embed, 
                                                vocab = src_vocab), 
                                positional_encoding = PositionalEncoding(
                                                d_embed = d_embed)), 
	
                trg_embed = TransformerEmbedding(    # TRG embedding 생성
                                embedding = Embedding(
                                                d_embed = d_embed, 
                                                vocab = trg_vocab), 
                                positional_encoding = PositionalEncoding(
                                                d_embed = d_embed)),
                encoder = Encoder(                    # Encoder 생성
                                sub_layer = EncoderLayer(
                                                multi_head_attention_layer = cp(multi_head_attention_layer),
                                                position_wise_feed_forward_layer = cp(position_wise_feed_forward_layer),
                                                norm_layer = cp(norm_layer)),
                                n_layer = n_layer),
                decoder = Decoder(                    # Decoder 생성
                                sub_layer = DecoderLayer(
                                                masked_multi_head_attention_layer = cp(multi_head_attention_layer),
                                                multi_head_attention_layer = cp(multi_head_attention_layer),
                                                position_wise_feed_forward_layer = cp(position_wise_feed_forward_layer),
                                                norm_layer = cp(norm_layer)),
                                n_layer = n_layer),
                fc_layer = nn.Linear(d_model, len(trg_vocab)))    # Generator의 FC Layer 생성
    
    return model
```

# Reference

- #### [Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

- #### [WikiDocs](https://wikidocs.net/31379)
