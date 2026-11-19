# Contrato da API de visão (v1)

É assim que o componente de visão conversa com o resto do Luz Segura: a automação de
chamados, o pipeline de dados e o painel. O contrato completo, com tipos e valores permitidos,
aparece em `http://localhost:8000/docs` (OpenAPI) quando a API está no ar
(`luzsegura api` ou `uvicorn luzsegura.api:app`).

## Endpoints

| Método | Caminho | Uso |
|---|---|---|
| `GET` | `/v1/saude` | verificação de vida (monitoramento) |
| `POST` | `/v1/analisar` | analisa uma foto e devolve o diagnóstico |

### `POST /v1/analisar` (multipart/form-data)

| Campo | Obrigatório | Descrição |
|---|---|---|
| `arquivo` | sim | foto JPEG, PNG ou WebP, até 10 MB e 80 megapixels; foto com lado maior acima de 1920 px é reduzida a 1920 px antes da análise |
| `lat`, `lon` | não | localização do relato; com `capturado_em`, define o período pela posição do Sol |
| `capturado_em` | não | data e hora da foto em ISO 8601 **com fuso**, ex.: `2026-11-18T21:30:00-03:00` ou `2026-11-19T00:30:00Z`; o fuso é obrigatório sempre que o campo for enviado |
| `periodo` | não | `auto` (padrão), `noite` ou `dia`; fora de `auto`, vale mais que `capturado_em` |
| `incluir_imagem` | não | `true` devolve em base64 a imagem anotada, já com as pessoas e os veículos detectados pelo filtro de privacidade desfocados (veja "Privacidade") |

O formato da foto é conferido pelos primeiros bytes do arquivo. O tipo declarado na parte do
formulário é só indicativo: parte sem tipo (como o `requests` do Python envia) ou com
`application/octet-stream` também é aceita.

**Recomendação para o MVP:** envie sempre `capturado_em` e `lat`/`lon`. Se faltar algum deles (com
`periodo=auto`), o período é só estimado pelo brilho da imagem e todo diagnóstico `luminaria_apagada`
vem com `requer_revisao_humana=true`.

Respostas de erro:

| Código | Causa |
|---|---|
| `415` | tipo declarado na parte não suportado (ex.: `image/gif`, `image/heic`) |
| `413` | arquivo maior que 10 MB (o corpo grande é recusado antes de o formulário ser lido) ou resolução acima de 80 megapixels (proteção contra bomba de descompressão) |
| `422` | bytes que não são JPEG, PNG ou WebP válidos (mesmo com o tipo declarado certo), `capturado_em` sem fuso ou coordenada fora da faixa |

As análises são feitas uma por vez, fora do laço de eventos do servidor: requisições simultâneas
esperam na fila, e `/v1/saude` continua respondendo durante uma análise.

## Resposta (exemplo real)

Saída da API para a foto do chamado 101003779202 do serviço 311 de San Francisco (luminária
queimada; DataSF, domínio público), enviada com o horário e o local do chamado. Os valores são os
devolvidos pela API; só a quebra de linha do JSON foi ajustada para leitura.

```bash
curl -F "arquivo=@101003779202.jpg" -F "lat=37.77814" -F "lon=-122.44683" \
     -F "capturado_em=2026-03-30T21:14:09-07:00" http://127.0.0.1:8000/v1/analisar
```

```json
{
  "versao_contrato": "1.0",
  "modelo": "luzsegura-yolo26n",
  "origem": "foto",
  "periodo": "noite",
  "periodo_fonte": "posicao_do_sol",
  "altura_do_sol_graus": -20.5,
  "diagnostico": "luminaria_apagada",
  "severidade": "media",
  "requer_revisao_humana": true,
  "acao_sugerida": "Enviar para revisão humana; se confirmado, abrir chamado de manutenção para a luminária apagada.",
  "explicacao": "Encontrei 1 luminária apagada (confiança até 36%). Período: noite (Sol a -20° do horizonte no horário da captura). Brilho do chão: 134 de 255, acima do mínimo de 40 usado pelo sistema. A medição de brilho confirma 1 luminária apagada (nenhum ponto de luz na área). Atenção: confiança do modelo abaixo de 55%. Resultado: luminária apagada (prioridade média). Enviar para revisão humana; se confirmado, abrir chamado de manutenção para a luminária apagada.",
  "contagem": {
    "luminaria_acesa": 0,
    "luminaria_apagada": 1
  },
  "luminosidade": {
    "cena_media": 107.4,
    "piso_media": 133.9,
    "fracao_escura": 0.03,
    "fracao_saturada": 0.0018
  },
  "luminarias": [
    {
      "classe": "luminaria_apagada",
      "confianca": 0.356,
      "caixa_norm": [0.4432, 0.2974, 0.4804, 0.5437],
      "estado_por_brilho": "apagada",
      "concorda_com_modelo": true,
      "maior_ponto_de_luz_px": 0,
      "contraste_com_entorno": 0.87,
      "id_rastreio": null
    }
  ],
  "privacidade": {
    "pessoas_desfocadas": 0,
    "veiculos_desfocados": 0,
    "imagem_armazenada": false,
    "filtro_ativo": true
  },
  "imagem": {
    "largura": 864,
    "altura": 1152
  },
  "tempo_processamento_ms": 78.9,
  "localizacao": {
    "lat": 37.77814,
    "lon": -122.44683
  },
  "capturado_em": "2026-03-30T21:14:09-07:00",
  "imagem_anotada_jpeg_base64": null
}
```

`luminarias` lista todas as detecções com confiança de pelo menos 0,30 (acesas e apagadas), da
maior para a menor confiança, e não só as com defeito; `contagem` resume essa mesma lista. As
medidas de `luminosidade` vão de 0 a 255 (luma); `imagem` é o tamanho analisado, já reduzido.

### Campos de cada item de `luminarias`

| Campo | Significado |
|---|---|
| `classe`, `confianca` | classe prevista pelo detector e a confiança dele (0 a 1) |
| `caixa_norm` | caixa `[x1, y1, x2, y2]` dividida pela largura e pela altura da imagem (0 a 1) |
| `estado_por_brilho` | `acesa`, `apagada` ou `indefinido` pela regra de brilho, medida nos pixels da caixa, independente do modelo |
| `concorda_com_modelo` | se o brilho concorda com a classe do modelo; `null` quando o brilho é `indefinido` |
| `maior_ponto_de_luz_px` | área, em pixels, da maior mancha saturada na caixa, medida com o lado maior da imagem reduzido a 1024 px, a resolução em que a regra foi calibrada (imagem menor não é ampliada) |
| `contraste_com_entorno` | brilho médio da caixa dividido pelo do anel em volta dela (acima de 1: mais clara que o entorno) |
| `id_rastreio` | ID da luminária num vídeo; `null` em fotos |

## Campos que o MVP usa

| Campo | Quem consome | Para quê |
|---|---|---|
| `diagnostico`, `severidade` | automação (RPA) | abrir e priorizar o chamado |
| `requer_revisao_humana` | automação e painel | mandar para a fila de revisão em vez de abrir o chamado automático |
| `explicacao`, `acao_sugerida` | app do cidadão e assistente (GenAI) | dizer ao cidadão, em linguagem simples, o que foi visto e o que acontece agora |
| `contagem`, `luminosidade`, `localizacao` | pipeline de dados (camada Ouro) | indicadores por região: luminárias apagadas e trechos escuros |
| `privacidade` | auditoria (LGPD) | registrar se o filtro de privacidade estava ativo e quantas pessoas e veículos foram desfocados; `imagem_armazenada` é sempre `false`, porque a API não grava a imagem. É um registro, não uma garantia: o filtro só desfoca o que detecta |

## Valores de `diagnostico`

| Valor | Quando |
|---|---|
| `luminaria_apagada` | à noite ou na transição, pelo menos uma luminária detectada como apagada |
| `iluminacao_insuficiente` | à noite ou na transição, há luminária acesa, mas o chão está escuro |
| `trecho_escuro` | à noite ou na transição, nenhuma luminária detectada e chão escuro |
| `iluminacao_adequada` | à noite ou na transição, chão iluminado e nenhuma luminária apagada detectada (com ou sem luminária acesa; sem nenhuma luminária na foto, vai para revisão humana) |
| `fora_do_periodo_noturno` | foto diurna sem luminária acesa confirmada pelo brilho, ou com o período só estimado pelo brilho da imagem: luminária apagada de dia é o esperado |
| `acesa_durante_o_dia` | **experimental:** luminária acesa, confirmada pela medição de brilho, com período `dia` dado pela posição do Sol ou informado pelo cliente (`periodo=dia`); nunca quando o período é só estimado pelo brilho da imagem. Indica fotocélula com defeito |

O período vem de `periodo`, quando informado; senão, da altura do Sol calculada com `capturado_em`
e `lat`/`lon` (dia acima de +2°, noite abaixo de -4°, transição entre os dois); sem esses campos,
é estimado pelo brilho médio da imagem.

`acesa_durante_o_dia` é experimental: na avaliação, reconheceu só 3 dos 60 chamados reais de luz
acesa o tempo todo do SF 311. No treino, toda luminária fotografada de dia foi rotulada como apagada,
então o modelo tende a ver como apagada até a luminária acesa de dia, e a resposta sai
`fora_do_periodo_noturno`, cuja ação sugerida pede uma foto depois do pôr do sol. Por isso, um
relato de luminária acesa de dia não deve ser encerrado só com essa resposta.

## Revisão humana

`requer_revisao_humana=true` manda o caso para a fila de revisão em vez do chamado automático. Os
motivos aparecem na `explicacao`, cada um começando com "Atenção:".

| Diagnóstico | Vai para revisão quando |
|---|---|
| `luminaria_apagada` | o brilho contradiz o modelo ou não confirma a apagada (`indefinido`) em alguma luminária apagada; todas as apagadas têm confiança abaixo de 0,55; o período foi só estimado pelo brilho da imagem; ou o período é de transição |
| `iluminacao_insuficiente`, `trecho_escuro` | sempre: o "chão" é a faixa de baixo da foto (40% da altura), e numa foto apontada para cima essa faixa mostra céu, fachada ou o próprio poste |
| `iluminacao_adequada` | à noite ou na transição, nenhuma luminária foi encontrada na foto (`contagem` zerada: o resultado vem só do brilho do chão), ou o brilho não encontrou ponto de luz numa luminária classificada como acesa (o modelo pode ter visto como acesa uma luminária apagada) |
| `acesa_durante_o_dia` | nenhuma acesa confirmada tem confiança de pelo menos 0,55; a luz não é mais clara que o entorno (contraste até 1,0: pode ser o céu claro atrás da luminária); ou o Sol está abaixo de 8° ou é desconhecido (período só informado) |
| `fora_do_periodo_noturno` | nunca: não gera chamado |

Chamado automático (`requer_revisao_humana=false`) à noite só sai para `luminaria_apagada`
confirmada pela medição de brilho, com período conhecido (`capturado_em` com `lat`/`lon`, ou
`periodo=noite`) e confiança de pelo menos 0,55. De dia, só para `acesa_durante_o_dia`
(experimental) com confiança de pelo menos 0,55, mais clara que o entorno e com o Sol acima de 8°.

`iluminacao_adequada` não deve encerrar sozinha o relato de um cidadão. Sem revisão, ela só sai
quando o sistema encontrou pelo menos uma luminária acesa, o brilho não contradiz nenhuma e o chão
está claro; mesmo assim, a luminária queimada do relato pode estar fora do quadro ou não ter sido
detectada. Esse é o principal ponto cego do sistema: use o resultado para priorizar a fila, e não
para descartar o relato.

Quando `requer_revisao_humana` é `true`, `acao_sugerida` (e a última frase de `explicacao`) começa
com "Enviar para revisão humana; se confirmado, ..." e descreve o passo seguinte à confirmação do
técnico; em `iluminacao_adequada`, a ação é só "Enviar para revisão humana antes de encerrar o relato".

Para o brilho do chão valer como evidência, oriente o cidadão a fotografar incluindo a calçada ou a
rua na parte de baixo da foto. O horário enviado deve ser o da foto (metadados EXIF), e não o do
envio: uma foto diurna enviada à noite seria tratada como noturna.

## Resumo de vídeo (`luzsegura video` e aba "Vídeo" do app)

O vídeo não passa pela API: a linha de comando e o app analisam o arquivo quadro a quadro, com
rastreamento (ByteTrack) para contar cada luminária uma vez só, e devolvem um resumo do trajeto:

| Campo | Significado |
|---|---|
| `inicio_s` | segundo do arquivo em que a análise começou (ex.: depois de uma cartela de abertura); os tempos do resumo contam a partir dele |
| `quadros_processados`, `duracao_s`, `fps_video` | tamanho do trecho analisado |
| `fps_processamento_medio` | quadros por segundo da análise (detector com rastreamento, brilho, privacidade e diagnóstico); não contam a leitura e a redução de cada quadro, o desenho nem a gravação do vídeo anotado |
| `luminarias_unicas` | luminárias distintas por classe final; conta só trilhas vistas em pelo menos 3 quadros |
| `luminarias_apagadas` | uma entrada por luminária apagada: `id_rastreio`, `aparece_em_s`, `confianca_max` (maior confiança na classe apagada), `quadros`, `quadros_apagada`, `quadros_por_brilho` (`acesa`/`apagada`), `requer_revisao_humana` e `motivos_revisao` |
| `requer_revisao_humana` | se alguma luminária apagada do trajeto pede revisão |
| `fracao_quadros_por_diagnostico` | fração dos quadros em cada valor de `diagnostico` |
| `segundos_em_trecho_escuro` | tempo com diagnóstico `trecho_escuro`; é indicador e não entra em `requer_revisao_humana` |
| `video_anotado` | caminho do vídeo anotado e anonimizado, quando pedido |

A classe final de cada luminária vem de um voto ponderado pela confiança em todos os quadros da
trilha; empate fica como acesa. Motivos de revisão (`motivos_revisao`): `brilho_contradiz_modelo`,
`brilho_nao_confirma`, `confianca_baixa`, `quadros_divididos`, `periodo_transicao`, `periodo_dia` e
`periodo_estimado_pelo_brilho`.

## Privacidade

A API não grava a foto em disco: o corpo da requisição é limitado a cerca de 10 MB antes de ser
lido, o upload fica inteiro na memória e é descartado ao fim da análise, e o conteúdo da imagem
não aparece nos logs.

O filtro de privacidade (um YOLO26n pré-treinado no COCO, sem treino adicional) procura pessoas e
veículos, e as regiões encontradas são pixeladas e desfocadas antes de a imagem anotada ser
desenhada; quando `incluir_imagem=true`, é essa a imagem devolvida. O filtro só desfoca o que
detecta: à noite ou em ângulos incomuns, alguma pessoa ou algum veículo pode escapar. Por isso o
campo `privacidade` é um registro (se o filtro estava ativo e quantas pessoas e veículos foram
desfocados), e não uma garantia. No vídeo, o filtro procura pessoas e veículos a cada 2 quadros e
desfoca, em todos os quadros, as regiões da última busca.
