# Roteiro do vídeo de demonstração

Narração opcional: o vídeo já tem legendas. Se a equipe quiser gravar voz, este é o texto de cada trecho.

| Início | Fim | Trecho | Narração sugerida |
|---|---|---|---|
| 0:00.0 | 0:06.0 | Abertura | Somos a equipe do Luz Segura. Mostramos o componente de visão computacional do nosso MVP. |
| 0:06.0 | 0:22.0 | Problema | Iluminação pública que falha deixa o caminho a pé mais perigoso, e a falha demora a ser descoberta. |
| 0:22.0 | 0:42.0 | Como funciona | O detector encontra as luminárias, a medição de brilho confere, e o diagnóstico vira um JSON para o MVP. |
| 0:42.0 | 0:51.0 | Vídeo: sao_paulo_00035 | Calçada da Avenida Paulista à noite: a luminária acesa fica marcada, e os veículos detectados pelo filtro são desfocados antes da exibição; à noite, alguns ainda escapam do filtro. |
| 0:51.0 | 0:59.0 | Vídeo: sao_paulo_00056 | Câmera parada: a luminária acesa no alto do poste fica marcada em todos os quadros. |
| 0:59.0 | 1:09.0 | Vídeo: sao_paulo_00023 | Vista do alto de uma via, só com CPU: a análise completa roda a 17,8 quadros/s, cerca de 59% do ritmo do vídeo; faróis de carros às vezes ainda são marcados como luminária acesa. |
| 1:09.0 | 1:17.0 | Vídeo: sao_paulo_00038 | Praça com uma única luminária, distante e pequena demais para o detector: sem luminária acesa detectada e com o chão escuro, o sistema marca trecho escuro e pede revisão humana. |
| 1:17.0 | 1:25.0 | Vídeo: sao_paulo_00024 | Via mal iluminada: sem luminária acesa à vista e com o chão escuro, o painel marca trecho escuro; no trajeto inteiro, foram 48,8 de 51,5 s assim. |
| 1:25.0 | 1:33.0 | App: foto | Interface web: o cidadão envia a foto com horário e local e recebe a imagem anotada, com pessoas e veículos detectados desfocados, e o diagnóstico explicado. |
| 1:33.0 | 1:39.0 | App: JSON | O mesmo resultado em JSON, no contrato v1 que a automação e o app do MVP consomem. |
| 1:39.0 | 1:45.0 | API | API REST para o MVP: POST /v1/analisar recebe a foto e devolve o JSON; a documentação interativa fica em /docs. |
| 1:45.0 | 1:51.0 | Foto sf311_1_202000314168 | Foto real de um chamado de luminária queimada: o modelo vê 4 luminárias apagadas; a medição de brilho confirma 3 e contradiz 1 (marca ?), por isso o caso vai para revisão humana. |
| 1:51.0 | 1:57.0 | Foto sf311_2_101003779202 | Foto real de um chamado de luminária queimada: a medição de brilho confirma a luminária apagada, mas a confiança do modelo é de só 36%, por isso o caso vai para revisão humana. |
| 1:57.0 | 2:03.0 | Foto sf311_3_101003656777 | Aqui a medição de brilho contradiz o modelo (marca ?): onde ele vê uma luminária apagada, há um ponto de luz; em vez de abrir o chamado sozinho, o sistema pede revisão humana. |
| 2:03.0 | 2:31.0 | Resultados | Os números vêm de conjuntos que o modelo nunca viu no treino. |
| 2:31.0 | 2:51.0 | Privacidade | O sistema não guarda as fotos, pessoas e veículos detectados são desfocados (à noite, algum pode escapar do filtro), e os casos duvidosos que ele aponta passam por revisão humana. |
| 2:51.0 | 3:05.0 | Créditos | Créditos das fontes de dados e dos vídeos usados, com as licenças de cada um. |

Duração total: 185 s.

## Créditos (descrição do YouTube)

Luz Segura: visão computacional para encontrar luminárias públicas apagadas e trechos escuros (FIAP Global Solution 2026, Applied Computer Vision).
Equipe: Arthur Bessa Pian (RM99215); Davi Almeida Desenzi (RM550849); Fabrício Gutierrez Saavedra (RM97631); João Victor Soares (RM551410); Márcio Gastaldi de Oliveira Soares (RM98811).

Vídeos de terceiros (todos modificados: redimensionados, anotados e com pessoas e veículos detectados desfocados):
- São Paulo, SP, Brasil: Ace of Spades / publicvideos.org, https://archive.org/details/sampa_night. Licença: Domínio público (CC0), https://creativecommons.org/publicdomain/zero/1.0/. Trechos: sao_paulo_00035, sao_paulo_00056, sao_paulo_00023, sao_paulo_00038, sao_paulo_00024.

Fotos e dados:
- Fotos de chamados: San Francisco 311 (DataSF), domínio público (ODC PDDL), https://data.sf.gov/resource/vw6y-z8j6. Modificadas: anotadas, com pessoas e veículos detectados desfocados.
- Treino: Open Images V7 (Google), anotações CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) e fotos CC BY 2.0 de seus autores no Flickr (https://creativecommons.org/licenses/by/2.0/), https://storage.googleapis.com/openimages/web/index.html.
- Teste: Comprehensive Dataset for Urban Streetlight Analysis (arXiv:2407.01117), https://github.com/Team16Project/Street-Light-Dataset (só nas métricas; nenhuma foto aparece no vídeo).
