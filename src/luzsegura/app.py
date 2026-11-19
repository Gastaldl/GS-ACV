"""Interface web de demonstração (Gradio): analisar foto ou vídeo noturno.

Rodar:  luzsegura app   (ou python -m luzsegura.app) e abrir http://127.0.0.1:7860
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr

from .diagnostico import NOME_PT, Contexto
from .pipeline import Analisador, criar_analisador
from .video import processar_video

MAX_QUADROS_VIDEO = 900  # ~30 s a 30 quadros/s, para a interface responder rápido
TAMANHO_MAXIMO_ENVIO = "200mb"  # maior upload aceito pelo launch() (foto ou vídeo)
# O Gradio guarda cada upload e cada resultado no cache dele. A cada 5 min apaga o que tem mais de
# 15 min (pior caso: 20 min) e, ao fechar o app, apaga tudo. Os 15 min ficam bem acima da análise
# mais longa (1 a 2 min de vídeo em CPU): no Windows, apagar um vídeo ainda aberto derruba a limpeza.
LIMPEZA_CACHE = (300, 900)

SOBRE = """
### Como o Luz Segura decide

1. **Detector treinado (YOLO26n):** encontra cada luminária pública e diz se está **acesa** ou **apagada**.
2. **Medição de brilho (regra fixa, sem aprendizado de máquina):** confere cada luminária. Acesa tem um ponto de luz saturado; apagada não é mais clara que o entorno.
   Se o modelo e a medição discordarem, ou se a medição não confirmar a luminária apagada, o resultado pede **revisão humana**.
3. **Brilho do chão:** a faixa de baixo da imagem (calçada e rua) mostra se o trecho está escuro. Fotografe incluindo a calçada
   ou a rua na parte de baixo da foto. Numa foto apontada para cima essa faixa mostra céu ou fachada, por isso os resultados
   tirados do chão (iluminação insuficiente e trecho escuro) sempre vão para **revisão humana**. À noite ou na transição, se
   nenhuma luminária for encontrada na foto, a iluminação adequada também sai só do chão e vai para **revisão humana**.
4. **Período:** com horário (com fuso, ex.: 2026-11-18T21:30:00-03:00) e localização, o sistema calcula a **altura do Sol**.
   Luminária apagada só é falha depois do pôr do sol; no anoitecer e no amanhecer (transição), o que não estiver adequado vai para revisão.
   Faltando o horário ou a localização, estima pelo brilho, com menos precisão (o modo noturno do celular clareia a foto), e a luminária apagada vai para revisão.

| Diagnóstico | Quando | Ação sugerida |
|---|---|---|
| Luminária apagada | alguma luminária apagada à noite | chamado de manutenção |
| Iluminação insuficiente | há luminária acesa, mas o chão está escuro | vistoria (árvore, distância entre postes) |
| Trecho escuro | nenhuma luminária detectada e chão escuro | avaliar falta de iluminação |
| Iluminação adequada | chão iluminado e nenhuma apagada | nenhuma (sem luminária encontrada na foto, revisão humana antes de encerrar o relato) |
| Acesa durante o dia | luminária acesa, confirmada pelo brilho, de dia pela altura do Sol ou pelo período informado (nunca só pelo brilho da foto) | verificar a fotocélula |

**Revisão humana**, em vez de chamado automático ou de encerrar o relato, quando:
- o brilho contradiz o modelo ou não confirma a luminária apagada, ou a confiança do modelo na apagada é baixa (abaixo de 55%);
- à noite, uma luminária vista como acesa não mostra ponto de luz;
- o período foi só estimado pelo brilho da foto ou é de transição;
- o diagnóstico vem só do brilho do chão: iluminação insuficiente, trecho escuro e, à noite ou na transição, iluminação
  adequada sem nenhuma luminária encontrada na foto;
- de dia, a luminária acesa tem confiança baixa, não é mais clara que o entorno (pode ser o céu claro atrás dela)
  ou o Sol está a menos de 8° do horizonte (ou não pode ser conferido, sem horário e local).

Limite conhecido: iluminação adequada com alguma luminária acesa encontrada, sem contradição do brilho, não passa por
revisão. Se a luminária apagada do relato ficou fora da foto ou não foi detectada, o sistema não tem como notar.

### Privacidade (LGPD)

- Nada do que você envia é guardado: a foto ou o vídeo enviado e os resultados ficam só no cache temporário do app, no
  computador que o roda, e são **apagados automaticamente em até 20 minutos** (ou ao fechar o app). O último vídeo
  anotado, já anonimizado, fica numa pasta temporária até a próxima análise de vídeo ou até o app ser fechado.
- Pessoas e veículos (placas) **detectados pelo filtro** são **desfocados** antes da exibição do resultado; à noite ou em
  ângulos incomuns, o filtro pode deixar passar algum.
- O app não guarda o histórico das análises no navegador.
- A API usada pelo MVP não grava nada em disco: a foto é analisada na memória e descartada.
- Só o resultado (contagens, diagnóstico e a localização que você informar) segue para o sistema de chamados.
- Localização e horário são **opcionais**; sem eles, com o período no automático, a luminária apagada sempre vai para revisão humana.
"""


def _hora(texto: str | None) -> datetime | None:
    if not texto or not texto.strip():
        return None
    try:
        quando = datetime.fromisoformat(texto.strip())
    except ValueError:
        raise gr.Error("Data e hora inválidas. Use o formato 2026-11-18T21:30:00-03:00") from None
    # Fora do try: gr.Error é um ValueError e seria engolido pelo except acima. Sem fuso, o horário
    # seria lido como UTC: 19:30 em São Paulo viraria 16:30, com o Sol ainda alto.
    if quando.utcoffset() is None:
        raise gr.Error("Informe o fuso no fim da data e hora, ex.: 2026-11-18T21:30:00-03:00 (horário de Brasília)")
    return quando


def _coordenada(texto: str | None, nome: str, limite: float) -> float | None:
    """Latitude ou longitude opcional, digitada como texto (vazio = não informada).

    Um campo numérico do Gradio vazio chega como 0, e (0, 0) fica no oceano: a posição do Sol
    sairia de lá e o período poderia virar dia ou noite por engano.
    """
    if texto is None or not str(texto).strip():
        return None
    try:
        valor = float(str(texto).strip().replace(",", "."))
    except ValueError:
        raise gr.Error(f"{nome} inválida: use graus decimais, ex.: -23.5505") from None
    if not -limite <= valor <= limite:
        raise gr.Error(f"{nome} fora da faixa de -{limite:.0f} a {limite:.0f} graus")
    return valor


def _cartao(res: dict) -> str:
    nome = NOME_PT[res["diagnostico"]].capitalize()
    sev = {"nenhuma": "sem prioridade", "baixa": "prioridade baixa", "media": "prioridade média", "alta": "prioridade alta"}[res["severidade"]]
    revisao = "\n\n**Revisão humana recomendada.**" if res["requer_revisao_humana"] else ""
    return f"## {nome} · {sev}\n\n{res['explicacao']}{revisao}"


def _cartao_video(resumo: dict) -> str:
    u = resumo["luminarias_unicas"]
    apagadas = "".join(f"\n- luminária #{a['id_rastreio']} aos {a['aparece_em_s']:.1f} s (confiança {a['confianca_max']:.0%})"
                       f"{' · revisão humana' if a.get('requer_revisao_humana') else ''}"
                       for a in resumo["luminarias_apagadas"])
    revisao = "\n\n**Revisão humana recomendada.**" if resumo.get("requer_revisao_humana") else ""
    return (f"## Resumo do trajeto ({resumo['duracao_s']:.0f} s)\n\n"
            f"- Luminárias distintas acesas: **{u['luminaria_acesa']}**\n"
            f"- Luminárias distintas apagadas: **{u['luminaria_apagada']}**{apagadas}\n"
            f"- Tempo em trecho escuro: **{resumo['segundos_em_trecho_escuro']:.0f} s**\n"
            f"- Velocidade de processamento: **{resumo['fps_processamento_medio']} quadros/s** em CPU{revisao}")


def criar_interface(analisador: Analisador | None = None) -> gr.Blocks:
    analisador = analisador or criar_analisador()
    # Pasta só dos vídeos anotados. O Gradio copia cada vídeo devolvido para o cache dele (que a
    # limpeza acima apaga), então o anterior pode sair na próxima análise; o resto sai ao fechar o app.
    trabalho = Path(tempfile.mkdtemp(prefix="luzsegura_"))
    atexit.register(shutil.rmtree, trabalho, ignore_errors=True)
    # Foto e vídeo usam o mesmo modelo, que não é seguro entre threads (o rastreador do vídeo guarda
    # estado), e a limpeza da pasta de trabalho supõe um vídeo por vez: uma análise por vez, explícito
    # porque o padrão do Gradio muda com a variável GRADIO_DEFAULT_CONCURRENCY_LIMIT.
    uma_por_vez = {"concurrency_limit": 1, "concurrency_id": "modelo"}

    def analisar_foto(img_rgb, periodo, lat, lon, quando):
        if img_rgb is None:
            raise gr.Error("Envie uma foto.")
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ctx = Contexto(origem="foto", capturado_em=_hora(quando), lat=_coordenada(lat, "Latitude", 90),
                       lon=_coordenada(lon, "Longitude", 180),
                       periodo_informado=None if periodo == "automático" else periodo)
        res, anotada = analisador.analisar(bgr, ctx)
        return cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB), _cartao(res), res

    def analisar_video(caminho, periodo, progresso=gr.Progress()):
        if not caminho:
            raise gr.Error("Envie um vídeo.")
        # Uma análise por vez (uma_por_vez): os vídeos que estão aqui já foram copiados para o cache.
        for antigo in trabalho.glob("*.mp4"):
            with suppress(OSError):  # no Windows, um arquivo ainda aberto não sai agora; sai ao fechar o app
                antigo.unlink()
        saida = trabalho / f"trajeto_{uuid.uuid4().hex}.mp4"

        def avancar(n, total):
            progresso(min(n, MAX_QUADROS_VIDEO) / (min(total, MAX_QUADROS_VIDEO) or MAX_QUADROS_VIDEO),
                      desc="Analisando quadros")

        resumo = processar_video(caminho, analisador, saida=saida, max_quadros=MAX_QUADROS_VIDEO,
                                 periodo=None if periodo == "automático" else periodo, progresso=avancar)
        return str(saida), _cartao_video(resumo), resumo

    with gr.Blocks(title="Luz Segura", delete_cache=LIMPEZA_CACHE) as demo:
        gr.Markdown("# Luz Segura · visão computacional\nEncontra luminárias públicas **apagadas** e trechos "
                    "**escuros** em fotos e vídeos noturnos e explica cada decisão.")
        with gr.Tab("Foto"):
            with gr.Row():
                with gr.Column():
                    entrada = gr.Image(type="numpy", label="Foto noturna da rua ou da luminária, com a calçada embaixo")
                    periodo = gr.Radio(["automático", "noite", "dia"], value="automático", label="Período")
                    with gr.Row():
                        lat = gr.Textbox(label="Latitude (opcional)", placeholder="-23.5505")
                        lon = gr.Textbox(label="Longitude (opcional)", placeholder="-46.6333")
                    quando = gr.Textbox(label="Data e hora da foto, com fuso (opcional)",
                                        placeholder="2026-11-18T21:30:00-03:00")
                    botao = gr.Button("Analisar foto", variant="primary")
                with gr.Column():
                    saida_img = gr.Image(label="Resultado (pessoas e veículos detectados desfocados)")
                    cartao = gr.Markdown()
                    json_saida = gr.JSON(label="Contrato JSON v1 (o que o MVP recebe)")
            botao.click(analisar_foto, [entrada, periodo, lat, lon, quando], [saida_img, cartao, json_saida],
                        **uma_por_vez)
        with gr.Tab("Vídeo (trajeto)"):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(label="Vídeo noturno (até 200 MB; até 30 s são analisados)")
                    periodo_v = gr.Radio(["automático", "noite", "dia"], value="noite", label="Período")
                    botao_v = gr.Button("Analisar vídeo", variant="primary")
                with gr.Column():
                    video_saida = gr.Video(label="Trajeto anotado")
                    cartao_v = gr.Markdown()
                    json_v = gr.JSON(label="Resumo do trajeto")
            botao_v.click(analisar_video, [video, periodo_v], [video_saida, cartao_v, json_v], **uma_por_vez)
        with gr.Tab("Como funciona e privacidade"):
            gr.Markdown(SOBRE)
    return demo


if __name__ == "__main__":
    # run_history=False: o Gradio não guarda no navegador o histórico das análises (local e horário).
    criar_interface().launch(max_file_size=TAMANHO_MAXIMO_ENVIO, run_history=False)
