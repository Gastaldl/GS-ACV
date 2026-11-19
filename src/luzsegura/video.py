"""Análise de vídeo quadro a quadro: rastreia luminárias, conta as únicas e resume o trajeto.

Uso típico: um celular no painel do carro, no ônibus ou na mão de quem caminha grava
a rua à noite; cada luminária ganha um ID de rastreio (ByteTrack), o estado final é a
votação dos quadros em que ela aparece, ponderada pela confiança do modelo, e o resumo
diz quantas luminárias distintas estavam acesas ou apagadas, quais apagadas precisam de
revisão humana e por quanto tempo o trajeto ficou escuro.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from .config import CLASSE_ACESA, CLASSE_APAGADA
from .desenho import desenhar
from .diagnostico import TRECHO_ESCURO, Contexto, diagnosticar
from .luminancia import ACESA, APAGADA, luma, medir_cena
from .pipeline import Analisador
from .privacidade import sem_filtro


class EscritorVideo:
    """Grava MP4 H.264 (toca no navegador) via imageio-ffmpeg; se faltar, usa o codec mp4v do OpenCV."""

    def __init__(self, caminho: Path, largura: int, altura: int, fps: float):
        self._gen = None
        self._cv = None
        caminho.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio_ffmpeg

            self._gen = imageio_ffmpeg.write_frames(
                str(caminho), (largura, altura), fps=fps, codec="libx264", pix_fmt_in="bgr24",
                pix_fmt_out="yuv420p", macro_block_size=2, output_params=["-crf", "23", "-preset", "veryfast"],
            )
            self._gen.send(None)
        except Exception:
            self._gen = None
            self._cv = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))

    def escrever(self, quadro_bgr: np.ndarray) -> None:
        if self._gen is not None:
            self._gen.send(np.ascontiguousarray(quadro_bgr))
        else:
            self._cv.write(quadro_bgr)

    def fechar(self) -> None:
        if self._gen is not None:
            self._gen.close()
        if self._cv is not None:
            self._cv.release()


def processar_video(origem: str | int, analisador: Analisador, saida: Path | None = None, mostrar: bool = False,
                    max_quadros: int | None = None, periodo: str | None = "noite", largura_max: int = 1280,
                    privacidade_a_cada: int = 2, quadros_minimos_trilha: int = 3,
                    progresso: Callable[[int, int], None] | None = None, inicio_s: float = 0.0) -> dict:
    """Processa um arquivo de vídeo (ou o índice de uma webcam) e devolve o resumo do trajeto.

    `periodo`: "noite" quando se sabe que o vídeo é noturno; None para estimar pelo brilho.
    `inicio_s`: segundo do arquivo em que a análise começa (ex.: pular uma cartela de abertura); os
    tempos do resumo e o vídeo anotado contam a partir dele.
    Cada luminária apagada do resumo traz `requer_revisao_humana` e `motivos_revisao`: só dispensa
    revisão a que o brilho confirma, com confiança alta, voto folgado e período "noite" informado.
    Motivos possíveis: brilho_contradiz_modelo, brilho_nao_confirma, confianca_baixa, quadros_divididos,
    periodo_transicao, periodo_dia e periodo_estimado_pelo_brilho. O `requer_revisao_humana` do resumo
    indica se alguma luminária apagada pede revisão; o tempo em trecho escuro é só indicador e não entra nele.
    """
    cap = cv2.VideoCapture(origem)
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {origem}")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if inicio_s > 0:
        pulados = round(inicio_s * fps_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pulados)
        total = max(0, total - pulados)
    detector, lim = analisador.detector, analisador.lim
    detector.reiniciar_rastreio()

    votos: dict[int, Counter] = defaultdict(Counter)  # quadros por classe
    peso: dict[int, Counter] = defaultdict(Counter)  # soma das confianças por classe
    conf_max: dict[int, Counter] = defaultdict(Counter)  # maior confiança por classe
    brilho: dict[int, Counter] = defaultdict(Counter)  # estado pela regra de brilho, quadro a quadro
    primeira_vez: dict[int, float] = {}
    diagnosticos = Counter()
    tempos: list[float] = []
    escritor = None
    deteccoes_privacidade = None
    n = 0
    try:
        while True:
            ok, quadro = cap.read()
            if not ok:
                break
            if largura_max and quadro.shape[1] > largura_max:
                f = largura_max / quadro.shape[1]
                quadro = cv2.resize(quadro, (largura_max, round(quadro.shape[0] * f)), interpolation=cv2.INTER_AREA)
            inicio = perf_counter()
            luminarias = detector.detectar(quadro, rastrear=True)
            cena = medir_cena(luma(quadro), lim)
            if analisador.privacidade is not None:
                if deteccoes_privacidade is None or n % privacidade_a_cada == 0:
                    deteccoes_privacidade = analisador.privacidade.detectar(quadro)
                seguro, info = analisador.privacidade.aplicar(quadro, deteccoes_privacidade)
            else:
                seguro, info = quadro, sem_filtro()
            altura, largura = quadro.shape[:2]
            resultado = diagnosticar(luminarias, cena, largura, altura,
                                     Contexto(origem="video", periodo_informado=periodo), lim, None, info)
            segundo = n / fps_video
            for l in luminarias:
                if l.id_rastreio is None or l.confianca < lim.conf_minima:
                    continue
                tid = l.id_rastreio
                votos[tid][l.classe] += 1
                peso[tid][l.classe] += l.confianca
                conf_max[tid][l.classe] = max(conf_max[tid][l.classe], l.confianca)
                brilho[tid][l.medida.estado] += 1
                primeira_vez.setdefault(tid, segundo)
            diagnosticos[resultado["diagnostico"]] += 1
            tempos.append(perf_counter() - inicio)
            fps_proc = len(tempos[-30:]) / sum(tempos[-30:])
            anotado = desenhar(seguro, luminarias, resultado, fps=fps_proc, conf_minima=lim.conf_minima)
            if saida is not None:
                if escritor is None:
                    escritor = EscritorVideo(saida, largura, altura, fps_video)
                escritor.escrever(anotado)
            if mostrar:
                cv2.imshow("Luz Segura (q para sair)", anotado)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            n += 1
            if progresso:
                progresso(n, total)
            if max_quadros and n >= max_quadros:
                break
    finally:
        cap.release()
        if escritor is not None:
            escritor.fechar()
        if mostrar:
            cv2.destroyAllWindows()

    trilhas = {tid: v for tid, v in votos.items() if sum(v.values()) >= quadros_minimos_trilha}
    # Voto ponderado pela confiança: nos primeiros quadros a luminária está longe e pequena e o modelo
    # erra mais. Empate exato fica como acesa: chamado não sai de cara ou coroa.
    classe_final = {tid: CLASSE_APAGADA if peso[tid][CLASSE_APAGADA] > peso[tid][CLASSE_ACESA] else CLASSE_ACESA
                    for tid in trilhas}
    apagadas = sorted((tid for tid, c in classe_final.items() if c == CLASSE_APAGADA), key=lambda t: primeira_vez[t])

    def motivos_revisao(tid: int) -> list[str]:
        """A mesma política das fotos, aplicada à trilha inteira da luminária."""
        m = []
        b = brilho[tid]
        if b[ACESA] > b[APAGADA]:
            m.append("brilho_contradiz_modelo")
        elif 2 * b[APAGADA] <= sum(b.values()):
            m.append("brilho_nao_confirma")  # "apagada" não é a maioria dos quadros: reflexo ou brilho difuso
        if conf_max[tid][CLASSE_APAGADA] < lim.conf_confiavel:
            m.append("confianca_baixa")
        if 2 * peso[tid][CLASSE_ACESA] > peso[tid][CLASSE_APAGADA]:
            m.append("quadros_divididos")
        # Apagada só é falha à noite: na transição ou de dia pode ser o esperado, e sem período
        # informado a noite é só estimada pelo brilho de cada quadro.
        if periodo in ("transicao", "dia"):
            m.append(f"periodo_{periodo}")
        elif periodo != "noite":
            m.append("periodo_estimado_pelo_brilho")
        return m

    itens = []
    for tid in apagadas:
        motivos = motivos_revisao(tid)
        itens.append({
            "id_rastreio": tid,
            "aparece_em_s": round(primeira_vez[tid], 1),
            "confianca_max": round(conf_max[tid][CLASSE_APAGADA], 3),
            "quadros": sum(trilhas[tid].values()),
            "quadros_apagada": trilhas[tid][CLASSE_APAGADA],
            "quadros_por_brilho": {ACESA: brilho[tid][ACESA], APAGADA: brilho[tid][APAGADA]},
            "requer_revisao_humana": bool(motivos),
            "motivos_revisao": motivos,
        })
    return {
        "origem": str(origem),
        "inicio_s": inicio_s,
        "quadros_processados": n,
        "duracao_s": round(n / fps_video, 1),
        "fps_video": round(fps_video, 2),
        "fps_processamento_medio": round(n / sum(tempos), 1) if tempos else None,
        "luminarias_unicas": {
            CLASSE_ACESA: sum(1 for c in classe_final.values() if c == CLASSE_ACESA),
            CLASSE_APAGADA: len(apagadas),
        },
        "luminarias_apagadas": itens,
        "requer_revisao_humana": any(i["requer_revisao_humana"] for i in itens),
        "fracao_quadros_por_diagnostico": {k: round(v / max(n, 1), 3) for k, v in diagnosticos.most_common()},
        "segundos_em_trecho_escuro": round(diagnosticos[TRECHO_ESCURO] / fps_video, 1),
        "video_anotado": str(saida) if saida else None,
    }
