"""Gera o relatório técnico (HTML e PDF) da entrega de Applied Computer Vision.

Contagens e métricas vêm dos arquivos em results/ e data/manifests/ (e de data/interim/, quando existir);
os limiares de decisão vêm de luzsegura.config. Nenhum número de dados ou de resultado é digitado à mão.
Os parágrafos de análise e os links ficam em docs/relatorio/textos_resultados.json (chaves
RESULTADOS_DETECCAO, RESULTADOS_CHENNAI, RESULTADOS_SF, RESULTADOS_VIDEOS, CONCLUSAO, repo e video);
--repo e --video têm prioridade sobre o JSON.
O PDF é impresso pelo Microsoft Edge, Chrome ou Chromium em modo headless.

Sem --rascunho, o script falha se faltar texto ou resultado: o PDF entregue nunca sai incompleto. Os links
do repositório e do vídeo são opcionais (o enunciado não os pede no PDF): sem eles, o parágrafo de links
fica de fora; depois de publicar, gere de novo com --repo e --video (ou as chaves do textos_resultados.json).
Com --rascunho, gera uma versão de trabalho com cada lacuna marcada como pendente.

Uso:
  python scripts/docs/gerar_relatorio.py
  python scripts/docs/gerar_relatorio.py --repo https://github.com/<org>/<repo> --video https://youtu.be/<id>
  python scripts/docs/gerar_relatorio.py --rascunho
Saídas: docs/relatorio/relatorio.html e docs/Relatorio_LuzSegura_ACV.pdf
        (rascunho: data/outputs/relatorio/rascunho.html e Relatorio_LuzSegura_ACV_rascunho.pdf, fora do git)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from datetime import datetime
from pathlib import Path

from luzsegura.config import CLASSE_ACESA, CLASSE_APAGADA, LIMIARES_PADRAO

RAIZ = Path(__file__).resolve().parents[2]
EQUIPE = [("Arthur Bessa Pian", "RM99215"), ("Davi Almeida Desenzi", "RM550849"),
          ("Fabrício Gutierrez Saavedra", "RM97631"), ("João Victor Soares", "RM551410"),
          ("Márcio Gastaldi de Oliveira Soares", "RM98811")]
NAVEGADORES = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
               Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
               Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]
# Nomes no PATH: no Linux, Chrome, Chromium e Edge têm executáveis com outros nomes.
NAVEGADORES_PATH = ("msedge", "chrome", "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
                    "microsoft-edge", "microsoft-edge-stable")
# Faixas de brilho médio da cena de scripts/dados/montar_dataset.py: noite < 55 <= anoitecer < 105 <= dia.
FAIXA_NOITE, FAIXA_DIA = 55, 105
# Limiar de confiança padrão da validação do Ultralytics 8.4.215, que scripts/avaliacao/avaliar_deteccao.py
# mantém (o mAP precisa da curva inteira); a matriz de confusão da Figura 2 sai dessa mesma validação.
CONF_VALIDACAO = 0.001
CHAVES_TEXTO = ("RESULTADOS_DETECCAO", "RESULTADOS_CHENNAI", "RESULTADOS_SF", "RESULTADOS_VIDEOS", "CONCLUSAO")
DIAGNOSTICOS_PT = {"luminaria_apagada": "luminária apagada", "iluminacao_insuficiente": "iluminação insuficiente",
                   "trecho_escuro": "trecho escuro", "iluminacao_adequada": "iluminação adequada",
                   "fora_do_periodo_noturno": "fora do período noturno", "acesa_durante_o_dia": "acesa durante o dia"}
# Severidade do contrato JSON escrita como no painel das imagens ("nenhuma" não aparece).
PRIORIDADE_PT = {"baixa": "prioridade baixa", "media": "prioridade média", "alta": "prioridade alta"}
# Métricas da escolha entre v1 e v2 (results/modelo_final.json): chave, rótulo e formato.
METRICAS_ESCOLHA = [
    ("chennai_f1_sistema", "Chennai: F1 do sistema", "num"),
    ("chennai_precisao_sistema", "Chennai: precisão do sistema", "pct"),
    ("chennai_revocacao_sistema", "Chennai: revocação do sistema", "pct"),
    ("sf311_noite_acusou_falha_sistema", "SF 311 noturno: falha confirmada pelo sistema", "pct"),
    ("sf311_noite_chamado_automatico", "SF 311 noturno: chamado automático (sem revisão)", "pct"),
    ("sf311_dia_acusou_acesa_de_dia", "SF 311 diurno: luminária acesa durante o dia", "pct"),
    ("oi_teste_map50", "Open Images, teste completo: mAP50", "num"),
    ("oi_teste_map50_95", "Open Images, teste completo: mAP50-95", "num"),
    ("oi_noite_map50", "Open Images, teste noturno: mAP50", "num"),
]


def ler_json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else {}


def ler_csv(caminho: Path) -> list[dict]:
    if not caminho.exists():
        return []
    with caminho.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# Valor ausente vira "–": o rascunho sai mesmo com resultados incompletos.
def pct(x: float | None) -> str:
    # Arredondamento decimal, metade para cima: 0,2145 vira 21,5% (em ponto flutuante, 21,45 vira 21,4).
    if x is None:
        return "–"
    return f"{(Decimal(str(x)) * 100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%".replace(".", ",")


def num(x: float | None, casas: int = 3) -> str:
    # Mesmo arredondamento de pct (decimal, metade para cima), para o texto e as tabelas mostrarem o mesmo valor:
    # 0,4805 vira 0,481 (em ponto flutuante, viraria 0,480).
    if x is None:
        return "–"
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-casas), rounding=ROUND_HALF_UP)).replace(".", ",")


def inteiro(n: int | None) -> str:
    return "–" if n is None else f"{n:,}".replace(",", ".")


def palavras(texto: str) -> int:
    return len(re.findall(r"\w+(?:[-']\w+)*", re.sub(r"<[^>]+>", " ", texto)))


def tabela(cabecalho: list[str], linhas: list[list], classe: str = "") -> str:
    th = "".join(f"<th>{c}</th>" for c in cabecalho)
    corpo = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in l) + "</tr>" for l in linhas)
    return f'<table class="{classe}"><thead><tr>{th}</tr></thead><tbody>{corpo}</tbody></table>'


def figura(caminho: str, legenda: str, largura: str = "100%") -> str:
    return (f'<figure><img src="{caminho}" style="width:{largura}"><figcaption>{legenda}</figcaption></figure>')


def rel(caminho: Path, base: Path) -> str:
    """Caminho do arquivo visto da pasta do HTML (o navegador resolve as figuras a partir dele)."""
    try:
        return Path(os.path.relpath(caminho, base)).as_posix()
    except ValueError:  # outra unidade de disco no Windows
        return caminho.resolve().as_uri()


def nome_licenca(licenca: str) -> str:
    """'https://creativecommons.org/licenses/by/2.0/' vira 'CC BY 2.0 (<link>)'; outros textos ficam iguais."""
    m = re.search(r"creativecommons\.org/licenses/([a-z-]+)/(\d\.\d)", licenca)
    return f"CC {m[1].upper()} {m[2]} ({licenca})" if m else licenca


def contar_ambiguas() -> dict:
    """Imagens noturnas e de anoitecer descartadas por terem alguma caixa "indefinido" (de data/interim, se existir)."""
    imagens = ler_csv(RAIZ / "data/interim/luminancia_openimages_imagens.csv")
    caixas = ler_csv(RAIZ / "data/interim/luminancia_openimages_caixas.csv")
    if not imagens or not caixas:
        return {}
    indefinidas = {c["image_id"] for c in caixas if c["estado"] == "indefinido"}
    cont = {"noite": Counter(), "anoitecer": Counter()}
    for r in imagens:
        media = float(r["cena_media"])
        if media < FAIXA_DIA:
            faixa = cont["noite" if media < FAIXA_NOITE else "anoitecer"]
            faixa["total"] += 1
            faixa["excluidas"] += r["image_id"] in indefinidas
    return cont


def coletar() -> dict:
    d = {}
    resumo = {}
    for r in ler_csv(RAIZ / "data/manifests/dataset_composicao.csv"):
        k = (r["split"], r["grupo"])
        c = resumo.setdefault(k, Counter())
        c["imagens"] += 1
        c["acesas"] += int(r["acesas"])
        c["apagadas"] += int(r["apagadas"])
    d["composicao"] = resumo
    d["totais"] = {s: Counter() for s in ("train", "val", "test")}
    for (s, _), c in resumo.items():
        d["totais"][s].update(c)
    d["oi_imagens"] = len(ler_csv(RAIZ / "data/manifests/openimages_imagens.csv"))
    # Registros do CSV, não linhas do arquivo. Origem da caixa: xclick = desenhada à mão;
    # activemil = proposta por algoritmo e confirmada por humanos.
    origens = Counter(r["origem"] for r in ler_csv(RAIZ / "data/manifests/openimages_caixas.csv"))
    d["oi_caixas"] = sum(origens.values())
    d["oi_frac_xclick"] = origens["xclick"] / max(1, d["oi_caixas"])
    d["chennai_manifesto"] = Counter(r["estado_pasta"] for r in ler_csv(RAIZ / "data/manifests/chennai_imagens.csv"))
    sf311 = ler_csv(RAIZ / "data/manifests/sf311.csv")
    d["sf311_manifesto"] = Counter(r["grupo"] for r in sf311)
    # Resultado por chamado noturno (diagnóstico e revisão), para contar chamados automáticos de cada tipo.
    d["sf311_noite_linhas"] = ler_csv(RAIZ / "results/avaliacao/imagens_sf311_noite.csv")
    d["sf311_inicio"] = min((r["aberto_em"] for r in sf311 if r["grupo"] == "noturno_queimada"), default="")
    # A regra de distância do pool de treino só alcança chamados de avaliação com coordenadas.
    d["sf311_sem_coordenadas"] = sum(1 for r in sf311 if not (r.get("lat") and r.get("lon")))
    pool = ler_csv(RAIZ / "data/manifests/sf311_treino.csv")
    # Pseudo-rótulos do v1 no pool (montar_dataset_v2.py): base da tentativa de autotreinamento.
    d["pseudo_v2"] = ler_csv(RAIZ / "data/manifests/pseudo_rotulos_v2.csv")
    d["sf311_treino_dist_min"] = min((float(v) for r in pool if (v := r.get("distancia_min_teste_m"))), default=None)
    # Fotos repetidas tiradas dos manifestos de avaliação (scripts/dados/deduplicar_manifestos.py).
    d["duplicatas"] = Counter("conflito" if r["motivo"] == "rotulos_conflitantes" else "copia"
                              for r in ler_csv(RAIZ / "data/manifests/duplicatas_removidas.csv"))
    d["ambiguas"] = contar_ambiguas()
    d["deteccao"] = ler_json(RAIZ / "results/avaliacao/deteccao.json")
    d["imagens"] = ler_json(RAIZ / "results/avaliacao/imagens.json")
    d["fps"] = ler_json(RAIZ / "results/avaliacao/fps.json")
    d["treino"] = ler_json(RAIZ / "results/treino/resumo_treino.json")
    d["treino_v2"] = ler_json(RAIZ / "results/treino_v2/resumo_treino.json")
    d["modelo"] = ler_json(RAIZ / "results/modelo_final.json")
    d["videos"] = {p.stem: ler_json(p) for p in sorted((RAIZ / "results/videos").glob("*.json"))}
    d["creditos"] = ler_csv(RAIZ / "results/exemplos/creditos.csv")
    return d


CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.4pt; line-height: 1.45; color: #1d1f24; }
h1 { font-size: 22pt; color: #7a0c3a; margin: 0 0 4px; line-height: 1.15; }
h2 { font-size: 14pt; color: #7a0c3a; border-bottom: 2px solid #f0c2d4; padding-bottom: 3px; margin: 20px 0 8px; }
h3 { font-size: 11.5pt; margin: 14px 0 6px; color: #2b2e36; }
p { margin: 0 0 8px; text-align: justify; }
.sub { color: #555b66; font-size: 11pt; margin-bottom: 14px; }
.capa { border-left: 6px solid #e8195e; padding: 6px 0 6px 16px; margin-bottom: 14px; }
.equipe td { padding: 2px 14px 2px 0; border: none; }
.equipe { border: none; margin-bottom: 6px; }
.caixa { background: #f6f7f9; border: 1px solid #e1e4ea; border-radius: 6px; padding: 10px 12px; margin: 8px 0 10px; }
.caixa p:last-child { margin-bottom: 0; }
.palavras { color: #777d88; font-size: 8.5pt; text-align: right; margin-top: -4px; }
.pendente { color: #9b1c1c; background: #fff4f4; border: 1px dashed #d33; border-radius: 4px; padding: 4px 8px; text-align: left; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 9.2pt; page-break-inside: avoid; }
th { background: #7a0c3a; color: #fff; text-align: left; padding: 4px 6px; font-weight: 600; }
td { border-bottom: 1px solid #e3e5ea; padding: 3.5px 6px; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfc; }
figure { margin: 8px 0 12px; page-break-inside: avoid; text-align: center; }
figure img { border: 1px solid #d9dce2; }
figcaption { font-size: 8.8pt; color: #555b66; margin-top: 3px; text-align: left; overflow-wrap: anywhere; }
.lado { display: flex; gap: 10px; }
.lado figure { flex: 1; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f1f2f5; padding: 0 3px; border-radius: 3px; }
pre { font-family: Consolas, monospace; font-size: 8.4pt; background: #f6f7f9; border: 1px solid #e1e4ea; padding: 8px; white-space: pre-wrap; page-break-inside: avoid; }
ol li, ul li { margin-bottom: 4px; }
.quebra { page-break-before: always; }
.refs li { font-size: 9pt; margin-bottom: 3px; }
.fluxo { display: flex; align-items: stretch; gap: 6px; margin: 8px 0 12px; font-size: 8.8pt; }
.fluxo div { flex: 1; border: 1px solid #d9b3c4; background: #fdf3f7; border-radius: 6px; padding: 6px; text-align: center; }
.fluxo b { display: block; color: #7a0c3a; margin-bottom: 2px; }
"""


def montar(d: dict, repo: str, video: str, base: Path) -> tuple[str, dict, list[str]]:
    """Monta o corpo do relatório. Devolve o HTML, a contagem de palavras e as pendências de resultados."""
    det, img, fps, treino, videos = d["deteccao"], d["imagens"], d["fps"], d["treino"], d["videos"]
    lim = LIMIARES_PADRAO
    pendencias: list[str] = []

    def pendente(desc: str) -> str:
        pendencias.append(desc)
        return f'<p class="pendente">Pendente: {html.escape(desc)}</p>'

    def fig(caminho: str, legenda: str) -> str:
        arquivo = RAIZ / caminho
        if not arquivo.exists():
            return pendente(f"{caminho} ({legenda.split('.')[0]})")
        return figura(rel(arquivo, base), legenda)

    def instancias(r: dict, classe: str) -> int | None:
        n = r.get("por_classe", {}).get(classe, {}).get("instancias")
        if n is None:
            n = r.get("instancias", {}).get(classe)
        if n is None and r.get("por_classe") and classe not in r["por_classe"]:
            n = 0  # o Ultralytics só lista as classes que têm caixa de referência no subconjunto
        return n

    # ------------------------------------------------------------ modelo avaliado (v1 ou v2)
    modelo = d["modelo"]
    versao = modelo.get("versao", "v1")
    comparacao = modelo.get("comparacao") or {}
    m1, m2 = comparacao.get("v1") or {}, comparacao.get("v2") or {}
    tv2 = modelo.get("treino_v2") or {}
    com_v2 = versao == "v2" or bool(m2)
    sha = modelo.get("sha256", "")

    tt = d["totais"]
    n_treino = tt["train"]["imagens"]
    sim = sum(d["composicao"].get((s, "noite_simulada"), Counter())["imagens"] for s in ("train", "val"))
    reservadas = sum(d["composicao"].get(("test", g), Counter())["imagens"] for g in ("noite_reservado", "anoitecer_reservado"))
    noite = det.get("noite", {})
    ch, sf, sfd = img.get("chennai", {}), img.get("sf311_noite", {}), img.get("sf311_dia", {})
    chs = ch.get("sistema", {})
    op = det.get("conf_operacao", lim.conf_minima)  # o limiar com que a Tabela 2 foi de fato calculada
    amb = d["ambiguas"]
    txt_amb = (f"{inteiro(amb['noite']['excluidas'])} das {inteiro(amb['noite']['total'])} imagens noturnas e "
               f"{inteiro(amb['anoitecer']['excluidas'])} das {inteiro(amb['anoitecer']['total'])} de anoitecer do Open Images"
               if amb else "")

    pc_noite = noite.get("por_classe", {})
    frase_noite = ""
    if CLASSE_ACESA in pc_noite:
        n_ac, n_ap = instancias(noite, CLASSE_ACESA), instancias(noite, CLASSE_APAGADA)
        frase_noite = (f"No teste noturno, cujo estado de referência vem da regra de brilho, o detector teve mAP50 de "
                       f"{num(pc_noite[CLASSE_ACESA].get('mAP50'), 2)} nas luminárias acesas ({inteiro(n_ac)} instâncias)"
                       + (f" e {num(pc_noite[CLASSE_APAGADA].get('mAP50'), 2)} nas apagadas ({inteiro(n_ap)})"
                          if CLASSE_APAGADA in pc_noite else "") + ". ")
    # Velocidade do pipeline completo: a do benchmark (filtro de privacidade em todo quadro) e a média de cada vídeo
    # da seção 5.4 (rastreamento, diagnóstico e filtro a cada dois quadros; sem desenho nem gravação).
    f_fps = fps.get("fps", {})
    pipeline_bench = max((v for v in (f_fps.get("pipeline_pytorch_com_privacidade"),
                                      f_fps.get("pipeline_onnx_com_privacidade")) if v), default=None)
    fps_videos = [r["fps_processamento_medio"] for r in videos.values() if r.get("fps_processamento_medio")]
    # Referência para comparar: o ritmo dos próprios vídeos (quadros por segundo da gravação).
    ritmos = [r["fps_video"] for r in videos.values() if r.get("fps_video")]
    ritmo = sum(ritmos) / len(ritmos) if ritmos else None
    faixa_pipeline = [v for v in (pipeline_bench, *fps_videos) if v]
    frase_fps = ""
    if faixa_pipeline:
        lo, hi = num(min(faixa_pipeline), 0), num(max(faixa_pipeline), 0)
        threads = fps.get("threads")
        frase_fps = (f"Em CPU{f' de {threads} threads' if threads else ''}, o pipeline completo roda a "
                     f"{lo if lo == hi else f'{lo}–{hi}'} quadros por segundo. ")
    # Limite de 250 palavras com todos os resultados presentes (conferido em main).
    resumo = (
        "Luminárias apagadas deixam calçadas e pontos de ônibus escuros e expõem quem anda a pé à noite; "
        "a prefeitura só descobre a falha por reclamações mal localizadas. "
        "O componente de visão computacional do Luz Segura recebe uma foto ou vídeo noturno, encontra cada luminária, "
        "diz se está acesa ou apagada e gera um diagnóstico explicável (luminária apagada, iluminação insuficiente, "
        "trecho escuro ou adequada) em JSON para o MVP. "
        f"Treinamos um detector YOLO26n com {inteiro(n_treino)} imagens do Open Images, parte delas convertidas em noite "
        "simulada, com o estado de cada luminária dado por uma regra de brilho, sem gravação própria"
        + (f"; o modelo final foi refinado por autotreinamento com {inteiro(tv2.get('aceitas'))} fotos reais fora da "
           "avaliação" if versao == "v2" and tv2 else "")
        + ". Na inferência, a mesma regra confere cada luminária e a posição do Sol define o período; se os sinais "
        "divergem ou não confirmam a falha, há revisão humana. "
        "Pessoas e veículos detectados são desfocados; nenhuma foto é guardada. "
        + frase_noite
        + (f"Em {inteiro(ch.get('imagens'))} fotos reais de Chennai, o sistema acusou {pct(chs.get('revocacao'))} das "
           f"apagadas com precisão de {pct(chs.get('precisao'))}; " if chs else "")
        + (f"em {inteiro(sf.get('imagens'))} chamados de luminária queimada do 311 de San Francisco, apontou a falha em "
           f"{pct(sf.get('acusou_falha_sistema'))}. " if sf else "")
        + frase_fps
        + ("O sistema é preciso, mas ainda deixa passar a maioria das luminárias apagadas em fotos de celular; "
           "os próximos passos são rótulos humanos nesse domínio e captura guiada pelo app."
           if chs and chs.get("revocacao", 1) < 0.5 and chs.get("precisao", 0) >= 0.7 else "")
    )
    objetivo = ("Treinar e avaliar um detector de objetos que identifique luminárias públicas acesas e apagadas em fotos "
                "e vídeos noturnos e, combinado à medição de brilho e à posição do Sol, gere em CPU um "
                "diagnóstico explicável e anonimizado para o MVP Luz Segura.")
    contagens = {"resumo": palavras(resumo), "objetivo": palavras(objetivo)}

    # ------------------------------------------------------------ tabelas
    linhas_comp = []
    nomes_grupo = {"noite": "Noite real", "anoitecer": "Anoitecer real", "dia": "Dia real",
                   "noite_simulada": "Noite simulada (de fotos diurnas)", "noite_reservado": "Noite real (reservado antes do treino)",
                   "anoitecer_reservado": "Anoitecer real (reservado antes do treino)"}
    for split, nome_split in (("train", "Treino"), ("val", "Validação"), ("test", "Teste")):
        for (s, g), c in sorted(d["composicao"].items()):
            if s == split:
                linhas_comp.append([nome_split, nomes_grupo.get(g, g), inteiro(c["imagens"]), inteiro(c["acesas"]), inteiro(c["apagadas"])])
    for split, nome_split in (("train", "Treino"), ("val", "Validação"), ("test", "Teste")):
        c = tt[split]
        linhas_comp.append([f"<b>{nome_split} (total)</b>", "", f"<b>{inteiro(c['imagens'])}</b>",
                            f"<b>{inteiro(c['acesas'])}</b>", f"<b>{inteiro(c['apagadas'])}</b>"])
    tab_comp = (tabela(["Divisão", "Grupo", "Imagens", "Caixas acesas", "Caixas apagadas"], linhas_comp) if d["composicao"]
                else pendente("data/manifests/dataset_composicao.csv (Tabela 1)"))

    # Tabela 2: precisão e revocação no ponto de operação (conf_minima), não no limiar de melhor F1 do Ultralytics.
    linhas_det = []
    nomes_det = {"todos": "Teste completo", "noite": "Noite", "anoitecer": "Anoitecer", "dia": "Dia"}
    for k in ("todos", "noite", "anoitecer", "dia"):
        r = det.get(k)
        if not r:
            continue
        pc = r.get("por_classe", {})
        sem_gabarito = r.get("falsos_positivos_sem_gabarito_op", {})
        pr, ap50 = {}, {}
        for c in (CLASSE_ACESA, CLASSE_APAGADA):
            if c in pc:
                # Precisão nula = nenhuma detecção da classe no ponto de operação (aparece como "–").
                pr[c] = f"{num(pc[c].get('precisao_op'))} / {num(pc[c].get('revocacao_op'))}"
                ap50[c] = num(pc[c].get("mAP50"))
            else:
                # Classe sem caixa de referência no subconjunto (acesa no "dia"): toda detecção dela é erro.
                pr[c] = (f"sem referência: {inteiro(sem_gabarito[c])} detecções em "
                         f"{inteiro(sem_gabarito.get('imagens_afetadas'))} imagens" if c in sem_gabarito else "–")
                ap50[c] = "–"
        linhas_det.append([nomes_det[k], inteiro(r.get("imagens")),
                           f"{inteiro(instancias(r, CLASSE_ACESA))} / {inteiro(instancias(r, CLASSE_APAGADA))}",
                           pr[CLASSE_ACESA], pr[CLASSE_APAGADA], ap50[CLASSE_ACESA], ap50[CLASSE_APAGADA],
                           num(r.get("mAP50")), num(r.get("mAP50_95"))])
    tab_det = (tabela(["Conjunto", "Imagens", "Instâncias (acesa / apagada)", "Acesa: precisão / revocação",
                       "Apagada: precisão / revocação", "mAP50 acesa", "mAP50 apagada", "mAP50", "mAP50-95"], linhas_det)
               if linhas_det else pendente("results/avaliacao/deteccao.json (Tabela 2)"))
    # As tabelas precisam ser do modelo escolhido e dos manifestos atuais (sem duplicatas).
    aviso_det = ""
    if sha and det.get("sha256_pesos") and det["sha256_pesos"] != sha:
        aviso_det = pendente("results/avaliacao/deteccao.json avaliou outros pesos (SHA-256 diferente do de "
                             "results/modelo_final.json): rodar scripts/avaliacao/comparar_modelos.py")
    aviso_aval = ""
    if sha and img.get("sha256_pesos") and img["sha256_pesos"] != sha:
        aviso_aval += pendente("results/avaliacao/imagens.json avaliou outros pesos (SHA-256 diferente do de "
                               "results/modelo_final.json): rodar scripts/avaliacao/comparar_modelos.py")
    n_ch_manifesto = sum(d["chennai_manifesto"].values())
    if ch and n_ch_manifesto and ch.get("imagens") != n_ch_manifesto:
        aviso_aval += pendente(f"results/avaliacao/imagens.json tem {ch.get('imagens')} fotos de Chennai e o manifesto tem "
                               f"{n_ch_manifesto}: refazer a avaliação por imagem")
    for chave, grupo, r in (("sf311_noite", "noturno_queimada", sf), ("sf311_dia", "diurno_acesa", sfd)):
        n_manifesto = d["sf311_manifesto"].get(grupo, 0)
        if r and n_manifesto and r.get("imagens") != n_manifesto:
            aviso_aval += pendente(f"results/avaliacao/imagens.json tem {r.get('imagens')} fotos em {chave} e o manifesto do "
                                   f"SF 311 tem {n_manifesto}: refazer a avaliação por imagem")

    # O chamado automático de Chennai depende do período informado (scripts/avaliacao/avaliar_imagens.py).
    periodo_chennai = ("Como as fotos não têm horário nem local, o período foi informado como noite; com o período estimado "
                       "só pelo brilho, toda luminária apagada iria para revisão humana e nenhum chamado seria automático.")
    if not ch:
        tab_chennai = pendente("results/avaliacao/imagens.json (Tabela 3)")
        legenda_chennai = f"<p><b>Tabela 3.</b> Dataset de Chennai (rótulo do próprio dataset). {periodo_chennai}</p>"
    else:
        n_ch, n_def = ch.get("imagens", 0), ch.get("com_defeito", 0)
        legenda_chennai = (f"<p><b>Tabela 3.</b> Dataset de Chennai ({inteiro(n_ch)} fotos ao anoitecer: {inteiro(n_ch - n_def)} "
                           f"luminárias funcionando e {inteiro(n_def)} com defeito; rótulo do próprio dataset). "
                           f"{periodo_chennai}</p>")
        nomes_modo = {"modelo": "Só o modelo (classe apagada)", "brilho": "Só a regra de brilho",
                      "sistema": "Sistema (modelo conferido pelo brilho)",
                      "chamado_automatico": "Chamado automático (sem revisão)"}
        linhas = []
        for modo, nome in nomes_modo.items():
            m = ch.get(modo, {})
            linhas.append([nome, m.get("vp", "–"), m.get("fp", "–"), m.get("fn", "–"), m.get("vn", "–"), pct(m.get("precisao")),
                           pct(m.get("revocacao")), num(m.get("f1"))])
        tab_chennai = tabela(["Decisão", "VP", "FP", "FN", "VN", "Precisão", "Revocação", "F1"], linhas)
        if "falso_alarme_piso_funcionando" in ch:
            # O critério do chão erra quando a faixa inferior da foto não mostra o chão (achado de revisão 5b).
            tab_chennai += (f"<p>Entre as {inteiro(n_ch - n_def)} fotos de luminárias funcionando, "
                            f"{pct(ch['falso_alarme_piso_funcionando'])} receberam um diagnóstico pelo chão "
                            "(iluminação insuficiente ou trecho escuro), que sempre vai para revisão humana: nessas fotos, "
                            "tiradas de perto e apontadas para o poste, a faixa inferior costuma mostrar céu ou fachada, "
                            "e não a calçada.</p>")

    if not sf:
        tab_sf = pendente("results/avaliacao/imagens.json (Tabela 4)")
    else:
        linhas = [[DIAGNOSTICOS_PT.get(k, k), v, pct(v / max(1, sf.get("imagens", 0)))]
                  for k, v in sf.get("diagnosticos", {}).items()]
        tab_sf = tabela(["Diagnóstico do sistema", "Chamados", "%"], linhas)
    frase_automatico = (f" Chamado automático de luminária apagada, sem revisão humana: {pct(sf['chamado_automatico'])}."
                        if "chamado_automatico" in sf else "")
    # O outro tipo de chamado automático: luminária acesa durante o dia sem revisão (Sol já alto no horário do chamado).
    n_acesa_auto = sum(1 for l in d["sf311_noite_linhas"]
                       if l.get("diagnostico") == "acesa_durante_o_dia" and l.get("revisao") == "False")
    if sf and n_acesa_auto:
        um = n_acesa_auto == 1
        frase_automatico += (f" {'Um chamado' if um else f'{inteiro(n_acesa_auto)} chamados'} "
                             f"({pct(n_acesa_auto / max(1, sf.get('imagens', 0)))}) "
                             f"{'virou chamado automático' if um else 'viraram chamados automáticos'} de outro tipo, luminária "
                             f"acesa durante o dia: {'no horário do chamado' if um else 'no horário de cada um'}, o Sol estava a "
                             f"pelo menos {num(lim.sol_min_acesa_de_dia, 0)}° acima do horizonte, e o brilho confirmou uma "
                             f"luminária acesa {'na foto' if um else 'em cada foto'}.")
    frase_dia = (f"<p>Nos {inteiro(sfd.get('imagens'))} chamados diurnos de luminária acesa o tempo todo (mesma fonte), o sistema "
                 f"apontou luminária acesa durante o dia em {pct(sfd.get('acusou_acesa_de_dia'))} das fotos.</p>" if sfd else "")

    if not fps:
        tab_fps = pendente("results/avaliacao/fps.json (velocidade em CPU)")
    else:
        f = fps.get("fps", {})
        cfg = (f"CPU AMD Ryzen 7 5700X3D de computador de mesa, {fps.get('threads', '–')} threads, "
               f"quadros {str(fps.get('resolucao', '–')).replace('x', '×')}")
        tab_fps = tabela([f"Configuração ({cfg})", "Quadros por segundo"], [
            ["Detector em PyTorch", num(f.get("detector_pytorch"), 1)], ["Detector em ONNX Runtime", num(f.get("detector_onnx"), 1)],
            ["Pipeline completo em PyTorch (detector + brilho + filtro de privacidade)", num(f.get("pipeline_pytorch_com_privacidade"), 1)],
            ["Pipeline completo com o detector em ONNX Runtime (filtro de privacidade em PyTorch)",
             num(f.get("pipeline_onnx_com_privacidade"), 1)]])
        # O Ultralytics limita o PyTorch a 2 threads quando um modelo ONNX é carregado em CPU: no pipeline,
        # o ganho do detector em ONNX se perde no filtro de privacidade, que roda em PyTorch.
        tab_fps += (f"<p>Medido em {inteiro(fps.get('quadros'))} quadros do vídeo <code>{html.escape(str(fps.get('video', '')))}</code>"
                    f" (a partir de {num(fps.get('inicio_s'), 0)} s, depois da cartela de abertura), com o filtro de "
                    "privacidade em todo quadro (no vídeo anotado ele roda a cada dois quadros). Com o detector em "
                    f"ONNX, o Ultralytics reduz o PyTorch a {fps.get('threads_pytorch_com_onnx', 2)} threads, e o filtro "
                    "de privacidade, que roda em PyTorch, fica mais lento: por isso os vídeos da demonstração usam os "
                    "pesos em PyTorch.</p>")
        # Referência para falar em tempo real: o ritmo em que os próprios vídeos da seção 5.4 foram gravados.
        detectores = {"PyTorch": f.get("detector_pytorch"), "ONNX": f.get("detector_onnx")}
        medidos = [nome for nome, v in detectores.items() if v]
        if ritmo and pipeline_bench and fps_videos and medidos:
            acima = [nome for nome in medidos if detectores[nome] > ritmo]
            frase_det = ("passa desse ritmo" if len(acima) == len(medidos)
                         else f"passa desse ritmo só em {acima[0]}" if acima else "não chega a esse ritmo")
            lo_v, hi_v = min(fps_videos), max(fps_videos)
            nos_videos = (f"{pct(lo_v / ritmo)} nos próprios vídeos ({num(lo_v, 1)} quadros por segundo)" if lo_v == hi_v
                          else f"entre {pct(lo_v / ritmo)} e {pct(hi_v / ritmo)} nos próprios vídeos ({num(lo_v, 1)} a "
                               f"{num(hi_v, 1)} quadros por segundo)")
            tab_fps += (f"<p>Para comparar, os vídeos da seção 5.4 foram gravados a {num(ritmo, 2)} quadros por segundo. O "
                        f"detector sozinho {frase_det}; o pipeline completo fica em {pct(pipeline_bench / ritmo)} dele com o "
                        f"filtro de privacidade em todo quadro e {nos_videos}, com rastreamento, diagnóstico e o filtro a cada "
                        "dois quadros.</p>")

    if not videos:
        tab_videos = pendente("results/videos/*.json (vídeos noturnos processados)")
    else:
        linhas = []
        for vid, r in videos.items():
            u = r.get("luminarias_unicas", {})
            apagadas = r.get("luminarias_apagadas", [])
            revisao = (sum(1 for a in apagadas if a.get("requer_revisao_humana"))
                       if all("requer_revisao_humana" in a for a in apagadas) else None)
            linhas.append([vid.replace("_", " "), r.get("fonte", {}).get("local", "–"), f"{num(r.get('duracao_s'), 0)} s",
                           inteiro(u.get(CLASSE_ACESA)), inteiro(u.get(CLASSE_APAGADA)), inteiro(revisao),
                           f"{num(r.get('segundos_em_trecho_escuro'), 0)} s", num(r.get("fps_processamento_medio"), 1)])
        tab_videos = tabela(["Vídeo", "Local", "Duração", "Acesas (únicas)", "Apagadas (únicas)", "Apagadas em revisão humana",
                             "Trecho escuro", "Quadros/s"], linhas)
    # Os vídeos precisam vir dos mesmos pesos das tabelas (o .onnx antigo sobrevivia a um novo treino).
    aviso_videos = ""
    if sha and videos:
        outros = [vid for vid, r in videos.items() if r.get("sha256") != sha]
        if outros:
            aviso_videos = pendente("vídeos processados com outros pesos ou sem registro dos pesos (refazer "
                                    f"scripts/demo/processar_videos.py): {', '.join(outros)}")

    tab_escolha = ""
    if com_v2:
        if m2:
            linhas = []
            for chave, rotulo, formato in METRICAS_ESCOLHA:
                fmt = pct if formato == "pct" else num
                linhas.append([rotulo, fmt(m1.get(chave)), fmt(m2.get(chave))])

            def criterio(m: dict) -> float | None:
                a, b = m.get("chennai_f1_sistema"), m.get("sf311_noite_acusou_falha_sistema")
                return None if a is None or b is None else (a + b) / 2
            linhas.append(["<b>Critério: média de F1 (Chennai) e falha confirmada (SF 311 noturno)</b>",
                           f"<b>{num(criterio(m1))}</b>", f"<b>{num(criterio(m2))}</b>"])
            tab_escolha = tabela(["Métrica", "v1", "v2 (autotreino)"], linhas)
        else:
            tab_escolha = pendente("comparacao.v2 em results/modelo_final.json (Tabela 5)")
    regra = (html.escape(modelo.get("regra", "")) or pendente("regra em results/modelo_final.json")) if com_v2 else ""

    # ------------------------------------------------------------ exemplos
    # Todos os exemplos de results/exemplos/creditos.csv, na ordem do arquivo: a curadoria é feita em
    # scripts/demo/exemplos_fotos.py, que também escreve a nota de interpretação de cada um (coluna "nota").
    # A numeração continua a das Figuras 1 e 2 (seção 5.1).
    if not d["creditos"]:
        exemplos_html = pendente("results/exemplos/creditos.csv (exemplos anotados)")
    else:
        figs = []
        for n_fig, c in enumerate(d["creditos"], 3):
            res = ler_json(RAIZ / "results/exemplos" / c["exemplo"].replace(".jpg", ".json"))
            licenca = html.escape(nome_licenca(c.get("licenca", "")))
            # CC BY pede título, autor, link da obra e da licença e a indicação de que a foto foi modificada.
            if c.get("titulo"):
                fonte = (f'Adaptado de "{html.escape(c.get("titulo", ""))}" ({html.escape(c.get("link", ""))}), '
                         f'de {html.escape(c.get("autor", ""))}, {licenca}.')
            else:
                fonte = f"Fonte: {html.escape(c.get('fonte', ''))}, {html.escape(c.get('autor', ''))}, {licenca}."
            if c.get("modificacao"):
                fonte += f" Modificações: {html.escape(c['modificacao'])}."
            diag = res.get("diagnostico", "")
            nome_diag = DIAGNOSTICOS_PT.get(diag, diag.replace("_", " "))
            partes = [f"<b>{html.escape(nome_diag[:1].upper() + nome_diag[1:])}</b>"]
            if res.get("severidade") in PRIORIDADE_PT:
                partes.append(PRIORIDADE_PT[res["severidade"]])
            if res.get("requer_revisao_humana"):
                partes.append("revisão humana")
            nota = (c.get("nota") or "").strip()
            if nota and not nota.endswith((".", "!", "?")):
                nota += "."
            leg = f"Figura {n_fig}. {' · '.join(partes)}.{' ' + html.escape(nota) if nota else ''} {fonte}"
            figs.append(fig(f"results/exemplos/{c['exemplo']}", leg))
        pares = ["<div class='lado'>" + "".join(figs[i:i + 2]) + "</div>" for i in range(0, len(figs), 2)]
        exemplos_html = "".join(pares)
    tem_sf = any(c.get("exemplo", "").startswith("sf311_") for c in d["creditos"])
    tem_oi = any(c.get("exemplo", "").startswith("openimages_") for c in d["creditos"])
    legenda_exemplos = (
        "Imagens já anonimizadas pelo próprio sistema (pessoas e veículos detectados pelo filtro foram desfocados). "
        "Caixas âmbar: luminária acesa; magenta: apagada. O sinal ? depois da confiança indica que a medição de brilho "
        "contradiz a classe prevista."
        + (" Nas fotos do SF 311, o período vem da posição do Sol no horário e no local do chamado." if tem_sf else "")
        + (" As fotos do Open Images são do teste reservado, nunca usado no treino; como não têm horário nem local, foram "
           "analisadas com o período informado como noite." if tem_oi else ""))

    # ------------------------------------------------------------ textos que dependem dos resultados
    def texto_de_treino(t: dict, versao_treino: str = "") -> str:
        # scripts/treino/resumir_treino.py escolhe a época como o Ultralytics escolhe o best.pt: maior mAP50-95.
        return (f"O treino{versao_treino} rodou {t.get('epocas_executadas')} épocas em {num(t.get('horas'), 1)} h; os pesos são "
                f"os da época {t.get('melhor_epoca')}, a de maior mAP50-95 na validação, que é o critério de escolha do "
                f"Ultralytics (mAP50 {num(t.get('melhor_map50'))}, mAP50-95 {num(t.get('melhor_map50_95'))}).")

    texto_treino = (texto_de_treino(treino, " do v1" if com_v2 else "") if treino
                    else pendente("results/treino/resumo_treino.json (resumo do treino)"))

    # A seção 5.6 compara as duas versões e a Figura 1 é sempre o treino do v1: a frase não pode cobri-las.
    procedencia = (f"<p>Os resultados das seções 5.1 a 5.5 são do modelo <b>{html.escape(versao)}</b>"
                   f"{' (com autotreinamento; só a Figura 1 mostra o treino do v1)' if versao == 'v2' else ''}"
                   f"{', escolhido pela regra pré-registrada (seção 5.6)' if com_v2 else ''}: "
                   f"pesos <code>{html.escape(modelo.get('pesos', ''))}</code>, publicados como "
                   f"<code>models/luzsegura_yolo26n.pt</code> (SHA-256 <code>{html.escape(sha[:16])}…</code>).</p>" if modelo
                   else pendente("results/modelo_final.json (versão e SHA-256 do modelo avaliado)"))

    passo_v2 = ""
    passo_escolha = ""
    inicio = d["sf311_inicio"]
    antes = (f", todos abertos antes do chamado noturno de avaliação mais antigo "
             f"({datetime.fromisoformat(inicio).strftime('%d/%m/%Y')})" if inicio else "")
    dist = d["sf311_treino_dist_min"]
    longe = (f", nenhum a menos de {math.floor(dist)} m de um chamado de avaliação"
             f"{' com coordenadas' if d['sf311_sem_coordenadas'] else ''}" if dist is not None else "")
    if com_v2:
        if tv2:
            treino_v2 = f" {texto_de_treino(d['treino_v2'], ' do v2')}" if d["treino_v2"] else ""
            corpo_v2 = (
                "As fotos que o Luz Segura recebe na prática são de celular, diferentes das fotos do Flickr. Para aproximar "
                "os domínios sem gravar nada, ajustamos os pesos do v1 com um conjunto separado de chamados reais do SF 311: "
                f"{inteiro(tv2.get('fotos_pool'))} fotos de chamados noturnos de luminária queimada{antes}{longe} e sem nenhuma "
                "foto idêntica às de avaliação. O modelo v1 propôs as caixas e a regra de brilho definiu o estado de cada uma "
                "(onde o brilho não decidia, valeu a classe prevista pelo v1, só com confiança alta); "
                "ficaram só as fotos sem caixa ambígua e com ao menos uma luminária apagada, como diz o chamado: "
                f"{inteiro(tv2.get('aceitas'))} fotos, com {inteiro(tv2.get('caixas_acesas'))} caixas acesas e "
                f"{inteiro(tv2.get('caixas_apagadas'))} apagadas. A lista de treino do v2 teve {inteiro(tv2.get('lista_treino'))} "
                "entradas, que misturam essas fotos, repetidas para pesarem mais, com parte do treino do v1; a validação "
                f"continuou a do v1.{treino_v2} Chennai e os chamados de avaliação do SF 311 continuam fora de qualquer treino.")
        else:
            corpo_v2 = pendente("treino_v2 em results/modelo_final.json (autotreinamento do v2)")
        passo_v2 = f"<li><b>Autotreinamento (modelo v2).</b> {corpo_v2}</li>"
        passo_escolha = (
            f"<li><b>Escolha pré-registrada do modelo.</b> {regra} Como a escolha usa os próprios conjuntos de avaliação, "
            "o critério foi definido de antemão e só compara duas versões; a comparação está na Tabela 5 (seção 5.6).</li>")
    elif d["pseudo_v2"]:
        # O v2 não foi treinado: o relatório mostra por quê, com os números dos pseudo-rótulos.
        pseudo = d["pseudo_v2"]
        aceitas = sum(r.get("aceita") == "1" for r in pseudo)
        motivos = Counter(r.get("motivo") for r in pseudo if r.get("aceita") != "1")
        nomes = {"sem_apagada": "o v1 só encontrou luminárias acesas", "sem_caixa": "o v1 não encontrou luminária",
                 "brilho_indefinido": "o brilho não decidia o estado de uma caixa de confiança baixa",
                 "foto_ilegivel": "a foto não abriu"}
        razoes = "; ".join(f"em {inteiro(v)}, {nomes.get(k, k)}" for k, v in motivos.most_common())
        passo_v2 = (
            "<li><b>Tentativa de autotreinamento (v2, não treinado).</b> Para aproximar o modelo das fotos de celular "
            f"sem gravar nada, o v1 rotulou automaticamente {inteiro(len(pseudo))} fotos de chamados noturnos de "
            f"luminária queimada do SF 311, separadas das de avaliação{antes}{longe}. A regra de brilho definiu o "
            "estado de cada caixa, regiões duvidosas foram cobertas de cinza e só valia foto com ao menos uma "
            f"luminária apagada, como diz o chamado. Só {inteiro(aceitas)} fotos ({pct(aceitas / len(pseudo))}) "
            f"passaram: {razoes}. Com tão poucos rótulos, o v2 não foi treinado e o modelo entregue é o v1. O "
            "autotreinamento só reforça o que o v1 já vê, e o que falta é justamente encontrar a luminária apagada "
            "nessas fotos. O procedimento fica no repositório (<code>scripts/dados/montar_dataset_v2.py</code> e "
            "<code>--com-v2</code> em <code>scripts/reproduzir.py</code>).</li>")

    dup = d["duplicatas"]
    duplicatas = (f" Antes, tiramos desses conjuntos as fotos repetidas (mesmo conteúdo): {inteiro(dup['copia'])} cópias saíram, "
                  f"ficando uma de cada, e {inteiro(dup['conflito'])} fotos que apareciam com rótulos contraditórios foram "
                  "descartadas (registro em <code>data/manifests/duplicatas_removidas.csv</code>)." if dup else "")
    aval_b = ("nos chamados de avaliação do SF 311, nunca usados no treino (o autotreinamento do v2 usou outros chamados)"
              if com_v2 else "no SF 311, que nunca foram usados no treino")
    fig1 = fig("results/treino/results.png",
               f"Figura 1. Curvas de treino{' do v1' if com_v2 else ''}, por época: perdas de caixa (box, baseada no IoU entre "
               "a caixa prevista e a de referência), de classe (cls) e L1 das distâncias até as quatro bordas da caixa (l1), "
               "no treino (train) e na validação (val), e precisão, revocação, mAP50 e mAP50-95 na validação.")
    fig2 = fig("results/avaliacao/deteccao_noite/confusion_matrix_normalized.png",
               f"Figura 2. Matriz de confusão normalizada no teste noturno (detecções com confiança acima de "
               f"{num(CONF_VALIDACAO)}, o limiar da validação do Ultralytics, e não no ponto de operação de "
               f"{num(op, 2)}; IoU 0,5).")
    # Passo 8: o ONNX acelera o detector sozinho, mas não necessariamente o pipeline com o filtro de privacidade.
    dpt, donnx = f_fps.get("detector_pytorch"), f_fps.get("detector_onnx")
    ppt, ponnx = f_fps.get("pipeline_pytorch_com_privacidade"), f_fps.get("pipeline_onnx_com_privacidade")
    if None in (dpt, donnx, ppt, ponnx):
        frase_onnx = "o detector também pode ser exportado para ONNX (seção 5.3)"
    elif donnx > dpt and ppt >= ponnx:
        frase_onnx = (f"o detector também foi exportado para ONNX, que o deixa mais rápido sozinho ({num(donnx, 1)} contra "
                      f"{num(dpt, 1)} quadros por segundo); no pipeline completo, porém, o PyTorch foi mais rápido "
                      f"({num(ppt, 1)} contra {num(ponnx, 1)}), e é ele que os vídeos usam (seção 5.3)")
    else:
        frase_onnx = (f"o detector também foi exportado para ONNX: sozinho, ele roda a {num(donnx, 1)} quadros por segundo em "
                      f"ONNX e a {num(dpt, 1)} em PyTorch; no pipeline completo, a {num(ponnx, 1)} e a {num(ppt, 1)} (seção 5.3)")
    n_sf_noite, n_sf_dia = d["sf311_manifesto"].get("noturno_queimada", 0), d["sf311_manifesto"].get("diurno_acesa", 0)
    ambiguas_passo = f" ({txt_amb}, fora do treino e também do teste)" if txt_amb else ""
    ambiguas_legenda = f" ({txt_amb})" if txt_amb else ""
    links = (f"<p><b>Repositório:</b> <a href='{html.escape(repo)}'>{html.escape(repo)}</a><br><b>Vídeo de demonstração:</b> "
             f"<a href='{html.escape(video)}'>{html.escape(video)}</a></p>") if repo and video else ""
    secao_escolha = (f"""<h3>5.6 Escolha do modelo final (v1 × v2)</h3>
<p><b>Tabela 5.</b> Comparação entre o modelo v1 e o v2 (autotreinamento) nos mesmos conjuntos de avaliação, pela regra
pré-registrada descrita no último passo da metodologia. Modelo escolhido: <b>{html.escape(versao)}</b>.</p>
{tab_escolha}""" if com_v2 else "")

    corpo = f"""
<div class="capa">
  <h1>Luz Segura: detecção de luminárias públicas apagadas e trechos escuros com visão computacional</h1>
  <div class="sub">Applied Computer Vision · Enxergando o mundo real · FIAP Global Solution 2026.2 · Engenharia de Software, 4º ano (4ES)<br>
  Professores: Arnaldo Viana e Paulo Rodrigues</div>
  <table class="equipe">{''.join(f'<tr><td>{n}</td><td>{rm}</td></tr>' for n, rm in EQUIPE)}</table>
  {links}
</div>

<h2>Resumo</h2>
<div class="caixa"><p>{resumo}</p></div>
<div class="palavras">{contagens['resumo']} palavras</div>

<h2>1. Introdução</h2>
<p>A iluminação pública é uma das infraestruturas que mais pesam na sensação de segurança de quem anda a pé à noite:
trabalhadores de turno, estudantes de cursos noturnos e idosos que dependem do transporte público. Uma luminária queimada em frente
a um ponto de ônibus muda a rotina de quem passa por ali, mas raramente é notada pelo poder público com rapidez. O ciclo atual depende
do cidadão: alguém percebe, liga ou abre uma solicitação, descreve o local de memória, e a equipe de manutenção recebe registros duplicados,
mal localizados e sem prioridade. O Luz Segura, CitizenTech desenvolvida pela equipe na Global Solution 2026, ataca esse ciclo em
sete componentes integrados; este relatório trata do componente de <b>visão computacional</b>, responsável por "enxergar o problema por
imagens e transformar o que vê em informação".</p>
<p>Câmeras já são usadas para monitorar iluminação pública. O MIT, em parceria com a Ferrovial, mostrou em 2016 que câmeras e sensores
montados em veículos conseguem mapear luminárias com defeito ao percorrer a cidade [7]; o projeto UMBRELLA, no Reino Unido, publicou
imagens de câmeras instaladas nos próprios postes para monitorar se a lâmpada está acesa [8]; e trabalhos recentes aplicam detectores
da família YOLO a vídeos noturnos de câmeras veiculares [9]. No Brasil, prefeituras como a de São José dos Campos já usam inteligência
artificial na zeladoria urbana [10]. Nossa contribuição é juntar três ideias em um componente pequeno o bastante para rodar só com CPU,
sem placa de vídeo: (1) um detector treinado para separar luminária acesa de apagada, (2) uma verificação física complementar, baseada
no brilho da própria imagem e na posição do Sol, que explica cada decisão e aciona revisão humana quando há dúvida, e (3) privacidade
por padrão: pessoas e veículos detectados pelo filtro são desfocados antes da exibição (à noite ou em ângulos incomuns, algum pode
escapar), e o sistema não guarda as fotos (seção 3.4).</p>
<p>Uma restrição definiu o projeto: <b>nenhuma foto ou vídeo foi gravado pela equipe</b>. Todos os dados vêm de fontes públicas ou
foram simulados com geração documentada, como o briefing da GS permite; as fontes têm licença aberta, exceto o dataset de Chennai, que
não declara licença e por isso entra só nas métricas, com citação, sem ser redistribuído nem exibido.</p>

<h2>2. Objetivo</h2>
<div class="caixa"><p>{objetivo}</p></div>
<div class="palavras">{contagens['objetivo']} palavras</div>

<h2>3. Explicação técnica das ferramentas</h2>
<h3>3.1 Detector de objetos: YOLO26n (Ultralytics)</h3>
<p>Detectores de um estágio da família YOLO preveem, em uma única passada da rede, a caixa, a classe e a confiança de cada objeto. Usamos
o <b>YOLO26n</b>, a versão nano da geração YOLO26 da biblioteca Ultralytics [1]: cerca de 2,4 milhões de parâmetros e 5,4 GFLOPs, com
cabeça de detecção de ponta a ponta, que dispensa a etapa de supressão de não máximos (NMS) na inferência e por isso é
especialmente rápida em CPU. Partimos dos pesos pré-treinados no COCO (aprendizado por transferência) e substituímos a cabeça por duas
classes: <code>luminaria_acesa</code> e <code>luminaria_apagada</code>. É o "treinamento customizado de uma arquitetura compatível com o
problema" pedido na atividade: o problema pede um modelo leve, que rode só com CPU, sem placa de vídeo. Medimos a velocidade num
computador de mesa (seção 5.3); celulares e computadores de bordo de ônibus seriam alvos naturais, mas não foram testados.</p>
<h3>3.2 Verificação física: brilho e posição do Sol</h3>
<p>Para cada caixa detectada, medimos o brilho (luma BT.601, de 0 a 255) com OpenCV, numa cópia da imagem reduzida a
{lim.lado_medida} px no lado maior, a resolução em que a regra foi calibrada. Uma luminária acesa contém uma <b>mancha contígua de
pixels saturados</b> (≥ {lim.pixel_nucleo}, com pelo menos {lim.nucleo_min_pixels} pixels), encontrada por rotulagem de componentes
conectados; uma luminária apagada não tem essa mancha e não é mais clara que o anel ao redor dela (contraste
≤ {num(lim.contraste_max_apagada, 2)}). É a mesma regra que definiu o estado das luminárias nos rótulos de treino (seção 4, passo 2):
reaplicada à caixa prevista, ela não é um juiz independente do modelo, mas aponta contradições, como uma luminária prevista como
apagada que mostra uma mancha de luz saturada. O brilho médio da faixa inferior da imagem ({lim.fracao_piso * 100:.0f}%, onde ficam
calçada e rua) indica se o chão está escuro. Como o modo noturno dos celulares clareia fotos tiradas à noite, o <b>período</b> (dia,
transição ou noite) é decidido pela altura do Sol, calculada com o algoritmo da NOAA [6] a partir do horário e do local da captura; o
horário precisa trazer o fuso, e sem ele a análise é recusada, para não trocar dia por noite. Só na falta de horário e local usamos o
brilho da cena como estimativa, e então nenhum chamado é aberto sem revisão humana. O período também pode ser informado por quem envia
a imagem (<code>periodo</code> na API, o campo Período no app e <code>--periodo</code> na linha de comando) e, nesse caso, prevalece
sobre o Sol e o brilho; foi assim em Chennai, cujas fotos não têm horário nem local, nos vídeos noturnos (seção 5.4) e nos exemplos do
Open Images (seção 5.5).</p>
<h3>3.3 Rastreamento em vídeo: ByteTrack</h3>
<p>Em vídeo, cada luminária recebe um identificador estável com o ByteTrack [5], que associa detecções entre quadros. O estado final de
cada luminária é a votação dos quadros em que ela aparece, o que permite contar luminárias <b>distintas</b> acesas e apagadas em um
trajeto, em vez de contar a mesma luminária várias vezes.</p>
<h3>3.4 Privacidade: anonimização com YOLO26n do COCO</h3>
<p>Um segundo YOLO26n, sem treino adicional, localiza pessoas, bicicletas, motos, carros, ônibus e caminhões; cada região detectada é
pixelada e desfocada de forma irreversível antes da exibição (em vídeo, o filtro detecta a cada dois quadros e desfoca todos os quadros
com as caixas mais recentes). O filtro não é infalível: à noite ou em ângulos incomuns, alguma pessoa ou veículo pode escapar dele.
Nada é guardado: a API analisa a foto em memória e não grava nada em disco; a interface web de demonstração mantém os arquivos enviados
só no seu cache temporário, apagado em até 20 minutos ou quando o app é fechado, e o último vídeo anotado, já anonimizado, fica numa
pasta temporária até a próxima análise de vídeo ou até o app ser fechado. Desfocar pessoas e veículos e não guardar imagens segue o
princípio da necessidade da LGPD [11], que limita o tratamento de dados pessoais ao mínimo necessário para a finalidade.</p>
<h3>3.5 Demais ferramentas</h3>
{tabela(["Ferramenta", "Versão", "Uso no projeto"], [
        ["Python", "3.12.10", "linguagem de todo o componente"],
        ["Ultralytics", "8.4.215", "treino, validação, exportação e rastreamento do YOLO26n"],
        ["PyTorch (CPU)", "2.15.0", "treino da rede neural em CPU"],
        ["ONNX Runtime", "1.32.0", "inferência opcional do detector exportado para ONNX (seção 5.3)"],
        ["OpenCV", "5.1.0", "leitura de imagens e vídeos, medição de brilho, componentes conectados"],
        ["Pillow", "13.0.0", "desenho do painel com acentuação"],
        ["FastAPI e Uvicorn", "0.163 / 0.54", "API REST com o contrato JSON consumido pelo MVP"],
        ["Gradio", "6.32.0", "interface web de demonstração (foto e vídeo)"],
        ["imageio-ffmpeg", "0.8.1", "gravação dos vídeos anotados em H.264"],
        ["pytest", "9.2.0", "testes automatizados (regras, contrato, privacidade e API)"],
    ])}
<h3>3.6 Métricas de avaliação</h3>
<p>O <b>IoU</b> (interseção sobre união) mede a sobreposição de duas caixas: a área da interseção entre a caixa prevista e a de
referência dividida pela área da união das duas. Uma previsão é um verdadeiro positivo (VP) quando tem a classe certa e IoU ≥ 0,5 com
uma caixa de referência ainda sem par; a previsão sem par é um falso positivo (FP), e a caixa de referência que nenhuma previsão
encontrou é um falso negativo (FN). A <b>precisão</b>, VP/(VP + FP), é a fração das previsões que estão certas; a <b>revocação</b>,
VP/(VP + FN), é a fração das luminárias de referência encontradas; o <b>F1</b> é a média harmônica das duas,
2 × precisão × revocação/(precisão + revocação). Variando o limiar de confiança, precisão e revocação traçam uma curva, e a área sob ela
é a precisão média (AP) da classe. O <b>mAP50</b> é a média do AP das classes com IoU mínimo de 0,5; o <b>mAP50-95</b> é a média do AP
com IoU mínimo de 0,50 a 0,95, em passos de 0,05, e por isso também cobra caixas bem ajustadas. Na avaliação de ponta a ponta
(Tabela 3), as mesmas contas valem por foto: VP é uma foto com defeito em que o sistema acusou luminária apagada; FP, uma foto de
luminária funcionando em que ele acusou; FN, uma foto com defeito em que ele não acusou; e VN (verdadeiro negativo), uma foto de
luminária funcionando em que ele não acusou.</p>

<h2>4. Metodologia (passo a passo)</h2>
<div class="fluxo">
  <div><b>1. Dados públicos</b>Open Images, Chennai, SF 311, vídeos livres</div>
  <div><b>2. Regra de brilho</b>estado de cada caixa + conferência visual</div>
  <div><b>3. Noite simulada</b>fotos diurnas escurecidas (rotuladas)</div>
  <div><b>4. Treino</b>YOLO26n, 2 classes, CPU{'; v2 com autotreinamento' if com_v2 else ''}</div>
  <div><b>5. Diagnóstico</b>modelo + brilho + Sol → JSON</div>
  <div><b>6. Avaliação</b>teste reservado, Chennai, SF 311, velocidade{'; escolha v1 × v2' if com_v2 else ''}</div>
</div>
<ol>
<li><b>Coleta sem gravação própria.</b> Do Open Images V7 [2] extraímos todas as imagens com caixas da classe "Street light" feitas ou
conferidas por anotadores humanos ({pct(d['oi_frac_xclick'])} desenhadas à mão; as demais, de origem "activemil", foram propostas por
algoritmo e confirmadas por humanos). Descartamos imagens com caixas de grupo, desenhos ou rotação, ficando com {inteiro(d['oi_imagens'])}
imagens e {inteiro(d['oi_caixas'])} caixas, com autor e licença de cada foto registrados em <code>data/manifests/</code>; esses
manifestos registram a origem e o rótulo de cada imagem e vídeo usados, na linha das <i>datasheets</i> para conjuntos de dados [14].
Para testes independentes usamos {inteiro(n_ch_manifesto)} fotos do dataset de Chennai [3] (luminárias funcionando e com defeito, fotografadas ao
anoitecer) e {inteiro(n_sf_noite + n_sf_dia)} fotos de chamados reais do serviço 311 de San Francisco [4] ({inteiro(n_sf_noite)} noturnos
de luminária queimada e {inteiro(n_sf_dia)} diurnos de luminária acesa o tempo todo).{duplicatas}</li>
<li><b>Estado de cada luminária pela regra de brilho.</b> Medimos o brilho de cada imagem e de cada caixa. As imagens foram separadas em
noite (brilho médio &lt; {FAIXA_NOITE}), anoitecer ({FAIXA_NOITE} a {FAIXA_DIA}) e dia (≥ {FAIXA_DIA}). À noite e ao anoitecer, o estado
de cada caixa veio da regra da seção 3.2; imagens com qualquer caixa "indefinida" foram descartadas para não ensinar rótulos
duvidosos{ambiguas_passo}. De dia, toda luminária foi rotulada como apagada. Conferimos a regra visualmente em mosaicos de amostras
antes de usá-la.</li>
<li><b>Noite simulada.</b> Luminárias apagadas à noite são raras em bancos públicos. Por isso, {inteiro(sim)} fotos diurnas foram convertidas
em noite simulada (gama de 2,2 a 3,2, exposição de 35% a 65%, céu escurecido no topo, tonalidade de LED ou de vapor de sódio, ruído de
sensor e leve desfoque), mantendo as caixas e rotulando as luminárias como apagadas. Cada imagem simulada tem seus parâmetros no manifesto,
e nenhuma imagem simulada entra no teste. A queda de desempenho de detectores entre o dia e a noite é um problema conhecido de adaptação
de domínio, tratado na literatura com o realce adaptativo da imagem [12] ou com treino não supervisionado do dia para a noite [13]; a
noite simulada é a alternativa mais simples, que aproveita os rótulos das fotos diurnas.</li>
<li><b>Montagem do dataset.</b> Antes do treino, reservamos {inteiro(reservadas)} imagens noturnas e de anoitecer nunca usadas e as imagens
dos subconjuntos oficiais de validação e teste do Open Images como conjunto de teste. A composição final está na Tabela 1.</li>
<li><b>Treinamento.</b> YOLO26n pré-treinado no COCO, imagens de 640 px, lotes de 16, até 25 épocas com parada antecipada, otimizador
escolhido automaticamente pelo Ultralytics e aumentos de dados (mosaico, espelhamento, escala, translação e variação de cor). A variação de
brilho foi reduzida (hsv_v = 0,25) para não "apagar" artificialmente luminárias acesas. Tudo em CPU (AMD Ryzen 7 5700X3D). {texto_treino}</li>
{passo_v2}
<li><b>Diagnóstico explicável.</b> Cada detecção com confiança ≥ {num(op, 2)} é conferida pela regra de brilho. Luminária apagada à noite
gera o diagnóstico <i>luminária apagada</i> (prioridade alta se o chão também estiver escuro); sem luminária apagada, chão escuro com
luminária acesa vira <i>iluminação insuficiente</i>, e sem nenhuma acesa vira <i>trecho escuro</i>; com o chão claro, o diagnóstico é
<i>iluminação adequada</i>. De dia, luminária acesa confirmada pelo brilho indica fotocélula com defeito (<i>acesa durante o dia</i>);
sem ela, a foto recebe <i>fora do período noturno</i>, sem chamado. Chamado automático, sem revisão humana, só sai em dois casos:
luminária apagada confirmada pelo brilho, à noite, com o período conhecido e confiança ≥ {num(lim.conf_confiavel, 2)}; ou, de dia,
luminária acesa confirmada pelo brilho, com confiança ≥ {num(lim.conf_confiavel, 2)}, mais clara que o entorno e com o Sol a pelo
menos {num(lim.sol_min_acesa_de_dia, 0)}° acima do horizonte. Vão para revisão humana: a luminária apagada que o brilho contradiz ou
não confirma, cujas detecções têm todas confiança menor que {num(lim.conf_confiavel, 2)} ou que aparece na transição entre dia e noite
ou com o período estimado só pelo brilho da imagem; os diagnósticos baseados no brilho do chão (<i>iluminação insuficiente</i> e
<i>trecho escuro</i>), sempre; a imagem noturna ou de transição em que o brilho contradiz uma luminária acesa; a <i>iluminação
adequada</i> à noite ou na transição sem nenhuma luminária encontrada na foto, que vem só do brilho do chão; e a luminária acesa de dia
com confiança baixa, que não contrasta com o entorno ou com o Sol a menos de {num(lim.sol_min_acesa_de_dia, 0)}° (ou com o período
apenas informado). Fica um ponto cego: a <i>iluminação adequada</i> com alguma luminária acesa encontrada e sem contradição do brilho
não vai para revisão, e uma luminária queimada que o detector não vê ao lado de outras acesas passa despercebida. A saída segue o
contrato JSON v1 consumido pela automação, pelo pipeline de dados e pelo assistente do MVP.</li>
<li><b>Vídeo e privacidade.</b> Em vídeo, o rastreador conta luminárias distintas e mede o tempo em trecho escuro, e cada luminária
apagada do trajeto informa se precisa de revisão humana e por quê; o filtro de privacidade detecta pessoas e veículos a cada dois
quadros e desfoca todos os quadros com as caixas mais recentes; {frase_onnx}.</li>
<li><b>Avaliação.</b> (a) Métricas de detecção no teste reservado; (b) avaliação de ponta a ponta por imagem em Chennai e
{aval_b}; (c) velocidade em CPU; (d) análise qualitativa em vídeos noturnos de licença livre, incluindo a Avenida Paulista, em
São Paulo. Em Chennai, cujas fotos não têm horário nem local, o período foi informado como noite; no SF 311, ele veio da posição do
Sol no horário e no local de cada chamado.</li>
{passo_escolha}
</ol>
<p><b>Tabela 1.</b> Composição do dataset{' do modelo v1' if com_v2 else ''} (imagens e caixas por classe).</p>
{tab_comp}

<h2 class="quebra">5. Exemplos de resultados</h2>
{procedencia}
<h3>5.1 Detecção no conjunto de teste</h3>
<p><b>Tabela 2.</b> Métricas do detector em imagens nunca vistas no treino (IoU 0,5 para mAP50; média de IoU 0,5 a 0,95 para mAP50-95;
definições na seção 3.6).
Precisão e revocação de cada classe no ponto de operação do sistema (confiança ≥ {num(op, 2)}); instâncias são as caixas de referência de
cada classe. À noite e ao anoitecer, o estado de referência de cada caixa vem da própria regra de brilho (seção 3.2), e as imagens com
qualquer caixa indefinida ficaram fora do treino e também do teste{ambiguas_legenda}: essas linhas medem o quanto o modelo reproduz a
regra; as Tabelas 3 e 4 avaliam com rótulos que não vêm dela. De dia, toda luminária foi rotulada apagada; na linha Dia, a coluna da
classe acesa conta as detecções de luminária acesa, que esse rótulo trata como erro.</p>
{aviso_det}
{tab_det}
<div class="lado">
{fig1}
{fig2}
</div>
{{RESULTADOS_DETECCAO}}

<h3>5.2 Ponta a ponta: o sistema acusa a luminária apagada?</h3>
{aviso_aval}
{legenda_chennai}
{tab_chennai}
{{RESULTADOS_CHENNAI}}
<p><b>Tabela 4.</b> Diagnóstico do sistema para {inteiro(sf['imagens']) if 'imagens' in sf else 'os'} chamados noturnos reais de luminária queimada (San Francisco 311).
O período foi calculado pela posição do Sol no horário e no local de cada chamado.{frase_automatico}</p>
{tab_sf}
{frase_dia}
{{RESULTADOS_SF}}

<h3>5.3 Velocidade em CPU</h3>
{tab_fps}
<h3>5.4 Vídeos noturnos</h3>
{aviso_videos}
{tab_videos}
{{RESULTADOS_VIDEOS}}
<h3>5.5 Exemplos anotados</h3>
<p>{legenda_exemplos}</p>
{exemplos_html}
{secao_escolha}

<h2>6. Conclusão</h2>
{{CONCLUSAO}}

<h2>Referências</h2>
<ol class="refs">
<li>ULTRALYTICS. YOLO26. Documentação oficial, 2026. https://docs.ultralytics.com/models/yolo26/</li>
<li>KUZNETSOVA, A. et al. The Open Images Dataset V4. International Journal of Computer Vision, v. 128, p. 1956-1981, 2020. https://doi.org/10.1007/s11263-020-01316-z</li>
<li>S, E. F. S.; T, S.; P, S. K.; S, J. S. Comprehensive Dataset for Urban Streetlight Analysis. arXiv:2407.01117, 2024. https://github.com/Team16Project/Street-Light-Dataset</li>
<li>SAN FRANCISCO 311. 311 Cases (Streetlights). DataSF, licença ODC PDDL. https://data.sf.gov/resource/vw6y-z8j6</li>
<li>ZHANG, Y. et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV, 2022. https://arxiv.org/abs/2110.06864</li>
<li>NOAA GLOBAL MONITORING LABORATORY. Solar Calculation Details. https://gml.noaa.gov/grad/solcalc/calcdetails.html</li>
<li>KUMAR, S.; DESHPANDE, A.; HO, S. S.; KU, J. S.; SARMA, S. E. Urban Street Lighting Infrastructure Monitoring Using a Mobile Sensor Platform. IEEE Sensors Journal, v. 16, n. 12, p. 4981-4994, 2016. https://doi.org/10.1109/JSEN.2016.2552249</li>
<li>MAVROMATIS, I. et al. A dataset of images of public streetlights with operational monitoring using computer vision techniques. Data in Brief, v. 45, 108658, 2022. https://doi.org/10.1016/j.dib.2022.108658</li>
<li>AIYMBAY, S. et al. Applying Computer Vision for the Detection and Analysis of the Condition and Operation of Street Lighting. Symmetry, v. 17, n. 8, 1294, 2025. https://doi.org/10.3390/sym17081294</li>
<li>PREFEITURA DE SÃO JOSÉ DOS CAMPOS. Tecnologia com IA reforça cuidados e zeladoria em São José. 22 set. 2025. https://www.sjc.sp.gov.br/noticias/2025/setembro/22/tecnologia-com-ia-reforca-cuidados-e-zeladoria-em-sao-jose/</li>
<li>BRASIL. Lei nº 13.709, de 14 de agosto de 2018. Lei Geral de Proteção de Dados Pessoais (LGPD).</li>
<li>LIU, W. et al. Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions. AAAI, 2022. https://github.com/wenyyu/Image-Adaptive-YOLO</li>
<li>KENNERLEY, M. et al. 2PCNet: Two-Phase Consistency Training for Day-to-Night Unsupervised Domain Adaptive Object Detection. CVPR, 2023. https://arxiv.org/abs/2303.13853</li>
<li>GEBRU, T. et al. Datasheets for Datasets. Communications of the ACM, v. 64, n. 12, 2021. https://arxiv.org/abs/1803.09010</li>
</ol>
"""
    return corpo, contagens, pendencias


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="", help="URL pública do repositório no GitHub (padrão: chave \"repo\" de --textos)")
    ap.add_argument("--video", default="", help="URL do vídeo de demonstração, YouTube não listado (padrão: chave \"video\" de --textos)")
    ap.add_argument("--textos", type=Path, default=RAIZ / "docs/relatorio/textos_resultados.json",
                    help="parágrafos de análise dos resultados e links (escritos após a avaliação)")
    ap.add_argument("--rascunho", action="store_true",
                    help="gera mesmo sem textos, links ou resultados, marcando cada lacuna (versão de trabalho, fora do git)")
    ap.add_argument("--saida-html", type=Path, default=None,
                    help="padrão: docs/relatorio/relatorio.html (rascunho: data/outputs/relatorio/rascunho.html)")
    ap.add_argument("--saida-pdf", type=Path, default=None,
                    help="padrão: docs/Relatorio_LuzSegura_ACV.pdf (rascunho: data/outputs/relatorio/Relatorio_LuzSegura_ACV_rascunho.pdf)")
    args = ap.parse_args()
    # O rascunho nunca sobrescreve o PDF de entrega.
    saida_html = args.saida_html or RAIZ / ("data/outputs/relatorio/rascunho.html" if args.rascunho else "docs/relatorio/relatorio.html")
    saida_pdf = args.saida_pdf or RAIZ / ("data/outputs/relatorio/Relatorio_LuzSegura_ACV_rascunho.pdf" if args.rascunho
                                          else "docs/Relatorio_LuzSegura_ACV.pdf")

    d = coletar()
    textos = ler_json(args.textos)
    repo = args.repo or textos.get("repo", "")
    video = args.video or textos.get("video", "")
    faltando = [c for c in CHAVES_TEXTO if not (isinstance(textos.get(c), list) and textos[c])]
    sem_link = [n for n, v in (("repo", repo), ("video", video)) if not v]
    if sem_link:
        print(f"AVISO: sem {' e '.join(sem_link)}: o relatório sai sem o parágrafo de links "
              "(gere de novo com --repo e --video depois de publicar)", file=sys.stderr)
    corpo, contagens, pendencias = montar(d, repo, video, saida_html.resolve().parent)
    if (faltando or pendencias) and not args.rascunho:
        print("ERRO: relatório incompleto; use --rascunho para gerar uma versão de trabalho.", file=sys.stderr)
        if faltando:
            print(f"  faltam em {args.textos} (ou via --repo/--video): {', '.join(faltando)}", file=sys.stderr)
        for p in pendencias:
            print(f"  pendente: {p}", file=sys.stderr)
        return 1
    for chave in CHAVES_TEXTO:
        paragrafos = textos.get(chave) if isinstance(textos.get(chave), list) else []
        bloco = "".join(f"<p>{p}</p>" for p in paragrafos) or \
            f'<p class="pendente">Pendente: texto {chave} em {html.escape(args.textos.name)}</p>'
        corpo = corpo.replace("{" + chave + "}", bloco)
    if contagens["resumo"] > 250 or contagens["objetivo"] > 50:
        print(f"ERRO: limites de palavras excedidos: {contagens}", file=sys.stderr)
        return 1
    if args.rascunho:
        corpo = '<p class="pendente">RASCUNHO: versão de trabalho, não entregar.</p>' + corpo
    documento = (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
                 f"<title>Luz Segura · Relatório ACV{' (rascunho)' if args.rascunho else ''}</title>"
                 f"<style>{CSS}</style></head><body>{corpo}</body></html>")
    saida_html.parent.mkdir(parents=True, exist_ok=True)
    saida_html.write_text(documento, encoding="utf-8")
    navegador = (next((n for n in NAVEGADORES if n.exists()), None)
                 or next(filter(None, map(shutil.which, NAVEGADORES_PATH)), None))
    if not navegador:
        print("Edge, Chrome ou Chromium não encontrado: abra o HTML e imprima em PDF.", file=sys.stderr)
        return 1
    saida_pdf.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(navegador), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=10000", f"--print-to-pdf={saida_pdf.resolve()}",
                    saida_html.resolve().as_uri()], check=True, capture_output=True)
    if args.rascunho and (faltando or pendencias):
        print(f"AVISO: rascunho com {len(faltando) + len(pendencias)} pendência(s):", file=sys.stderr)
        for p in faltando + pendencias:
            print(f"  - {p}", file=sys.stderr)
    print(f"HTML: {saida_html}\nPDF: {saida_pdf}\npalavras: {contagens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
