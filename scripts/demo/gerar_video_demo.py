"""Monta o vídeo de demonstração do Luz Segura (1280x720, legendado em português, sem gravação própria).

Sequência: abertura, problema, como funciona, vídeos noturnos analisados quadro a quadro em CPU, o app e a
API em funcionamento (capturas de tela em docs/video/capturas), fotos reais de chamados com o JSON
gerado, resultados medidos, privacidade e créditos.
Todo número medido que aparece na tela vem dos arquivos em results/. Os trechos de vídeo vêm de
data/outputs/videos (gerados por scripts/demo/processar_videos.py).

Os créditos (na tela e na descrição do YouTube) são montados a partir de data/manifests/videos.csv,
só com os trechos que de fato entraram no vídeo: autor, link, licença e a indicação de que o trecho
foi adaptado, como as licenças CC BY e CC BY-SA exigem. Trecho sem vídeo é pulado com aviso;
trecho usado sem linha no manifesto interrompe a montagem.

Saídas: docs/video/luz_segura_demo.mp4 e docs/video/roteiro.md (narração opcional, com tempos,
e os créditos para a descrição do YouTube)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from luzsegura.config import LIMIARES_PADRAO
from luzsegura.desenho import FONTE_PERIODO_PT, PERIODO_PT, fonte
from luzsegura.video import EscritorVideo

L, A, FPS = 1280, 720, 30
FUNDO = (14, 17, 24)
DESTAQUE = (255, 190, 0)
MAGENTA = (240, 50, 210)
CINZA = (190, 196, 205)
EQUIPE = ["Arthur Bessa Pian (RM99215)", "Davi Almeida Desenzi (RM550849)", "Fabrício Gutierrez Saavedra (RM97631)",
          "João Victor Soares (RM551410)", "Márcio Gastaldi de Oliveira Soares (RM98811)"]
ADAPTACAO = "anotações e desfoque"
# Capturas de tela do app e da API rodando com o modelo final (arquivo, segundos, legenda, título no roteiro).
CAPTURAS = [
    ("app_foto.png", 8, "Interface web: o cidadão envia a foto com horário e local e recebe a imagem anotada, "
     "com pessoas e veículos detectados desfocados, e o diagnóstico explicado.", "App: foto"),
    ("app_json.png", 6, "O mesmo resultado em JSON, no contrato v1 que a automação e o app do MVP consomem.", "App: JSON"),
    ("api_docs.png", 6, "API REST para o MVP: POST /v1/analisar recebe a foto e devolve o JSON; a documentação "
     "interativa fica em /docs.", "API"),
]
# Crédito das fotos de chamados, nas capturas do app e nos painéis de fotos (a foto enviada aparece sem
# desfoque na entrada do app; o desfoque vale para a imagem anotada).
CREDITO_SF311 = ("Foto: chamado ao serviço 311 de San Francisco (DataSF, domínio público). "
                 "Na imagem anotada, pessoas e veículos detectados são desfocados.")
MODIFICACOES = "redimensionados, anotados e com pessoas e veículos detectados desfocados"


def dec(x: float, casas: int = 2) -> str:
    """Número com vírgula decimal (o vídeo é em português)."""
    return f"{x:.{casas}f}".replace(".", ",")


def porcento(x: float) -> str:
    # Arredondamento decimal, metade para cima (o mesmo critério do relatório).
    return f"{(Decimal(str(x)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP)}%"


def inteiro(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def plural(n: int, singular: str, plural_: str) -> str:
    return f"{inteiro(n)} {singular if n == 1 else plural_}"


def url_licenca(licenca: str) -> str:
    """Link da licença Creative Commons citada no manifesto ("" se não for uma licença CC)."""
    if "CC0" in licenca:
        return "https://creativecommons.org/publicdomain/zero/1.0/"
    m = re.search(r"CC (BY(?:-[A-Z]{2})*) (\d\.\d)", licenca)
    return f"https://creativecommons.org/licenses/{m[1].lower()}/{m[2]}/" if m else ""


def origem(pagina: str) -> str:
    if "wikimedia" in pagina:
        return "Wikimedia Commons"
    if "archive.org" in pagina:
        return "Internet Archive"
    return re.sub(r"^https?://(www\.)?", "", pagina).split("/")[0] or pagina


def quebrar(texto: str, f, largura_max: int, d: ImageDraw.ImageDraw) -> list[str]:
    linhas, atual = [], ""
    for palavra in texto.split():
        teste = f"{atual} {palavra}".strip()
        if d.textlength(teste, font=f) <= largura_max:
            atual = teste
        else:
            linhas.append(atual)
            atual = palavra
    if atual:
        linhas.append(atual)
    return linhas


def cartela(titulo: str, blocos: list[tuple[str, int, tuple]], rodape: str | None = None) -> np.ndarray:
    """Cartela de texto: título em destaque e blocos (texto, tamanho da fonte, cor)."""
    img = Image.new("RGB", (L, A), FUNDO)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 12, A), fill=DESTAQUE)
    y = 70
    ft = fonte(46)
    for linha in quebrar(titulo, ft, L - 180, d):
        d.text((90, y), linha, font=ft, fill=(255, 255, 255))
        y += 60
    y += 20
    for texto, tamanho, cor in blocos:
        f = fonte(tamanho)
        for linha in quebrar(texto, f, L - 180, d):
            d.text((90, y), linha, font=f, fill=cor)
            y += int(tamanho * 1.45)
        y += int(tamanho * 0.6)
    if y > A - 20:
        print(f"AVISO: texto da cartela \"{titulo}\" passa da borda inferior", file=sys.stderr)
    if rodape:
        d.text((90, A - 60), rodape, font=fonte(20), fill=(130, 136, 148))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _faixa_legenda(texto: str, credito: str | None, d: ImageDraw.ImageDraw) -> tuple[list[str], list[str], int]:
    """Linhas da legenda e do crédito e a altura (px) da faixa inferior que elas ocupam."""
    linhas = quebrar(texto, fonte(26), L - 120, d)
    # O crédito também quebra em linhas: cortado, deixaria de atribuir autor e licença.
    linhas_credito = quebrar(credito, fonte(17), L - 120, d) if credito else []
    return linhas, linhas_credito, 24 + 36 * len(linhas) + 24 * len(linhas_credito)


def altura_legenda(texto: str, credito: str | None = None) -> int:
    """Altura (px) da faixa que legenda() desenha na parte de baixo do quadro."""
    return _faixa_legenda(texto, credito, ImageDraw.Draw(Image.new("RGB", (1, 1))))[2]


def legenda(quadro: np.ndarray, texto: str, credito: str | None = None) -> np.ndarray:
    """Faixa inferior com a legenda (e o crédito da fonte, quando houver)."""
    img = Image.fromarray(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    f, fc = fonte(26), fonte(17)
    linhas, linhas_credito, altura = _faixa_legenda(texto, credito, d)
    d.rectangle((0, A - altura, L, A), fill=(0, 0, 0, 185))
    y = A - altura + 12
    for linha in linhas:
        d.text((60, y), linha, font=f, fill=(255, 255, 255))
        y += 36
    for linha in linhas_credito:
        d.text((60, y), linha, font=fc, fill=(170, 176, 188))
        y += 24
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def encaixar(img: np.ndarray) -> np.ndarray:
    """Redimensiona mantendo a proporção, com faixas pretas, para 1280x720."""
    h, w = img.shape[:2]
    s = min(L / w, A / h)
    r = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    tela = np.zeros((A, L, 3), np.uint8)
    y0, x0 = (A - r.shape[0]) // 2, (L - r.shape[1]) // 2
    tela[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return tela


class Montagem:
    def __init__(self, saida: Path):
        self.escritor = EscritorVideo(saida, L, A, FPS)
        self.t = 0.0
        self.roteiro: list[tuple[float, float, str, str]] = []

    def _escrever(self, quadro):
        self.escritor.escrever(quadro)
        self.t += 1 / FPS

    def parada(self, quadro: np.ndarray, segundos: float, titulo: str, narracao: str, fade: float = 0.4) -> None:
        inicio = self.t
        n = round(segundos * FPS)
        nf = round(fade * FPS)
        for i in range(n):
            alfa = min(1.0, (i + 1) / nf, (n - i) / nf) if nf else 1.0
            self._escrever((quadro * alfa).astype(np.uint8))
        self.roteiro.append((inicio, self.t, titulo, narracao))

    def trecho(self, video: Path, inicio_s: float, duracao_s: float, texto: str, credito: str, titulo: str,
               narracao: str) -> bool:
        """Insere o trecho; devolve False se nada foi gravado (vídeo ausente ou curto demais)."""
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"AVISO: vídeo ausente, trecho pulado: {video}", file=sys.stderr)
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(inicio_s * fps))
        t0 = self.t
        passo = fps / FPS
        acumulado = 0.0
        quadro = None
        for _ in range(round(duracao_s * FPS)):
            while acumulado < 1.0:
                ok, q = cap.read()
                if not ok:
                    break
                quadro = q
                acumulado += 1 / passo
            if acumulado < 1.0:  # o vídeo acabou antes da duração pedida: encerra em vez de congelar o último quadro
                break
            acumulado -= 1.0
            self._escrever(legenda(encaixar(quadro), texto, credito))
        cap.release()
        if self.t == t0:
            print(f"AVISO: nenhum quadro lido a partir de {inicio_s} s, trecho pulado: {video}", file=sys.stderr)
            return False
        self.roteiro.append((t0, self.t, titulo, narracao))
        return True

    def fechar(self):
        self.escritor.fechar()


def carregar(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else {}


def painel_foto(img_path: Path, res: dict, faixa: int) -> np.ndarray:
    """Foto anotada à esquerda e os principais campos do JSON à direita.

    `faixa` é a altura da legenda que vai embaixo (altura_legenda): a foto fica inteira acima dela, para
    nenhuma caixa nem a marca ? ficar escondida atrás do texto.
    """
    tela = Image.new("RGB", (L, A), FUNDO)
    foto = cv2.imread(str(img_path))
    h, w = foto.shape[:2]
    livre = A - faixa - 40  # 20 px de margem em cima e 20 px entre a foto e a legenda
    s = min(700 / w, livre / h)
    foto = cv2.resize(foto, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    tela.paste(Image.fromarray(cv2.cvtColor(foto, cv2.COLOR_BGR2RGB)), (20, 20 + (livre - foto.shape[0]) // 2))
    d = ImageDraw.Draw(tela)
    x = 760
    d.text((x, 40), "Resposta da API (contrato JSON v1)", font=fonte(22), fill=CINZA)
    campos = [("diagnostico", res["diagnostico"]), ("severidade", res["severidade"]),
              ("requer_revisao_humana", str(res["requer_revisao_humana"]).lower()),
              ("periodo", f"{PERIODO_PT.get(res['periodo'], res['periodo'])} "
                           f"({FONTE_PERIODO_PT.get(res['periodo_fonte'], res['periodo_fonte'])})"),
              ("acesas / apagadas", f"{res['contagem']['luminaria_acesa']} / {res['contagem']['luminaria_apagada']}"),
              ("pessoas e veículos desfocados", f"{res['privacidade']['pessoas_desfocadas']} e {res['privacidade']['veiculos_desfocados']}")]
    y = 80
    for k, v in campos:
        d.text((x, y), k, font=fonte(18), fill=(150, 156, 168))
        d.text((x, y + 22), v, font=fonte(24), fill=DESTAQUE if k == "diagnostico" else (255, 255, 255))
        y += 62
    f = fonte(19)
    for linha in quebrar(res["acao_sugerida"], f, L - x - 30, d)[:4]:
        d.text((x, y + 6), linha, font=f, fill=(170, 255, 170))
        y += 26
    return cv2.cvtColor(np.asarray(tela), cv2.COLOR_RGB2BGR)


def legenda_foto(res: dict) -> str:
    """Legenda da foto de chamado: diz o que o resultado de fato mostra (confirmação, contradição ou as duas)."""
    apagadas = [l for l in res.get("luminarias", []) if l.get("classe") == "luminaria_apagada"]
    confirmadas = sum(1 for l in apagadas if l.get("concorda_com_modelo") is True)
    contraditas = sum(1 for l in apagadas if l.get("concorda_com_modelo") is False)
    if apagadas and confirmadas == len(apagadas):
        alvo = "a luminária apagada" if confirmadas == 1 else f"as {confirmadas} luminárias apagadas"
        texto = f"Foto real de um chamado de luminária queimada: a medição de brilho confirma {alvo}"
        conf = max(l["confianca"] for l in apagadas)
        if not res.get("requer_revisao_humana"):
            return texto + "; o chamado pode ser aberto sem revisão humana."
        if conf < LIMIARES_PADRAO.conf_confiavel:
            return texto + f", mas a confiança do modelo é de só {porcento(conf)}, por isso o caso vai para revisão humana."
        return texto + "; o JSON diz por que o caso ainda vai para revisão humana."
    if confirmadas and contraditas:
        return (f"Foto real de um chamado de luminária queimada: o modelo vê {len(apagadas)} luminárias apagadas; a "
                f"medição de brilho confirma {confirmadas} e contradiz {contraditas} (marca ?), por isso o caso vai "
                "para revisão humana.")
    if contraditas:
        alvo = ("uma luminária apagada, há um ponto de luz" if contraditas == 1
                else "luminárias apagadas, há pontos de luz")
        return (f"Aqui a medição de brilho contradiz o modelo (marca ?): onde ele vê {alvo}; em vez de abrir o "
                "chamado sozinho, o sistema pede revisão humana.")
    return "Foto real de um chamado de luminária queimada: o JSON diz o que foi visto, a prioridade e se precisa de revisão."


def blocos_resultados(det: dict, img: dict, fps: dict, videos: dict) -> list[tuple[str, int, tuple]]:
    """Números medidos, todos lidos de results/ (nada digitado à mão)."""
    branco = (255, 255, 255)
    blocos = []
    pc = det.get("noite", {}).get("por_classe", {})
    if "luminaria_acesa" in pc:
        # Mesma leitura do relatório: à noite o estado de referência vem da regra de brilho.
        texto = (f"Teste com fotos noturnas nunca vistas no treino ({inteiro(det['noite']['imagens'])} imagens; estado de "
                 f"referência pela regra de brilho): mAP50 {dec(pc['luminaria_acesa']['mAP50'])} nas luminárias acesas")
        if "luminaria_apagada" in pc:
            n_ap = pc["luminaria_apagada"].get("instancias")
            texto += (f" e {dec(pc['luminaria_apagada']['mAP50'])} nas apagadas"
                      + (f" ({inteiro(n_ap)} casos)" if n_ap is not None else ""))
        blocos.append((texto + ".", 23, branco))
    if det.get("todos"):
        blocos.append((f"Teste completo ({inteiro(det['todos']['imagens'])} imagens, noite, anoitecer e dia): "
                       f"mAP50 {dec(det['todos']['mAP50'])}.", 23, branco))
    if img.get("chennai"):
        c = img["chennai"]["sistema"]
        # As fotos de Chennai não têm horário nem local: na avaliação, o período foi informado como noite.
        blocos.append((f"Luminárias com defeito ao anoitecer (Chennai, {inteiro(img['chennai']['imagens'])} fotos sem "
                       f"horário nem local, com o período informado como noite): o sistema acusou "
                       f"{porcento(c['revocacao'])} das apagadas, com precisão de {porcento(c['precisao'])}.", 23, branco))
    if img.get("sf311_noite"):
        s = img["sf311_noite"]
        automatico = ""
        if "chamado_automatico" in s:  # só conta chamado automático de luminária apagada
            automatico = (" (nenhum chamado automático de luminária apagada)" if not s["chamado_automatico"] else
                          f" ({porcento(s['chamado_automatico'])} com chamado automático de luminária apagada, "
                          "sem revisão humana)")
        blocos.append((f"Chamados reais de luminária queimada (San Francisco, {inteiro(s['imagens'])} fotos): o sistema "
                       f"apontou a falha em {porcento(s['acusou_falha_sistema'])}{automatico}.", 23, branco))
    if videos:
        acesas = sum(r.get("luminarias_unicas", {}).get("luminaria_acesa", 0) for r in videos.values())
        apagadas = [a for r in videos.values() for a in r.get("luminarias_apagadas", [])]
        # Sem gabarito nos vídeos: o que se conta são trilhas do rastreador, e faróis de carros também viram trilha.
        texto = (f"Vídeos noturnos ({plural(len(videos), 'trajeto', 'trajetos')}, sem gabarito): o rastreador formou "
                 f"{plural(acesas, 'trilha classificada como acesa', 'trilhas classificadas como acesas')} (incluem "
                 f"faróis de carros) e {plural(len(apagadas), 'apagada', 'apagadas')}")
        if apagadas and all("requer_revisao_humana" in a for a in apagadas):
            n = sum(1 for a in apagadas if a["requer_revisao_humana"])
            if len(apagadas) == 1:
                texto += "; ela vai para revisão humana" if n else "; ela não precisou de revisão humana"
            elif n == len(apagadas):
                texto += "; ambas vão para revisão humana" if n == 2 else "; todas vão para revisão humana"
            else:
                texto += (f"; {inteiro(n)} {'delas vai' if n == 1 else 'delas vão'} para revisão humana" if n
                          else "; nenhuma delas precisou de revisão humana")
        blocos.append((texto + ".", 23, branco))
    if fps.get("fps"):
        f = fps["fps"]
        cpu = f"numa CPU de desktop com {fps['threads']} threads" if fps.get("threads") else "em CPU"
        texto = (f"Velocidade {cpu}: o detector sozinho roda a {dec(f['detector_pytorch'], 1)} quadros/s em PyTorch "
                 f"e a {dec(f['detector_onnx'], 1)} em ONNX")
        # Nos vídeos, o pipeline completo em PyTorch, o mais rápido de ponta a ponta nesta CPU (ver benchmark_fps.py).
        medidos = [r for r in videos.values() if r.get("fps_processamento_medio") and r.get("fps_video")]
        if medidos:
            vel = [r["fps_processamento_medio"] for r in medidos]
            fracao = [r["fps_processamento_medio"] / r["fps_video"] for r in medidos]
            ritmos = {r["fps_video"] for r in medidos}
            texto += (f"; o pipeline completo, com rastreamento, brilho, privacidade e diagnóstico, rodou nos vídeos "
                      f"entre {dec(min(vel), 1)} e {dec(max(vel), 1)} quadros/s, de {porcento(min(fracao))} a "
                      f"{porcento(max(fracao))} do ritmo deles"
                      + (f" ({dec(ritmos.pop(), 2)} quadros/s)" if len(ritmos) == 1 else ""))
        blocos.append((texto + ".", 23, DESTAQUE))
    return blocos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=Path("data/outputs/videos"))
    ap.add_argument("--resultados", type=Path, default=Path("results"))
    ap.add_argument("--manifesto-videos", type=Path, default=Path("data/manifests/videos.csv"))
    ap.add_argument("--saida", type=Path, default=Path("docs/video/luz_segura_demo.mp4"))
    ap.add_argument("--capturas", type=Path, default=Path("docs/video/capturas"))
    args = ap.parse_args()

    det = carregar(args.resultados / "avaliacao" / "deteccao.json")
    img = carregar(args.resultados / "avaliacao" / "imagens.json")
    fps = carregar(args.resultados / "avaliacao" / "fps.json")
    modelo = carregar(args.resultados / "modelo_final.json")
    videos = {p.stem: carregar(p) for p in sorted((args.resultados / "videos").glob("*.json"))}
    with args.manifesto_videos.open(encoding="utf-8", newline="") as f:
        fontes = {r["id"]: r for r in csv.DictReader(f)}

    def credito(vid: str) -> str:
        # Trecho na tela sem autor e licença descumpre a licença: melhor parar do que publicar assim.
        if vid not in fontes:
            raise SystemExit(f"sem crédito em {args.manifesto_videos}: {vid}")
        v = fontes[vid]
        return f"Vídeo: {v['autor']} · {v['licenca']} · {v['local']} · adaptado: {ADAPTACAO}"

    # Janelas conferidas quadro a quadro nos vídeos anotados: no 00035 a luminária central fica marcada o tempo
    # todo; no 00024, entre 25,5 e 33,5 s, não há nenhuma caixa e o painel fica em trecho escuro.
    r23, r24 = videos.get("sao_paulo_00023", {}), videos.get("sao_paulo_00024", {})
    velocidade = (f": a análise completa roda a {dec(r23['fps_processamento_medio'], 1)} quadros/s, cerca de "
                  f"{porcento(r23['fps_processamento_medio'] / r23['fps_video'])} do ritmo do vídeo"
                  if r23.get("fps_processamento_medio") and r23.get("fps_video") else "")
    escuro = (f"; no trajeto inteiro, foram {dec(r24['segundos_em_trecho_escuro'], 1)} de {dec(r24['duracao_s'], 1)} s "
              "assim" if r24.get("segundos_em_trecho_escuro") is not None and r24.get("duracao_s") else "")
    trechos = [
        ("sao_paulo_00035", 21, 9, "Calçada da Avenida Paulista à noite: a luminária acesa fica marcada, e os veículos "
         "detectados pelo filtro são desfocados antes da exibição; à noite, alguns ainda escapam do filtro."),
        ("sao_paulo_00056", 0, 8, "Câmera parada: a luminária acesa no alto do poste fica marcada em todos os quadros."),
        ("sao_paulo_00023", 9, 10, f"Vista do alto de uma via, só com CPU{velocidade}; faróis de carros às vezes ainda "
         "são marcados como luminária acesa."),
        ("sao_paulo_00038", 2, 8, "Praça com uma única luminária, distante e pequena demais para o detector: sem "
         "luminária acesa detectada e com o chão escuro, o sistema marca trecho escuro e pede revisão humana."),
        ("sao_paulo_00024", 25.5, 8, "Via mal iluminada: sem luminária acesa à vista e com o chão escuro, o painel marca "
         f"trecho escuro{escuro}."),
    ]
    # Créditos conferidos antes de gravar o primeiro quadro: um trecho disponível sem linha no manifesto para tudo.
    creditos = {vid: credito(vid) for vid, *_ in trechos if (args.videos / f"{vid}_anotado.mp4").exists()}

    m = Montagem(args.saida)
    m.parada(cartela("Luz Segura", [
        ("Visão computacional para encontrar luminárias públicas apagadas e trechos escuros", 30, DESTAQUE),
        ("FIAP · Global Solution 2026 · Applied Computer Vision · Enxergando o mundo real", 22, CINZA),
        ("  ·  ".join(EQUIPE[:3]), 18, CINZA), ("  ·  ".join(EQUIPE[3:]), 18, CINZA)]), 6,
        "Abertura", "Somos a equipe do Luz Segura. Mostramos o componente de visão computacional do nosso MVP.")
    m.parada(cartela("O problema", [
        ("Ponto de ônibus escuro, poste queimado, praça sem luz: quem volta a pé à noite fica mais exposto a quedas, "
         "atropelamentos e insegurança.", 28, (255, 255, 255)),
        ("Hoje a prefeitura só descobre a falha quando alguém reclama, e a reclamação chega mal localizada, "
         "duplicada e sem prioridade.", 28, CINZA),
        ("O Luz Segura transforma uma foto ou um vídeo noturno em um diagnóstico explicável, pronto para virar chamado.",
         28, DESTAQUE)]), 16,
        "Problema", "Iluminação pública que falha deixa o caminho a pé mais perigoso, e a falha demora a ser descoberta.")
    m.parada(cartela("Como funciona", [
        ("1. Detector YOLO26n treinado por nós encontra cada luminária e diz se está acesa ou apagada.", 26, (255, 255, 255)),
        ("2. Uma regra fixa de brilho confere cada luminária; se discordar do modelo ou não confirmar a falha, "
         "o caso vai para revisão humana.", 26, (255, 255, 255)),
        ("3. O brilho do chão e a posição do Sol completam o diagnóstico: luminária apagada, iluminação insuficiente "
         "(acesa, mas chão escuro), trecho escuro ou iluminação adequada.", 26, (255, 255, 255)),
        ("4. Pessoas e veículos detectados pelo filtro são desfocados, e o sistema não guarda a foto: só o JSON segue "
         "para o MVP.", 26, DESTAQUE)]), 20,
        "Como funciona", "O detector encontra as luminárias, a medição de brilho confere, e o diagnóstico vira um JSON para o MVP.")

    usados = []
    for vid, ini, dur, texto in trechos:
        video = args.videos / f"{vid}_anotado.mp4"
        if vid not in creditos:
            print(f"AVISO: vídeo ausente, trecho pulado: {video}", file=sys.stderr)
            continue
        if m.trecho(video, ini, dur, texto, creditos[vid], f"Vídeo: {vid}", texto):
            usados.append(vid)
    faltando = [t[0] for t in trechos if t[0] not in usados]

    for nome, segundos, texto, titulo in CAPTURAS:
        p = args.capturas / nome
        if not p.exists():
            print(f"AVISO: captura ausente, pulada: {p}", file=sys.stderr)
            continue
        captura = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        m.parada(legenda(encaixar(captura), texto, CREDITO_SF311 if nome.startswith("app") else None),
                 segundos, titulo, texto)

    exemplos = sorted((args.resultados / "exemplos").glob("sf311_*.json"))
    for ex in [e for e in exemplos if "_dia_" not in e.name][:3]:
        res = json.loads(ex.read_text(encoding="utf-8"))
        texto = legenda_foto(res)
        painel = painel_foto(ex.with_suffix(".jpg"), res, altura_legenda(texto, CREDITO_SF311))
        m.parada(legenda(painel, texto, CREDITO_SF311), 6, f"Foto {ex.stem}", texto)

    blocos = blocos_resultados(det, img, fps, videos)
    if blocos:
        m.parada(cartela("Resultados medidos", blocos), 28, "Resultados",
                 "Os números vêm de conjuntos que o modelo nunca viu no treino.")
    m.parada(cartela("Privacidade e IA responsável", [
        ("Nada é guardado: a API não grava a foto em disco; o app só a mantém no cache temporário, apagado em até "
         "20 minutos ou ao fechar, e o último vídeo anotado, já anonimizado, fica numa pasta temporária até a "
         "próxima análise de vídeo ou até o app fechar.", 24, (255, 255, 255)),
        ("Pessoas e veículos detectados pelo filtro são desfocados antes da exibição; à noite ou em ângulos incomuns, "
         "algum pode escapar do filtro.", 24, (255, 255, 255)),
        ("Cada decisão vem com uma explicação em linguagem simples e com o grau de confiança.", 24, (255, 255, 255)),
        ("Quando o modelo e a medição de brilho discordam, ou a falha não se confirma, uma pessoa revisa antes de abrir "
         "o chamado; sem nenhuma luminária na foto noturna, revisa antes de encerrar o relato.", 24, (255, 255, 255)),
        ("Limitação medida: em fotos noturnas de celular, muitas luminárias apagadas ainda escapam ao detector; se ele "
         "acha só luminárias acesas e o chão está claro, a falha passa sem revisão.", 22, CINZA)]), 20,
        "Privacidade", "O sistema não guarda as fotos, pessoas e veículos detectados são desfocados (à noite, algum "
        "pode escapar do filtro), e os casos duvidosos que ele aponta passam por revisão humana.")

    # Créditos só dos trechos que entraram no vídeo; o compartilhamento pela mesma licença só vale se houver trecho BY-SA.
    usados_fontes = [fontes[v] for v in usados]
    creditos_videos = list(dict.fromkeys((v["autor"], v["licenca"], origem(v["pagina"])) for v in usados_fontes))
    sa = sorted({v["licenca"] for v in usados_fontes if "BY-SA" in v["licenca"]})
    v2 = modelo.get("versao") == "v2"
    blocos_creditos = [
        ("Treino: Open Images V7 (Google; anotações CC BY 4.0, fotos CC BY 2.0 de seus autores no Flickr)"
         + ("; autotreinamento com fotos de outros chamados do San Francisco 311 (DataSF, ODC PDDL)." if v2 else "."), 21, CINZA),
        ("Teste: Comprehensive Dataset for Urban Streetlight Analysis (arXiv:2407.01117; só nas métricas, nenhuma foto exibida); "
         "San Francisco 311 (DataSF, ODC PDDL).", 21, CINZA)]
    if creditos_videos:
        blocos_creditos.append(("Vídeos: " + "; ".join(f"{a}, {l}, via {o}" for a, l, o in creditos_videos)
                                + f"; todos adaptados ({ADAPTACAO}).", 21, CINZA))
    if sa:
        blocos_creditos.append((f"Este vídeo contém trechos sob {', '.join(sa)} e é distribuído sob a mesma licença: "
                                f"{url_licenca(sa[0])}", 21, DESTAQUE))
    blocos_creditos.append(("Equipe: " + "; ".join(EQUIPE), 19, (255, 255, 255)))
    m.parada(cartela("Créditos das imagens e vídeos", blocos_creditos), 14,
             "Créditos", "Créditos das fontes de dados e dos vídeos usados, com as licenças de cada um.")
    m.fechar()

    linhas = ["# Roteiro do vídeo de demonstração", "",
              "Narração opcional: o vídeo já tem legendas. Se a equipe quiser gravar voz, este é o texto de cada trecho.", "",
              "| Início | Fim | Trecho | Narração sugerida |", "|---|---|---|---|"]
    for ini, fim, titulo, narracao in m.roteiro:
        linhas.append(f"| {int(ini // 60)}:{ini % 60:04.1f} | {int(fim // 60)}:{fim % 60:04.1f} | {titulo} | {narracao} |")
    linhas.append("")
    linhas.append(f"Duração total: {m.t:.0f} s.")
    if faltando:
        linhas.append(f"Trechos ausentes (sem vídeo anotado, fora da demo): {', '.join(faltando)}.")
    linhas += ["", "## Créditos (descrição do YouTube)", "",
               "Luz Segura: visão computacional para encontrar luminárias públicas apagadas e trechos escuros "
               "(FIAP Global Solution 2026, Applied Computer Vision).",
               "Equipe: " + "; ".join(EQUIPE) + ".", ""]
    if usados_fontes:
        # Uma linha por obra de origem (vários trechos podem vir da mesma página).
        obras: dict[tuple, list[str]] = {}
        for v in usados_fontes:
            obras.setdefault((v["local"], v["autor"], v["pagina"], v["licenca"]), []).append(v["id"])
        linhas.append(f"Vídeos de terceiros (todos modificados: {MODIFICACOES}):")
        for (local, autor, pagina, licenca), ids in obras.items():
            url = url_licenca(licenca)
            linhas.append(f"- {local}: {autor}, {pagina}. Licença: {licenca}" + (f", {url}" if url else "")
                          + f". Trechos: {', '.join(ids)}.")
        linhas.append("")
    if sa:
        linhas += [f"Este vídeo contém trechos sob {', '.join(sa)} e é distribuído sob a mesma licença: {url_licenca(sa[0])}", ""]
    linhas += ["Fotos e dados:",
               "- Fotos de chamados: San Francisco 311 (DataSF), domínio público (ODC PDDL), "
               "https://data.sf.gov/resource/vw6y-z8j6. Modificadas: anotadas, com pessoas e veículos detectados "
               "desfocados.",
               "- Treino: Open Images V7 (Google), anotações CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) e fotos "
               "CC BY 2.0 de seus autores no Flickr (https://creativecommons.org/licenses/by/2.0/), "
               "https://storage.googleapis.com/openimages/web/index.html."]
    if v2:
        linhas.append("- Autotreinamento do modelo final: fotos de outros chamados do San Francisco 311 (DataSF, ODC PDDL), "
                      "separados dos usados na avaliação.")
    linhas.append("- Teste: Comprehensive Dataset for Urban Streetlight Analysis (arXiv:2407.01117), "
                  "https://github.com/Team16Project/Street-Light-Dataset (só nas métricas; nenhuma foto aparece no vídeo).")
    (args.saida.parent / "roteiro.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"vídeo: {args.saida} ({m.t:.0f} s)")
    if faltando:
        print(f"AVISO: {len(faltando)} trecho(s) sem vídeo e fora da demo: {', '.join(faltando)}", file=sys.stderr)
    return 0 if usados else 1


if __name__ == "__main__":
    sys.exit(main())
