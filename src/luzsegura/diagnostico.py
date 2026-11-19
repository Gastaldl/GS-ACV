"""Diagnóstico explicável: transforma detecções e medidas em informação (contrato JSON v1).

Dois sinais complementares decidem o estado de cada luminária:
  1. a classe prevista pelo detector treinado (acesa ou apagada);
  2. a regra de brilho medida na própria caixa (luzsegura.luminancia).
À noite e no anoitecer, os rótulos de treino do detector vieram dessa mesma regra; por isso a
conferência pega sobretudo contradições, como "apagada" prevista onde há um ponto de luz.

O resultado pede revisão humana quando o brilho contradiz o modelo ou não confirma a luminária
apagada, quando a confiança é baixa e sempre que o diagnóstico vem só do brilho do chão: chão escuro
(iluminação insuficiente, trecho escuro) ou, à noite e na transição, iluminação adequada sem nenhuma
luminária encontrada na foto. Diagnóstico de falha na transição, ou com o período só estimado pelo
brilho da imagem, também vai para revisão. De dia, a luminária acesa só dispensa revisão com
confiança alta, mais clara que o entorno e com o Sol já alto. Luminária apagada só vira chamado
automático quando confirmada pelo brilho, à noite e com período conhecido: o sistema nunca abre um
chamado "no escuro" sem dizer o quanto confia nele.

Limitação: iluminação adequada com alguma luminária acesa encontrada, sem contradição do brilho, não
passa por revisão. Se a luminária apagada do relato não foi detectada, o sistema não tem como notar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .config import CLASSE_ACESA, CLASSE_APAGADA, LIMIARES_PADRAO, NOME_MODELO, VERSAO_CONTRATO, Limiares
from .luminancia import ACESA, APAGADA, INDEFINIDO, MedidaCaixa, MedidaCena
from .solar import altura_sol, periodo_por_sol

ADEQUADA = "iluminacao_adequada"
LUMINARIA_APAGADA = "luminaria_apagada"
INSUFICIENTE = "iluminacao_insuficiente"
TRECHO_ESCURO = "trecho_escuro"
FORA_DO_PERIODO = "fora_do_periodo_noturno"
ACESA_DE_DIA = "acesa_durante_o_dia"

NOME_PT = {
    ADEQUADA: "iluminação adequada",
    LUMINARIA_APAGADA: "luminária apagada",
    INSUFICIENTE: "iluminação insuficiente",
    TRECHO_ESCURO: "trecho escuro",
    FORA_DO_PERIODO: "fora do período noturno",
    ACESA_DE_DIA: "luminária acesa durante o dia",
}

ACAO = {
    ADEQUADA: "Nenhuma ação necessária.",
    LUMINARIA_APAGADA: "Abrir chamado de manutenção para a luminária apagada.",
    INSUFICIENTE: "Vistoriar o trecho: luminária acesa, mas o chão continua escuro (árvore, distância entre postes ou potência baixa).",
    TRECHO_ESCURO: "Avaliar a falta de iluminação: nenhuma luminária acesa e o trecho está escuro.",
    FORA_DO_PERIODO: "Foto diurna: envie uma foto depois do pôr do sol para avaliar a iluminação.",
    ACESA_DE_DIA: "Verificar a fotocélula: luminária acesa durante o dia gasta energia à toa.",
}


def _acao(diag: str, revisao: bool) -> str:
    """Ação sugerida. Com revisão humana, a ação vira o passo seguinte à confirmação do técnico."""
    if not revisao or diag == FORA_DO_PERIODO:
        return ACAO[diag]
    if diag == ADEQUADA:
        return "Enviar para revisão humana antes de encerrar o relato."
    acao = ACAO[diag]
    return f"Enviar para revisão humana; se confirmado, {acao[0].lower()}{acao[1:]}"


@dataclass
class Luminaria:
    """Uma detecção do modelo, com a medida de brilho da caixa."""

    classe: str
    confianca: float
    xyxy: tuple[float, float, float, float]
    medida: MedidaCaixa
    id_rastreio: int | None = None

    @property
    def estado_modelo(self) -> str:
        return ACESA if self.classe == CLASSE_ACESA else APAGADA

    @property
    def concorda_brilho(self) -> bool | None:
        if self.medida.estado == INDEFINIDO:
            return None
        return self.medida.estado == self.estado_modelo


@dataclass
class Contexto:
    """Metadados opcionais da captura."""

    origem: str = "foto"
    capturado_em: datetime | None = None
    lat: float | None = None
    lon: float | None = None
    periodo_informado: str | None = None  # noite | transicao | dia (ex.: vídeo sabidamente noturno)
    extras: dict = field(default_factory=dict)


def definir_periodo(cena: MedidaCena, ctx: Contexto, lim: Limiares) -> tuple[str, str, float | None]:
    """Devolve (período, fonte da decisão, altura do Sol)."""
    if ctx.periodo_informado in ("noite", "transicao", "dia"):
        return ctx.periodo_informado, "informado", None
    if ctx.capturado_em is not None and ctx.lat is not None and ctx.lon is not None:
        altura = altura_sol(ctx.capturado_em, ctx.lat, ctx.lon)
        return periodo_por_sol(altura), "posicao_do_sol", round(altura, 1)
    return ("dia" if cena.media >= lim.cena_diurna else "noite"), "luminosidade", None


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _explicacao(diag: str, sev: str, acesas: list, apagadas: list, cena: MedidaCena, periodo: str,
                fonte: str, altura: float | None, motivos: list[str], lim: Limiares, revisao: bool = False) -> str:
    partes = []
    if acesas or apagadas:
        itens = []
        if acesas:
            itens.append(_plural(len(acesas), "luminária acesa", "luminárias acesas"))
        if apagadas:
            melhor = max(l.confianca for l in apagadas)
            nomes = ("apagada", "apagadas") if acesas else ("luminária apagada", "luminárias apagadas")
            itens.append(f"{_plural(len(apagadas), *nomes)} (confiança até {melhor:.0%})")
        partes.append("Encontrei " + " e ".join(itens) + ".")
    else:
        partes.append("Não encontrei luminárias na imagem.")
    nome_periodo = "transição" if periodo == "transicao" else periodo  # o código do contrato não tem acento
    if fonte == "posicao_do_sol":
        partes.append(f"Período: {nome_periodo} (Sol a {altura:+.0f}° do horizonte no horário da captura).")
    elif fonte == "luminosidade":
        partes.append(f"Período: {nome_periodo} (estimado pelo brilho da imagem; informe o horário para mais precisão).")
    else:
        partes.append(f"Período: {nome_periodo} (informado).")
    if periodo != "dia":
        situacao = "abaixo" if cena.piso < lim.piso_escuro else "acima"
        partes.append(f"Brilho do chão: {cena.piso:.0f} de 255, {situacao} do mínimo de {lim.piso_escuro:.0f} usado pelo sistema.")
    if diag == LUMINARIA_APAGADA:
        confirmadas = sum(1 for l in apagadas if l.concorda_brilho is True)
        if confirmadas:
            partes.append(f"A medição de brilho confirma {_plural(confirmadas, 'luminária apagada', 'luminárias apagadas')} (nenhum ponto de luz na área).")
    for m in motivos:
        partes.append(f"Atenção: {m}.")
    rotulo = {"nenhuma": "", "baixa": " (prioridade baixa)", "media": " (prioridade média)", "alta": " (prioridade alta)"}[sev]
    partes.append(f"Resultado: {NOME_PT[diag]}{rotulo}. {_acao(diag, revisao)}")
    return " ".join(partes)


def diagnosticar(luminarias: list[Luminaria], cena: MedidaCena, largura: int, altura: int,
                 ctx: Contexto | None = None, lim: Limiares = LIMIARES_PADRAO,
                 tempo_ms: float | None = None, privacidade: dict | None = None) -> dict:
    """Gera o resultado no contrato JSON v1 (ver docs/contrato_api.md)."""
    ctx = ctx or Contexto()
    validas = [l for l in luminarias if l.confianca >= lim.conf_minima]
    acesas = [l for l in validas if l.classe == CLASSE_ACESA]
    apagadas = [l for l in validas if l.classe == CLASSE_APAGADA]
    periodo, fonte, altura_do_sol = definir_periodo(cena, ctx, lim)
    piso_escuro = cena.piso < lim.piso_escuro
    piso_muito_escuro = cena.piso < lim.piso_muito_escuro
    motivos: list[str] = []
    revisao = False

    if periodo == "dia":
        acesas_confirmadas = [l for l in acesas if l.concorda_brilho is True]
        if acesas_confirmadas and fonte != "luminosidade":
            diag, sev = ACESA_DE_DIA, "baixa"
            confiaveis = [l for l in acesas_confirmadas if l.confianca >= lim.conf_confiavel]
            if not confiaveis:
                revisao = True
                motivos.append(f"confiança do modelo abaixo de {lim.conf_confiavel:.0%}")
            # De dia o céu estourado atrás da luminária também satura a caixa: sem ser mais clara que o
            # entorno, ela não prova lâmpada acesa. Vale para as confiáveis (ou para todas, se nenhuma for).
            if all(l.medida.contraste <= 1.0 for l in (confiaveis or acesas_confirmadas)):
                revisao = True
                motivos.append("a caixa não é mais clara que o entorno, então o ponto de luz pode ser o céu "
                               "claro atrás da luminária")
            # Com o Sol baixo ou céu fechado a fotocélula acende legitimamente; período só informado
            # não traz a altura do Sol para conferir.
            if altura_do_sol is None:
                revisao = True
                motivos.append("com o período apenas informado, não dá para conferir se o Sol já estava alto")
            elif altura_do_sol < lim.sol_min_acesa_de_dia:
                revisao = True
                motivos.append(f"com o Sol a menos de {lim.sol_min_acesa_de_dia:.0f}° do horizonte, principalmente "
                               "com céu fechado, a fotocélula pode estar acesa legitimamente")
        else:
            diag, sev = FORA_DO_PERIODO, "nenhuma"
    elif apagadas:
        diag = LUMINARIA_APAGADA
        sev = "alta" if piso_escuro else "media"
        conflitos = [l for l in apagadas if l.concorda_brilho is False]
        if conflitos:
            revisao = True
            motivos.append(f"o brilho medido contradiz o modelo em {_plural(len(conflitos), 'luminária', 'luminárias')}; um técnico deve revisar")
        # "Indefinido" (reflexo, brilho difuso) é justamente como aparece uma lâmpada acesa atrás de um difusor.
        inconclusivas = [l for l in apagadas if l.concorda_brilho is None]
        if inconclusivas:
            revisao = True
            motivos.append(f"a medição de brilho não confirma {_plural(len(inconclusivas), 'luminária apagada', 'luminárias apagadas')} "
                           "(pequeno reflexo ou brilho difuso na área)")
        if all(l.confianca < lim.conf_confiavel for l in apagadas):
            revisao = True
            motivos.append(f"confiança do modelo abaixo de {lim.conf_confiavel:.0%}")
    elif piso_escuro:
        diag = INSUFICIENTE if acesas else TRECHO_ESCURO
        # Com luminária acesa na foto, chão escuro é indício fraco: no máximo prioridade média.
        sev = "alta" if piso_muito_escuro and not acesas else "media"
        # O "chão" é só a faixa de baixo da foto; numa foto apontada para cima, ela mostra céu ou fachada.
        revisao = True
        motivos.append(f"o brilho do chão é medido na faixa inferior da foto ({lim.fracao_piso:.0%}); se a foto "
                       "foi tirada apontando para cima, essa faixa pode mostrar céu, fachada ou o próprio poste")
    else:
        diag, sev = ADEQUADA, "nenhuma"
        # A foto do relato deveria mostrar a luminária. Sem nenhuma detecção válida, "adequada" vem só da
        # faixa de baixo da foto, e uma luminária apagada que o detector não viu encerraria o relato.
        if not validas:
            revisao = True
            motivos.append("nenhuma luminária foi encontrada na foto: o resultado vem só do brilho do chão")

    # O modelo também erra para o outro lado: luminária apagada vista como acesa. Com luminária
    # apagada já no diagnóstico, essa contradição só reforça o chamado.
    if periodo != "dia" and diag != LUMINARIA_APAGADA:
        sem_luz = [l for l in acesas if l.concorda_brilho is False]
        if sem_luz:
            revisao = True
            motivos.append("o brilho medido não encontrou ponto de luz em "
                           f"{_plural(len(sem_luz), 'luminária classificada como acesa', 'luminárias classificadas como acesas')}; "
                           "um técnico deve revisar")
    # Sem horário e local, "noite" é só estimativa: muita foto diurna com exposição automática fica
    # abaixo de cena_diurna, e a luminária apagada de dia viraria chamado.
    if fonte == "luminosidade" and diag in (LUMINARIA_APAGADA, INSUFICIENTE, TRECHO_ESCURO):
        revisao = True
        motivos.append("sem o horário e o local da captura, não dá para confirmar que era noite")
    if periodo == "transicao" and diag != ADEQUADA:
        revisao = True
        motivos.append("no horário de transição, as luminárias podem ainda estar acendendo e a luz do céu "
                       "altera o brilho do chão")

    def item(l: Luminaria) -> dict:
        x1, y1, x2, y2 = l.xyxy
        return {
            "classe": l.classe,
            "confianca": round(float(l.confianca), 3),
            "caixa_norm": [round(x1 / largura, 4), round(y1 / altura, 4), round(x2 / largura, 4), round(y2 / altura, 4)],
            "estado_por_brilho": l.medida.estado,
            "concorda_com_modelo": l.concorda_brilho,
            "maior_ponto_de_luz_px": l.medida.maior_mancha,
            "contraste_com_entorno": round(l.medida.contraste, 2),
            "id_rastreio": l.id_rastreio,
        }

    localizacao = {"lat": ctx.lat, "lon": ctx.lon} if ctx.lat is not None and ctx.lon is not None else None
    return {
        "versao_contrato": VERSAO_CONTRATO,
        "modelo": NOME_MODELO,
        "origem": ctx.origem,
        "periodo": periodo,
        "periodo_fonte": fonte,
        "altura_do_sol_graus": altura_do_sol,
        "diagnostico": diag,
        "severidade": sev,
        "requer_revisao_humana": revisao,
        "acao_sugerida": _acao(diag, revisao),
        "explicacao": _explicacao(diag, sev, acesas, apagadas, cena, periodo, fonte, altura_do_sol, motivos, lim, revisao),
        "contagem": {CLASSE_ACESA: len(acesas), CLASSE_APAGADA: len(apagadas)},
        "luminosidade": cena.como_dict(),
        "luminarias": [item(l) for l in sorted(validas, key=lambda l: -l.confianca)],
        "privacidade": privacidade or {"pessoas_desfocadas": 0, "veiculos_desfocados": 0, "imagem_armazenada": False},
        "imagem": {"largura": largura, "altura": altura},
        "tempo_processamento_ms": round(tempo_ms, 1) if tempo_ms is not None else None,
        "localizacao": localizacao,
        "capturado_em": ctx.capturado_em.isoformat() if ctx.capturado_em else None,
    }
