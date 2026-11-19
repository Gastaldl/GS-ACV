"""Gera exemplos anotados (já anonimizados) para o relatório, o README e o vídeo.

Usa só imagens de licença aberta: chamados do SF 311 (domínio público) e o teste reservado do
Open Images (CC BY 2.0, com crédito). As fotos de Chennai não têm licença declarada
e por isso não são republicadas; entram apenas nas métricas.

Os exemplos são uma curadoria fixa (EXEMPLOS_SF311, EXEMPLOS_SF311_DIA e EXEMPLOS_OPENIMAGES), cada um
com uma nota de uma frase sobre o que a figura mostra. Os chamados do SF 311 precisam estar entre os
avaliados em results/avaliacao (resultados sem sufixo, os do modelo escolhido por
scripts/avaliacao/comparar_modelos.py), e as fotos do Open Images, no teste reservado; os pesos são os
finais em models/. Exemplo da curadoria sem foto ou fora da avaliação interrompe o script antes de
carregar o modelo. Se o resultado de um exemplo deixar de ser o que a nota descreve (outros pesos ou
outra regra), o script avisa e termina com código 1. Fotos e JSONs de exemplos que saíram da curadoria
são apagados de results/exemplos, para não ficar foto publicada sem crédito em creditos.csv.

Saídas: results/exemplos/<nome>.jpg, results/exemplos/<nome>.json e results/exemplos/creditos.csv
(com título da obra e a modificação feita, como as licenças CC BY pedem, e a nota de cada exemplo)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import cv2

from luzsegura.diagnostico import Contexto
from luzsegura.pipeline import criar_analisador

SF = ZoneInfo("America/Los_Angeles")


class Exemplo(NamedTuple):
    """Exemplo da curadoria: o resultado que a nota descreve e a nota (uma frase, para a legenda da figura)."""

    diagnostico: str
    revisao_humana: bool
    acesas: int
    apagadas: int
    contraditas: int  # luminárias em que a medição de brilho contradiz o modelo (marca ?)
    nota: str


# Exemplos escolhidos por inspeção visual entre os resultados do modelo final. As métricas do relatório
# usam todos os chamados e todo o teste; estas listas só definem as ilustrações.
# Chamados noturnos de luminária queimada, na ordem das figuras (o vídeo mostra os três primeiros).
EXEMPLOS_SF311: dict[str, Exemplo] = {
    "202000314168": Exemplo("luminaria_apagada", True, 0, 4, 1,
                            "Chamado de luminária queimada: o modelo vê quatro luminárias apagadas, a medição de "
                            "brilho confirma três e contradiz uma (marca ?), e por isso o caso vai para revisão "
                            "humana."),
    "101003779202": Exemplo("luminaria_apagada", True, 0, 1, 0,
                            "Chamado de luminária queimada: a medição de brilho confirma a luminária apagada, mas a "
                            "confiança do modelo fica abaixo de 55%, e por isso o caso vai para revisão humana."),
    "101003656777": Exemplo("luminaria_apagada", True, 1, 1, 1,
                            "Revisão por contradição: a caixa apagada (marca ?) está numa luminária acesa distante, e "
                            "a luminária apagada do poste central não foi detectada."),
    "202000318762": Exemplo("iluminacao_adequada", True, 0, 0, 0,
                            "Falha não acusada: o globo apagado no alto do poste não foi detectado, e a iluminação "
                            "adequada, que vem só do brilho da faixa de baixo da foto (fachadas iluminadas), vai para "
                            "revisão humana em vez de encerrar o relato."),
}
# Chamados de luz acesa o tempo todo, com foto tirada de dia.
EXEMPLOS_SF311_DIA: dict[str, Exemplo] = {
    "101002403484": Exemplo("fora_do_periodo_noturno", False, 0, 1, 1,
                            "Chamado de luz acesa o tempo todo: a luminária está acesa em pleno dia, mas o modelo a vê "
                            "como apagada, porque no treino toda luminária de dia foi rotulada apagada, e o sistema "
                            "responde foto diurna."),
    "101002453966": Exemplo("fora_do_periodo_noturno", False, 0, 2, 2,
                            "Chamado de luz acesa o tempo todo: o LED aparece aceso contra o céu azul, mas o modelo "
                            "marca duas caixas grandes como apagadas, com confiança baixa, e o sistema responde foto "
                            "diurna em vez de luminária acesa durante o dia."),
}
# Teste reservado do Open Images (grupo noturno), com o período informado como noite.
EXEMPLOS_OPENIMAGES: dict[str, Exemplo] = {
    "0006ff9c44e561a7": Exemplo("iluminacao_adequada", False, 1, 0, 0,
                                "Foto noturna do teste reservado, com o período informado como noite: das 4 "
                                "luminárias acesas da referência, o detector acha uma, e o chão iluminado leva a "
                                "iluminação adequada, sem revisão."),
    "00250fb6c81181d6": Exemplo("iluminacao_insuficiente", True, 1, 0, 0,
                                "Foto noturna do teste reservado, com o período informado como noite: o detector "
                                "acha uma das 2 luminárias acesas da referência, mas a faixa de baixo da foto é água "
                                "escura, e a iluminação insuficiente vai para revisão humana."),
}
# Toda foto publicada é uma adaptação: a atribuição CC BY exige indicar o que mudou.
MODIFICACAO = "anotado; desfoque das pessoas e veículos detectados"


def resumo(res: dict) -> tuple:
    """O que a nota de um exemplo descreve, na ordem dos campos de Exemplo."""
    contraditas = sum(1 for l in res["luminarias"] if l["concorda_com_modelo"] is False)
    return (res["diagnostico"], res["requer_revisao_humana"], res["contagem"]["luminaria_acesa"],
            res["contagem"]["luminaria_apagada"], contraditas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", type=Path, default=Path("models/luzsegura_yolo26n.pt"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    ap.add_argument("--manifestos", type=Path, default=Path("data/manifests"))
    ap.add_argument("--avaliacao", type=Path, default=Path("results/avaliacao"))
    ap.add_argument("--saida", type=Path, default=Path("results/exemplos"))
    args = ap.parse_args()

    # Entradas conferidas antes de carregar o modelo: (nome, foto, contexto, crédito, exemplo da curadoria).
    entradas, faltando = [], []
    # SF 311: só chamados que entraram na avaliação (results/avaliacao).
    sf = {r["service_request_id"]: r for r in csv.DictReader(open(args.manifestos / "sf311.csv", encoding="utf-8"))}
    avaliados = {l["arquivo"] for nome in ("imagens_sf311_noite.csv", "imagens_sf311_dia.csv")
                 for l in csv.DictReader(open(args.avaliacao / nome, encoding="utf-8"))}
    # O número de ordem no nome mantém a sequência da curadoria (o vídeo mostra os primeiros).
    chamados = [(f"sf311_{ordem}_{i}", i, ex) for ordem, (i, ex) in enumerate(EXEMPLOS_SF311.items(), 1)]
    chamados += [(f"sf311_dia_{i}", i, ex) for i, ex in EXEMPLOS_SF311_DIA.items()]
    for nome, i, ex in chamados:
        r = sf.get(i)
        if r is None or r["arquivo"] not in avaliados or not (args.raw / "sf311" / r["arquivo"]).exists():
            faltando.append(nome)
            continue
        ctx = Contexto(capturado_em=datetime.fromisoformat(r["aberto_em"]).replace(tzinfo=SF),
                       lat=float(r["lat"]), lon=float(r["lon"]))
        entradas.append((nome, args.raw / "sf311" / r["arquivo"], ctx,
                         {"fonte": "San Francisco 311 (DataSF)", "autor": "cidadão anônimo (chamado 311)",
                          "licenca": "ODC PDDL (domínio público)", "link": r["url_foto"], "modificacao": MODIFICACAO},
                         ex))
    # Open Images: só fotos do teste reservado, com crédito ao autor.
    oi = {r["image_id"]: r for r in csv.DictReader(open(args.manifestos / "openimages_imagens.csv", encoding="utf-8"))}
    composicao = csv.DictReader(open(args.manifestos / "dataset_composicao.csv", encoding="utf-8"))
    teste = {r["image_id"]: r for r in composicao
             if r["split"] == "test" and r["grupo"] in ("noite_reservado", "anoitecer_reservado")}
    for i, ex in EXEMPLOS_OPENIMAGES.items():
        nome = f"openimages_{i}"
        if i not in oi or i not in teste or not (args.dataset / "images" / teste[i]["arquivo"]).exists():
            faltando.append(nome)
            continue
        meta = oi[i]
        entradas.append((nome, args.dataset / "images" / teste[i]["arquivo"], Contexto(periodo_informado="noite"),
                         {"fonte": "Open Images V7", "autor": meta["autor"], "licenca": meta["licenca"],
                          "link": meta["pagina_original"], "titulo": meta["titulo"], "modificacao": MODIFICACAO},
                         ex))
    if faltando:
        print(f"exemplos da curadoria sem foto, sem linha no manifesto ou fora da avaliação: {', '.join(faltando)}",
              file=sys.stderr)
        return 1

    analisador = criar_analisador(pesos=args.pesos, privacidade=True, threads=8)
    args.saida.mkdir(parents=True, exist_ok=True)
    creditos, divergentes = [], []
    for nome, foto, ctx, credito, ex in entradas:
        res, anotada = analisador.analisar(cv2.imread(str(foto)), ctx)
        cv2.imwrite(str(args.saida / f"{nome}.jpg"), anotada, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (args.saida / f"{nome}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        creditos.append({"exemplo": f"{nome}.jpg", "diagnostico": res["diagnostico"], **credito, "nota": ex.nota})
        if resumo(res) != tuple(ex[:5]):
            # A nota descreve o resultado da curadoria; com outros pesos ou outra regra ela pode não valer mais.
            print(f"AVISO: {nome} deu {resumo(res)}, mas a nota descreve {tuple(ex[:5])}; revise a nota ou troque "
                  "o exemplo", file=sys.stderr)
            divergentes.append(nome)

    with (args.saida / "creditos.csv").open("w", newline="", encoding="utf-8") as f:
        # Fotos do SF 311 não têm título: a coluna fica vazia (restval).
        w = csv.DictWriter(f, fieldnames=["exemplo", "diagnostico", "fonte", "titulo", "autor", "licenca", "link",
                                          "modificacao", "nota"], restval="")
        w.writeheader()
        w.writerows(creditos)
    # Exemplo que saiu da curadoria não pode continuar publicado: a foto ficaria sem linha em creditos.csv.
    gerados = {f"{nome}{ext}" for nome, *_ in entradas for ext in (".jpg", ".json")}
    antigos = [p for p in sorted(args.saida.iterdir()) if p.suffix in (".jpg", ".json") and p.name not in gerados]
    for p in antigos:
        p.unlink()
    print(f"{len(creditos)} exemplos em {args.saida}"
          + (f"; apagados {len(antigos)} arquivos de exemplos antigos" if antigos else ""))
    return 1 if divergentes else 0


if __name__ == "__main__":
    sys.exit(main())
