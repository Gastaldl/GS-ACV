"""Reproduz o componente de visão do Luz Segura do zero, na ordem certa.

  python scripts/reproduzir.py                  # tudo: dados, treino do v1 (cerca de 4 h em CPU), pseudo-rótulos
                                                # do v2, avaliação, escolha do modelo e demo
  python scripts/reproduzir.py --pular-treino   # usa os pesos versionados (models/luzsegura_yolo26n_v1.pt)
  python scripts/reproduzir.py --com-v2         # também treina e avalia o v2 (autotreinamento, mais cerca de 2 h)
  python scripts/reproduzir.py --pular-treino --etapas avaliacao_v1 comparar benchmark demo

Etapas (sempre nesta ordem, mesmo que --etapas as liste em outra):
  dados         downloads (refazem exatamente as linhas dos manifestos versionados), remoção de fotos
                repetidas e o pool de treino do v2 (fotos do SF 311 separadas das de avaliação)
  dataset       dataset do v1 (Open Images e noites simuladas)
  treino        treino do v1 (pesos também em models/luzsegura_yolo26n_v1.pt) e resumo em results/treino
  avaliacao_v1  teste de detecção e avaliação por imagem do v1 (resultados com sufixo _v1)
  v2            pseudo-rótulos do v1 no pool do SF 311 (data/manifests/pseudo_rotulos_v2.csv, que o relatório
                usa) e, só com --com-v2, o treino do v2 a partir do v1 e o resumo em results/treino_v2
  avaliacao_v2  só com --com-v2: o mesmo do avaliacao_v1 para o v2 (sufixo _v2)
  comparar      escolhe v1 ou v2 pela regra fixada antes (scripts/avaliacao/comparar_modelos.py), copia os
                pesos escolhidos para models/luzsegura_yolo26n.pt e publica os resultados deles sem sufixo
  benchmark     velocidade em CPU dos pesos escolhidos
  demo          exemplos de fotos e vídeos anotados

--pular-treino pula os treinos e o que só serve a eles (pool de treino e dataset v2); as avaliações usam
os pesos versionados em models/. Sem treino, a divisão do dataset precisa sair igual à versionada
(data/manifests/dataset_composicao.csv), senão o teste poderia conter fotos do treino desses pesos: se
alguma foto faltar no download, a etapa dataset para e diz o que fazer. O modelo entregue é o v1: os
pseudo-rótulos aceitos foram poucos e ruidosos (o relatório mostra os números), e o v2 só é treinado e
entra na escolha com --com-v2.
Uma etapa que falha interrompe tudo e diz como retomar dali.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

V1 = "models/luzsegura_yolo26n_v1.pt"
V2 = "models/luzsegura_yolo26n_v2.pt"
TREINO, SO_V2 = "treino", "v2"  # marcas: pulado por --pular-treino / sem --com-v2


def avaliar(pesos: str, sufixo: str, marcas: tuple = ()) -> list:
    return [(["scripts/avaliacao/avaliar_deteccao.py", "--pesos", pesos, "--sufixo", sufixo], marcas),
            (["scripts/avaliacao/avaliar_imagens.py", "--pesos", pesos, "--sufixo", sufixo], marcas)]


# Etapa -> comandos (script .py e argumentos) com as marcas que fazem o comando ser pulado.
ETAPAS = {
    "dados": [
        (["scripts/dados/baixar_anotacoes_openimages.py"], ()),
        (["scripts/dados/baixar_openimages.py"], ()),
        (["scripts/dados/baixar_chennai.py"], ()),
        (["scripts/dados/baixar_sf311.py"], ()),
        (["scripts/dados/baixar_videos.py"], ()),
        (["scripts/dados/deduplicar_manifestos.py"], ()),
        (["scripts/dados/baixar_sf311.py", "--treino"], (TREINO,)),
    ],
    "dataset": [
        (["scripts/dados/analisar_luminancia.py"], ()),
        (["scripts/dados/montar_dataset.py"], ()),
    ],
    "treino": [
        (["scripts/treino/treinar.py", "--dados", "data/dataset/luzsegura.yaml", "--nome", "luzsegura_yolo26n_v1",
          "--destino-resultados", "results/treino", "--epocas", "25"], (TREINO,)),
        (["scripts/treino/resumir_treino.py", "--resultados", "runs/treino/luzsegura_yolo26n_v1/results.csv",
          "--saida", "results/treino/resumo_treino.json"], (TREINO,)),
    ],
    "avaliacao_v1": avaliar(V1, "_v1"),
    "v2": [
        (["scripts/dados/montar_dataset_v2.py", "--pesos", V1], (TREINO,)),
        (["scripts/treino/treinar.py", "--dados", "data/dataset_v2/luzsegura_v2.yaml", "--modelo", V1,
          "--nome", "luzsegura_yolo26n_v2", "--destino-pesos", V2, "--destino-resultados", "results/treino_v2",
          "--epocas", "12"], (TREINO, SO_V2)),
        (["scripts/treino/resumir_treino.py", "--resultados", "runs/treino/luzsegura_yolo26n_v2/results.csv",
          "--saida", "results/treino_v2/resumo_treino.json"], (TREINO, SO_V2)),
    ],
    "avaliacao_v2": avaliar(V2, "_v2", (SO_V2,)),
    "comparar": [(["scripts/avaliacao/comparar_modelos.py"], ())],
    "benchmark": [(["scripts/avaliacao/benchmark_fps.py"], ())],
    "demo": [
        (["scripts/demo/exemplos_fotos.py"], ()),
        (["scripts/demo/processar_videos.py"], ()),
    ],
}
# Pesos que cada etapa precisa encontrar antes de começar, e a etapa que os grava (com --pular-treino,
# vêm do repositório).
PESOS_EXIGIDOS = {"avaliacao_v1": [V1], "v2": [V1], "avaliacao_v2": [V2]}
ORIGEM_PESOS = {V1: "treino", V2: "v2"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etapas", nargs="*", choices=list(ETAPAS), default=list(ETAPAS))
    ap.add_argument("--pular-treino", action="store_true", help="não treina: usa os pesos versionados em models/")
    ap.add_argument("--com-v2", action="store_true", help="também treina e avalia o v2 e o coloca na escolha")
    args = ap.parse_args()
    raiz = Path(__file__).resolve().parents[1]
    etapas = [e for e in ETAPAS if e in args.etapas]
    treina_v1 = "treino" in etapas and not args.pular_treino
    opcoes = [o for o, ativa in (("--pular-treino", args.pular_treino), ("--com-v2", args.com_v2)) if ativa]
    for i, etapa in enumerate(etapas):
        retomar = " ".join(["python scripts/reproduzir.py", *opcoes, "--etapas", *etapas[i:]])
        comandos = []
        for cmd, marcas in ETAPAS[etapa]:
            if TREINO in marcas and args.pular_treino:
                print(f"== {etapa}: {' '.join(cmd)} -- pulado (--pular-treino)", flush=True)
            elif SO_V2 in marcas and not args.com_v2:
                print(f"== {etapa}: {' '.join(cmd)} -- pulado (só com --com-v2)", flush=True)
            else:
                # Sem --com-v2, a escolha também ignora resultados antigos do v2 que estejam em results/.
                if etapa == "comparar" and not args.com_v2:
                    cmd = [*cmd, "--sem-v2"]
                # Divisão do dataset diferente da versionada só vale se o v1 for treinado de novo nesta execução;
                # sem treino, montar_dataset.py recusa (o teste poderia conter fotos do treino dos pesos versionados).
                elif cmd[0] == "scripts/dados/montar_dataset.py" and treina_v1:
                    cmd = [*cmd, "--nova-divisao"]
                comandos.append(cmd)
        for p in PESOS_EXIGIDOS.get(etapa, []) if comandos else []:
            if not (raiz / p).exists():
                print(f"falhou na etapa {etapa}: faltam os pesos {p}; eles saem da etapa {ORIGEM_PESOS[p]} "
                      f"(rode-a sem --pular-treino){'; para seguir só com o v1, rode sem --com-v2' if p == V2 else ''}. "
                      f"Depois retome com: {retomar}", file=sys.stderr)
                return 1
        for cmd in comandos:
            print(f"== {etapa}: {' '.join(cmd)}", flush=True)
            # Os scripts rodam no mesmo interpretador, em modo UTF-8 (acentos na saída de qualquer console).
            r = subprocess.run([sys.executable, "-X", "utf8", *cmd], cwd=raiz)
            if r.returncode != 0:
                print(f"falhou na etapa {etapa}: {' '.join(cmd)} (código {r.returncode}). Corrija e retome com: "
                      f"{retomar}", file=sys.stderr)
                return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
