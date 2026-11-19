"""Detector de luminárias (YOLO26n treinado no projeto) com medida de brilho por caixa."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import CLASSES, LIMIARES_PADRAO, Limiares, caminho_pesos_padrao
from .diagnostico import Luminaria
from .luminancia import luma, medir_caixa

# ByteTrack com trilhas mais longas: luminárias são paradas e o detector pisca nas pequenas (ver o yaml).
RASTREADOR = Path(__file__).with_name("rastreador_luminarias.yaml")


class DetectorLuminarias:
    """Carrega os pesos (.pt ou .onnx) e devolve luminárias com classe, confiança, caixa e brilho."""

    def __init__(self, pesos: str | Path | None = None, imgsz: int = 640, conf: float | None = None,
                 dispositivo: str = "cpu", lim: Limiares = LIMIARES_PADRAO, threads: int | None = None):
        from ultralytics import YOLO  # importado aqui para os testes não dependerem do modelo

        if threads:
            torch.set_num_threads(threads)
        self.pesos = Path(pesos) if pesos else caminho_pesos_padrao()
        if not self.pesos.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {self.pesos}. Rode scripts/treino/treinar.py "
                                    "ou defina LUZSEGURA_PESOS.")
        self.modelo = YOLO(str(self.pesos), task="detect")
        self.imgsz = imgsz
        self.dispositivo = dispositivo
        self.lim = lim
        self.conf = lim.conf_minima if conf is None else conf
        self.nomes = {int(k): v for k, v in self.modelo.names.items()}
        desconhecidas = set(self.nomes.values()) - set(CLASSES)
        if desconhecidas:
            raise ValueError(f"Modelo com classes inesperadas: {sorted(desconhecidas)}")

    def detectar(self, img_bgr: np.ndarray, rastrear: bool = False) -> list[Luminaria]:
        """Detecta luminárias. Com rastrear=True mantém IDs entre quadros de um vídeo (ByteTrack)."""
        kw = {"imgsz": self.imgsz, "conf": self.conf, "device": self.dispositivo, "verbose": False}
        if rastrear:
            # O ByteTrack usa detecções fracas para manter o ID entre quadros; o corte de
            # confiança (Limiares.conf_minima) continua valendo no diagnóstico, no desenho e na contagem.
            kw["conf"] = min(self.conf, 0.10)
            r = self.modelo.track(img_bgr, persist=True, tracker=str(RASTREADOR), **kw)[0]
        else:
            r = self.modelo.predict(img_bgr, **kw)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        # A regra de brilho conta pixels (mancha mínima, nenhuma mancha para "apagada"): é medida sempre
        # com o lado maior em lim.lado_medida, a resolução da calibração, para a mesma foto ter o mesmo
        # estado no app, na API, na linha de comando e no vídeo. A caixa devolvida segue na escala original.
        y = luma(img_bgr)
        f = min(1.0, self.lim.lado_medida / max(y.shape[:2]))
        if f < 1.0:
            y = cv2.resize(y, (max(1, round(y.shape[1] * f)), max(1, round(y.shape[0] * f))),
                           interpolation=cv2.INTER_AREA)
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        saida = []
        for xyxy, cls, cf, tid in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), ids):
            medida = medir_caixa(y, tuple(v * f for v in xyxy), self.lim)
            saida.append(Luminaria(self.nomes[cls], float(cf), tuple(xyxy), medida, tid))
        return remover_duplicatas(saida, self.lim.iou_mesma_luminaria)

    def reiniciar_rastreio(self) -> None:
        """Zera o rastreador antes de começar um novo vídeo."""
        preditor = getattr(self.modelo, "predictor", None)
        for rastreador in getattr(preditor, "trackers", None) or []:
            rastreador.reset()


def _iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def remover_duplicatas(luminarias: list[Luminaria], iou_max: float) -> list[Luminaria]:
    """Uma caixa por luminária, da mais confiante para a menos confiante.

    O YOLO26 é de ponta a ponta, sem supressão de não máximos entre classes: a mesma luminária às
    vezes sai duas vezes, como acesa e como apagada. Contada duas vezes, ela distorceria o diagnóstico.
    """
    mantidas: list[Luminaria] = []
    for l in sorted(luminarias, key=lambda l: -l.confianca):
        if all(_iou(l.xyxy, m.xyxy) <= iou_max for m in mantidas):
            mantidas.append(l)
    return mantidas


def onnx_atualizado(pt: str | Path, imgsz: int = 640) -> Path:
    """Devolve o .onnx exportado DESTES pesos .pt; reexporta se faltar ou se veio de outro .pt.

    Um selo SHA-256 do .pt fica ao lado do .onnx (<nome>.onnx.sha256). Sem ele, um .onnx antigo
    sobreviveria a um novo treino e os vídeos sairiam de um modelo diferente das tabelas.
    """
    from ultralytics import YOLO

    pt = Path(pt)
    onnx = pt.with_suffix(".onnx")
    selo = onnx.with_name(onnx.name + ".sha256")
    h = hashlib.sha256(pt.read_bytes()).hexdigest()
    if not (onnx.exists() and selo.exists() and selo.read_text(encoding="utf-8").strip() == h):
        exportado = Path(YOLO(str(pt)).export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False))
        if exportado.resolve() != onnx.resolve():
            exportado.replace(onnx)
        selo.write_text(h + "\n", encoding="utf-8")
    return onnx
