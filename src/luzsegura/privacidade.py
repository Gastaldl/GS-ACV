"""Filtro de privacidade (LGPD): anonimiza pessoas e veículos antes de exibir ou salvar uma imagem.

Imagens de rua captam rostos e placas. O Luz Segura não precisa de nenhum dos dois:
um detector pré-treinado no COCO localiza pessoas e veículos, e a região é pixelada
e desfocada de forma irreversível. A imagem original nunca é gravada; só o resultado
(contagens, diagnóstico, localização informada) e, quando necessário, a versão anonimizada.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import CLASSES_PRIVACIDADE, PESOS_PRIVACIDADE

CLASSE_PESSOA = 0


def anonimizar_regiao(img: np.ndarray, xyxy, margem: float = 0.08, blocos: int = 6) -> None:
    """Pixela e desfoca a região (in-place). ~`blocos` blocos no menor lado: rosto e placa ilegíveis."""
    altura, largura = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    dx, dy = (x2 - x1) * margem, (y2 - y1) * margem
    x1, y1 = int(max(0, x1 - dx)), int(max(0, y1 - dy))
    x2, y2 = int(min(largura, x2 + dx)), int(min(altura, y2 + dy))
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return
    bloco = max(4, round(min(w, h) / blocos))
    pequeno = cv2.resize(img[y1:y2, x1:x2], (max(1, w // bloco), max(1, h // bloco)), interpolation=cv2.INTER_AREA)
    # A região vira uma grade de ~`blocos` médias (irreversível); ampliá-la com interpolação bilinear pixela e
    # suaviza de uma vez, a custo fixo. O desfoque gaussiano com sigma proporcional ao bloco levava ~230 ms
    # numa região de 700 px (um ônibus perto da câmera) e derrubava o vídeo para 5 quadros por segundo.
    img[y1:y2, x1:x2] = cv2.resize(pequeno, (w, h), interpolation=cv2.INTER_LINEAR)


class FiltroPrivacidade:
    """Detecta pessoas e veículos (COCO) e anonimiza as regiões."""

    def __init__(self, pesos: str = PESOS_PRIVACIDADE, imgsz: int = 640, conf: float = 0.25, dispositivo: str = "cpu"):
        from ultralytics import YOLO

        self.modelo = YOLO(pesos)
        self.imgsz = imgsz
        self.conf = conf
        self.dispositivo = dispositivo
        self.classes = sorted(CLASSES_PRIVACIDADE)

    def detectar(self, img_bgr: np.ndarray) -> list[tuple[int, tuple]]:
        r = self.modelo.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, classes=self.classes,
                                device=self.dispositivo, verbose=False)[0]
        if r.boxes is None:
            return []
        return [(int(c), tuple(b)) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist())]

    def aplicar(self, img_bgr: np.ndarray, deteccoes: list[tuple[int, tuple]] | None = None) -> tuple[np.ndarray, dict]:
        """Devolve uma cópia anonimizada e a contagem do que foi desfocado."""
        deteccoes = self.detectar(img_bgr) if deteccoes is None else deteccoes
        saida = img_bgr.copy()
        for _, xyxy in deteccoes:
            anonimizar_regiao(saida, xyxy)
        pessoas = sum(1 for c, _ in deteccoes if c == CLASSE_PESSOA)
        return saida, {"pessoas_desfocadas": pessoas, "veiculos_desfocados": len(deteccoes) - pessoas,
                       "imagem_armazenada": False, "filtro_ativo": True}


def sem_filtro() -> dict:
    return {"pessoas_desfocadas": 0, "veiculos_desfocados": 0, "imagem_armazenada": False, "filtro_ativo": False}
