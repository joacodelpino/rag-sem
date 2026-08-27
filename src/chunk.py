"""
El tipo que viaja por todo el pipeline.

Vive en su propio módulo, y no dentro de retrieval.py, para que rerank.py
pueda construir Chunks sin importar a retrieval.py (que a su vez importa a
rerank.py, y sería un import circular).

El campo `id` no es decorativo: las métricas de recuperación (recall@k, MRR,
NDCG) se calculan comparando ids recuperados contra ids relevantes del set
dorado. Comparar strings de texto sería frágil.

`score` NO significa lo mismo en todas las configuraciones — es siempre el
score de la última etapa que tocó el chunk, que es la que explica el orden en
que salió:
    naive    -> similitud coseno (0–1, pero en la práctica 0.4–0.8)
    sparse   -> BM25 (sin techo)
    híbrida  -> score de RRF (valores chicos, ~0.03)
    rerank   -> probabilidad del cross-encoder (0–1, bien repartido)
Por eso los scores se pueden comparar DENTRO de una columna de la demo, pero
nunca entre columnas.

Los campos de identidad documental (titulo, numero_ley, version, seccion)
salen del manifiesto y se guardan en el payload de Qdrant durante la ingesta.
Tienen default vacío a propósito: así el código sigue funcionando contra un
índice viejo que no los tenga, en vez de reventar con KeyError. Si aparecen
vacíos en la app, es señal de que hay que reingestar.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    score: float
    titulo: str = ""
    numero_ley: str = ""
    version: str = ""
    seccion: str = ""

    def etiqueta(self) -> str:
        """Nombre legible de la fuente, para mostrar en la app y en las citas.

        El nombre de archivo no sirve: cinco documentos del corpus se llaman
        "0N_argentinagobar.pdf" y "08_ley-15.pdf" no es la ley 15 sino el
        Decreto-Ley 15.348. La versión va incluida porque es lo que evita que
        alguien lea el texto de la LCT de 1974 creyendo que es el vigente.
        """
        if not self.titulo:
            return self.source
        partes = [self.titulo]
        if self.version and self.version != "no_aplica":
            partes.append(self.version.replace("_", " "))
        if self.seccion:
            partes.append(self.seccion)
        return " · ".join(partes)
