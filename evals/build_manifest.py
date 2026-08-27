# -*- coding: utf-8 -*-
"""Genera data/manifest.csv: una fila por PDF de data/raw/.

La columna `version` NO se adivina. Se deriva de una señal verificable que
traen los propios documentos: la URL de origen impresa en la primera página.

  infoleg.gob.ar/.../norma.htm    -> el texto tal como se publicó (original)
  infoleg.gob.ar/.../texact.htm   -> texto actualizado con las modificatorias
  argentina.gob.ar/.../texto      -> texto original
  argentina.gob.ar/.../actualizacion -> texto actualizado

Para los documentos que no son normativa (fallos, contratos modelo, folletos)
la distinción no aplica y se marca `no_aplica`. Cuando no hay señal alguna se
marca `indeterminado`.
"""
import csv, pathlib, re
import pymupdf

RAW = pathlib.Path("data/raw")
SALIDA = pathlib.Path("data/manifest.csv")

# titulo y numero_ley curados a partir de la primera página de cada PDF.
# El nombre de archivo no sirve como título: varios son "argentinagobar.pdf"
# o vienen truncados ("ley-15" en realidad es el Decreto-Ley 15.348).
CURADO = {
 "01_argentinagobar.pdf": ("Ley 24.240 - Defensa del Consumidor. Normas de Protección y Defensa de los Consumidores", "24.240"),
 "02_argentinagobar.pdf": ("Ley 20.785 - Bienes objeto de secuestro en causas penales. Custodia y disposición", "20.785"),
 "03_argentinagobar.pdf": ("Decreto 1148/1991 - Reglamentación del artículo 39 de la Ley 23.737", ""),
 "04_argentinagobar.pdf": ("Ley 23.737 - Modificación al Código Penal. Narcotráfico (estupefacientes)", "23.737"),
 "05_argentinagobar.pdf": ("Ley 25.938 - Registro Nacional de Armas de Fuego y Materiales Controlados, Secuestrados o Incautados", "25.938"),
 "06_infoleg.pdf": ("Ley 20.744 - Régimen de Contrato de Trabajo (LCT)", "20.744"),
 "07_infoleg-ministerio-de-economía-y-finanzas-públicas.pdf": ("Ley 11.179 - Código Penal de la Nación", "11.179"),
 "08_ley-15.pdf": ("Decreto 897/1995 - Ley de Prenda. Texto ordenado del Decreto-Ley 15.348/46 (ratificado por Ley 12.962)", "15.348"),
 "09_ley-n-11.pdf": ("Ley 11.723 - Régimen Legal de la Propiedad Intelectual", "11.723"),
 "10_ley-de-bien-de-familia-edad-de-matrimonio.pdf": ("Ley 14.394 - Modificaciones al régimen de los menores y de la familia (bien de familia, edad de matrimonio)", "14.394"),
 "11_infoleg-ministerio-de-economía-y-finanzas-públicas.pdf": ("Ley 24.156 - Administración Financiera y de los Sistemas de Control del Sector Público Nacional", "24.156"),
 "12_siri-angel-1957.pdf": ("CSJN, \"Siri, Ángel\" (1957), Fallos 239:459 - amparo contra actos del Estado", ""),
 "13_samuel-kot-1958.pdf": ("CSJN, \"Samuel Kot SRL\" (1958), Fallos 241:291 - amparo contra actos de particulares", ""),
 "14_outón-1967.pdf": ("CSJN, \"Outón\" (1967), Fallos 267:215 - libertad de agremiación", ""),
 "15_alitt-2006.pdf": ("CSJN, \"ALITT c/ Inspección General de Justicia\" (2006) - personería jurídica", ""),
 "16_3275118-hooft-pedro-c-f-c-provincia-de-buenos-aires-16112004.pdf": ("CSJN, \"Hooft, Pedro C. F. c/ Provincia de Buenos Aires\" (2004), Fallos 327:5118", ""),
 "17_resolución-558-sistema-de-gestion-ambiental-deroganse-resoluciones.pdf": ("ENRE, Resolución 558/2022 - Sistema de Gestión Ambiental. Deróganse resoluciones", ""),
 "18_resolución-404-adendas-por-mayor-financiamiento-modificase.pdf": ("Ministerio de Obras Públicas, Resolución 404/2022 - Adendas por mayor financiamiento. Modifícase", ""),
 "19_resolución-160-resolucion-212017-modificacion.pdf": ("Secretaría de Agricultura, Ganadería y Pesca, Resolución 160/2022 - Modificación de la Resolución 21/2017", ""),
 "20_resolución-2118-pliego-de-bases-y-condiciones-generales-y-particulares-apruebase.pdf": ("ENACOM, Resolución 2118/2022 - Pliego de bases y condiciones generales y particulares. Apruébase", ""),
 "21_resolución-75-modelo-de-convenio.pdf": ("Secretaría de Comercio, Resolución 75/2022 - Modelo de convenio (defensa del consumidor)", ""),
 "22_resolución-46-bono-de-la-nacion-argentina-en-moneda-dual-ampliase.pdf": ("Secretarías de Finanzas y de Hacienda, Resolución Conjunta 46/2022 - Bono de la Nación Argentina en moneda dual. Amplíase", ""),
 "23_resolución-865-dumping-apertura-del-examen.pdf": ("Ministerio de Economía, Resolución 865/2022 - Dumping. Apertura del examen", ""),
 "24_resolución-867-dumping-apertura-del-examen.pdf": ("Ministerio de Economía, Resolución 867/2022 - Dumping. Apertura del examen", ""),
 "25_resolución-282-resolucion-692015-revocase.pdf": ("AABE, Resolución 282/2022 - Revócase la Resolución 69/2015", ""),
 "26_resolución-5289-informe-o-estado-de-situacion.pdf": ("IGJ y AFIP, Resolución General Conjunta 5289/2022 - Asociaciones civiles categoría I. Informe o estado de situación", ""),
 "27_resolución-99-reunion-invitase.pdf": ("Secretaría de Comercio, Resolución 99/2022 - Reunión. Invítase", ""),
 "28_resolución-286-cooperativa-agricola-ganadera-peñi-mapuche-limitada.pdf": ("AABE, Resolución 286/2022 - Permiso de uso a la Cooperativa Agrícola Ganadera Peñi Mapuche Limitada", ""),
 "29_resolución-274-municipalidad-de-rio-primero-de-la-provincia-de-cordoba.pdf": ("AABE, Resolución 274/2022 - Municipalidad de Río Primero, Provincia de Córdoba", ""),
 "30_disposición-899-cinemometros-controladores-de-velocidad-homologase-y-autorizase-.pdf": ("ANSV, Disposición 899/2022 - Cinemómetros controladores de velocidad. Homológase y autorízase uso", ""),
 "31_resolución-278-municipalidad-de-escobar.pdf": ("AABE, Resolución 278/2022 - Municipalidad de Escobar", ""),
 "contrato-alquiler-comercial-local-argentina-2026.pdf": ("Contrato de locación comercial - local / depósito (modelo con campos en blanco)", ""),
 "contrato-alquiler-oficina-argentina-2026.pdf": ("Contrato de locación comercial - oficina profesional (modelo con campos en blanco)", ""),
 "EstatutoUNLAR.pdf": ("Resolución ME 2485-E/2017 - Estatuto Académico de la Universidad Nacional de La Rioja (texto ordenado)", ""),
 "riesgos-del-trabajo-UNLAR.pdf": ("Preguntas más frecuentes sobre el Sistema de Riesgos del Trabajo (SRT, 2016) - material de divulgación", ""),
 "Texto-Ordenado---Reglamento-General-de-Alumnos---OHCS-283-04-y-modificatorias.pdf": ("Reglamento General de Alumnos - OHCS 283/04 y modificatorias (texto ordenado)", ""),
}

# documentos donde la distinción original/actualizado no aplica
NO_APLICA = {"12_", "13_", "14_", "15_", "16_", "contrato-", "riesgos-"}


def detectar_version(nombre: str, texto_p1: str) -> str:
    if any(nombre.startswith(p) for p in NO_APLICA):
        return "no_aplica"
    plano = re.sub(r"\s+", " ", texto_p1)
    # El documento dice de SÍ MISMO que es un texto ordenado. No alcanza con
    # que aparezca la frase suelta: varias resoluciones citan el t.o. DE OTRA
    # norma (28/29/31 mencionan el "(texto ordenado)" de la Resolución
    # 177/2022, que no es este documento). Por eso se exige el verbo
    # aprobatorio, o que el propio PDF arranque con ese título.
    if re.search(r"Apru[eé]base el texto ordenado|publicaci[oó]n del texto ordenado", plano, re.I):
        return "texto_ordenado"
    if re.match(r"\s*TEXTO ORDENADO", texto_p1, re.I):
        return "texto_ordenado"
    # Señal de origen: la URL que infoleg / argentina.gob.ar imprimen en la 1ª página.
    if re.search(r"texact\.htm|/actualizacion", plano):
        return "texto_actualizado"
    if re.search(r"norma\.htm|/texto\b", plano):
        return "texto_original"
    return "indeterminado"


filas = []
for p in sorted(RAW.glob("*.pdf")):
    doc = pymupdf.open(p)
    p1 = doc[0].get_text()
    doc.close()
    titulo, ley = CURADO.get(p.name, ("", ""))
    if not titulo:
        titulo = "indeterminado"
    filas.append({
        "archivo": p.name,
        "titulo": titulo,
        "numero_ley": ley,
        "version": detectar_version(p.name, p1),
    })

with SALIDA.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["archivo", "titulo", "numero_ley", "version"])
    w.writeheader()
    w.writerows(filas)

print("escrito:", SALIDA, "|", len(filas), "filas")
from collections import Counter
for k, v in Counter(f["version"] for f in filas).most_common():
    print(f"  {k}: {v}")
