
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

from extractor import extraer_features, NOMBRES

st.set_page_config(
    page_title="Detector de Phishing con IA",
    page_icon="🔐",
    layout="wide"
)


class DetectorPhishing:

    def __init__(self, carpeta):

        self.modelo = tf.keras.models.load_model(
            os.path.join(carpeta, "modelo.keras")
        )

        self.scaler = joblib.load(
            os.path.join(carpeta, "scaler.pkl")
        )

        with open(
            os.path.join(carpeta, "tld_map.json"),
            encoding="utf-8"
        ) as f:
            self.tld_map = json.load(f)

        with open(
            os.path.join(carpeta, "config.json"),
            encoding="utf-8"
        ) as f:
            cfg = json.load(f)

        self.features = cfg["features"]
        self.umbral = float(cfg["umbral"])
        self.prior = float(cfg["media_global"])


    @staticmethod
    def normalizar(url):

        url = str(url).strip()

        if re.match(
            r"^[a-zA-Z][a-zA-Z0-9+.\-]*://",
            url
        ):
            return url

        return "https://" + url


    def _matriz(self, urls):

        filas = []
        tlds = []

        for url in urls:
            f, t = extraer_features(url)
            filas.append(f)
            tlds.append(t)

        X = pd.DataFrame(filas)[NOMBRES]

        X["tld_te"] = [
            self.tld_map.get(t, self.prior)
            for t in tlds
        ]

        return X[self.features].astype("float32")


    def predecir(self, urls):

        normalizadas = [
            self.normalizar(u)
            for u in urls
        ]

        X = self._matriz(normalizadas)

        Xs = self.scaler.transform(
            np.log1p(X.values)
        )

        p = self.modelo.predict(
            Xs,
            verbose=0
        ).ravel()

        return pd.DataFrame({
            "url": urls,
            "prob_legitimo": p,
            "prob_phishing": 1 - p,
            "veredicto": np.where(
                p >= self.umbral,
                "Legítimo",
                "Phishing"
            )
        })


@st.cache_resource
def cargar_modelos():

    original = DetectorPhishing("original")
    mixto = DetectorPhishing("mixto")
    final = DetectorPhishing("final")

    return original, mixto, final


detector_original, detector_mixto, detector_final = cargar_modelos()


st.title("🔐 Detección de phishing mediante Inteligencia Artificial")
st.subheader("Evolución del modelo: Original → Mixto → Final")

st.write(
    """
    El sistema analiza características léxicas de una URL
    sin descargar ni ejecutar el contenido de la página web.
    """
)

url = st.text_input(
    "URL a analizar",
    value="https://www.unemi.edu.ec/"
)


def mostrar_resultado(nombre, detector, resultado):

    p_leg = float(resultado["prob_legitimo"])
    p_phi = float(resultado["prob_phishing"])
    veredicto = resultado["veredicto"]

    st.markdown(f"### {nombre}")

    if veredicto == "Legítimo":
        st.success(f"✅ {veredicto}")
    else:
        st.error(f"⚠️ {veredicto}")

    st.metric(
        "Probabilidad legítimo",
        f"{p_leg*100:.2f}%"
    )

    st.metric(
        "Probabilidad phishing",
        f"{p_phi*100:.2f}%"
    )

    st.write(
        f"**Umbral requerido:** "
        f"{detector.umbral*100:.0f}%"
    )


if st.button(
    "Analizar con los tres modelos",
    type="primary",
    use_container_width=True
):

    if not url.strip():
        st.warning("Ingrese una URL.")

    else:
        ro = detector_original.predecir([url]).iloc[0]
        rm = detector_mixto.predecir([url]).iloc[0]
        rf = detector_final.predecir([url]).iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            mostrar_resultado(
                "MODELO ORIGINAL",
                detector_original,
                ro
            )

        with col2:
            mostrar_resultado(
                "MODELO MIXTO",
                detector_mixto,
                rm
            )

        with col3:
            mostrar_resultado(
                "MODELO FINAL",
                detector_final,
                rf
            )


st.divider()

st.caption(
    "Prototipo académico de detección de phishing mediante análisis léxico de URLs."
)
