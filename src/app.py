import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Configuración de la página
st.set_page_config(page_title="Análisis de Clústeres de Tarjetas de Crédito", layout="wide")


# Cargar datos y modelos
@st.cache_data
def load_data():
    df = pd.read_csv("../data/processed_data/creditcards_clusters.csv")
    pca_model = joblib.load('../models/pca_model.pkl')
    kmeans_model = joblib.load('../models/kmeans_model.pkl')
    return df, pca_model, kmeans_model


df, pca_model, kmeans_model = load_data()


# Funciones de utilidad
def get_marketing_strategies(cluster):
    strategies = {
        0: [
            "Ofrecer programas de recompensas para mantener la lealtad",
            "Promocionar productos de inversión de alto rendimiento",
            "Invitar a eventos exclusivos para clientes premium"
        ],
        1: [
            "Incentivar el uso de la tarjeta con promociones especiales",
            "Ofrecer seguros y protecciones adicionales",
            "Proporcionar herramientas de gestión financiera personalizada"
        ],
        2: [
            "Ofrecer planes de pago flexibles",
            "Proporcionar educación financiera y asesoramiento",
            "Incentivar el uso de la tarjeta en compras cotidianas con cashback"
        ]
    }
    return strategies.get(cluster, ["Estrategia personalizada basada en el perfil del cliente"])


def predict_cluster(client_data):
    features = [col for col in df.columns if col not in ['CUST_ID', 'TENURE', 'Cluster']]
    client_features = client_data[features].values.reshape(1, -1)
    client_pca = pca_model.transform(client_features)
    cluster = kmeans_model.predict(client_pca)[0]
    return cluster


# Título principal
st.title("Análisis de Clústeres de Tarjetas de Crédito")

# Sidebar para navegación
page = st.sidebar.selectbox("Seleccione una página",
                            ["Visión General", "Recomendador de Estrategias", "Comparador de Clientes",
                             "Predictor de Cluster", "Análisis Detallado"])

if page == "Visión General":
    st.header("Visión General de los Clústeres")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Estadísticas Generales")
        st.write(df.describe())

    with col2:
        st.subheader("Distribución de Clústeres")
        fig = px.pie(df, names='Cluster', title='Distribución de Clientes por Clúster')
        st.plotly_chart(fig)

    st.subheader("Visualización de Clusters")
    if 'PC1' in df.columns and 'PC2' in df.columns:
        fig = px.scatter(df, x='PC1', y='PC2', color='Cluster',
                         hover_data=df.columns.tolist(),
                         title="Visualización 2D de Clústeres")
        st.plotly_chart(fig)
    else:
        st.write("No se encontraron las columnas PC1 y PC2 para la visualización.")

elif page == "Recomendador de Estrategias":
    st.header("Recomendador de Estrategias de Marketing")

    selected_cluster = st.selectbox("Seleccione un clúster", sorted(df['Cluster'].unique()))

    st.subheader(f"Estrategias para el Clúster {selected_cluster}")
    for strategy in get_marketing_strategies(selected_cluster):
        st.write(f"- {strategy}")

    st.subheader("Perfil del Clúster Seleccionado")
    cluster_profile = df[df['Cluster'] == selected_cluster].mean()
    st.write(cluster_profile)

    # ... (rest of the code for this page remains the same)

elif page == "Comparador de Clientes":
    st.header("Comparador de Clientes")

    selected_clients = st.multiselect(
        "Seleccione clientes para comparar",
        df['CUST_ID'].tolist(),
        max_selections=3
    )

    if selected_clients:
        comparison_df = df[df['CUST_ID'].isin(selected_clients)]

        # Seleccionar las columnas numéricas para la comparación
        numeric_columns = comparison_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in ['CUST_ID', 'Cluster']]

        fig = go.Figure()

        for client in selected_clients:
            client_data = comparison_df[comparison_df['CUST_ID'] == client]
            values = client_data[numeric_columns].values.flatten().tolist()
            values += values[:1]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=numeric_columns + [numeric_columns[0]],
                fill='toself',
                name=f'Cliente {client}'
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Comparación de Características de Clientes"
        )

        st.plotly_chart(fig)

        st.subheader("Detalles de los Clientes Seleccionados")
        st.write(comparison_df)
    else:
        st.write("Por favor, seleccione al menos un cliente para comparar.")

elif page == "Predictor de Cluster":
    st.header("Predictor de Cluster para Nuevo Cliente")

    # Obtener las columnas numéricas del DataFrame, excluyendo 'CUST_ID', 'TENURE' y 'Cluster'
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    input_columns = [col for col in numeric_columns if col not in ['CUST_ID', 'TENURE', 'Cluster']]

    col1, col2 = st.columns(2)
    new_client_data = {}

    for i, column in enumerate(input_columns):
        with col1 if i % 2 == 0 else col2:
            new_client_data[column] = st.number_input(f"{column}", value=0.0)

    if st.button("Predecir Cluster"):
        new_client_df = pd.DataFrame([new_client_data])

        predicted_cluster = predict_cluster(new_client_df)
        st.write(f"El nuevo cliente pertenecería al Cluster {predicted_cluster}")

        st.subheader("Estrategias de Marketing Recomendadas")
        for strategy in get_marketing_strategies(predicted_cluster):
            st.write(f"- {strategy}")

        st.subheader("Comparación con el Perfil del Cluster")
        cluster_profile = df[df['Cluster'] == predicted_cluster].mean()
        comparison = pd.concat([new_client_df.T, cluster_profile.to_frame().T]).T
        comparison.columns = ['Nuevo Cliente', 'Promedio del Cluster']
        st.write(comparison)

elif page == "Análisis Detallado":
    st.header("Análisis Detallado de Clústeres")

    st.subheader("Distribución de Variables por Clúster")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    variable = st.selectbox("Seleccione una variable para analizar",
                            [col for col in numeric_columns if col not in ['CUST_ID', 'Cluster']])

    fig = px.box(df, x='Cluster', y=variable, points="all")
    st.plotly_chart(fig)

    st.subheader("Matriz de Correlación")
    corr = df.drop(['CUST_ID', 'Cluster'], axis=1).corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig)

    st.subheader("Análisis de Componentes Principales")
    pca_vars = [col for col in df.columns if col.startswith('PC')]
    if pca_vars:
        pca_df = df[pca_vars]

        explained_variance = pca_model.explained_variance_ratio_
        cum_explained_variance = np.cumsum(explained_variance)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1, len(explained_variance) + 1)),
                             y=explained_variance,
                             name='Varianza Explicada Individual'))
        fig.add_trace(go.Scatter(x=list(range(1, len(cum_explained_variance) + 1)),
                                 y=cum_explained_variance,
                                 name='Varianza Explicada Acumulada'))
        fig.update_layout(title='Varianza Explicada por Componente Principal',
                          xaxis_title='Componente Principal',
                          yaxis_title='Proporción de Varianza Explicada')
        st.plotly_chart(fig)
    else:
        st.write("No se encontraron componentes principales en el conjunto de datos.")

# Pie de página
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado con ❤️ usando Streamlit")