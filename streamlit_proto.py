import streamlit as st
import pandas as pd
import numpy as np
import miceforest as mf
from autocluster.autohypothesis import autohypothesis_utils
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Fonction pour le preprocessing et l'imputation
def preprocess_and_impute(df):
    category_mappings = {}

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].astype('category')
            category_mappings[column] = dict(enumerate(df[column].cat.categories))
            df[column] = df[column].cat.codes.replace(-1, np.nan)

    kds = mf.ImputationKernel(df, save_all_iterations=True, random_state=100)
    kds.mice(10)
    df_imputed = kds.complete_data()

    for column in df_imputed.columns:
        if column in category_mappings:
            df_imputed[column] = df_imputed[column].map(category_mappings[column])

    non_ordinal_columns = [column for column in df_imputed.columns if df_imputed[column].dtype == 'object']
    df_imputed = pd.get_dummies(df_imputed, columns=non_ordinal_columns, drop_first=True)
    
    return df_imputed

# Interface Streamlit
st.title('Démonstrateur')
uploaded_file = st.file_uploader('Uploader un fichier CSV', type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    target = st.selectbox('Sélectionner la cible y', df.columns)
    
    df.rename(columns={target: 'target'}, inplace=True)
    # df.drop(columns=["brsa_clep"], inplace=True)

    df_imputed = preprocess_and_impute(df)

    st.write('Dataset')
    st.write(df_imputed.head())

    # Entrer le nombre de trials pour l'optimisation
    n_trials = st.number_input('Nombre de trials', min_value=1, value=500)
    run_optimization = st.button('Run')

    if run_optimization:
        # Réinitialiser l'état de l'optimisation pour garantir que l'ancienne optimisation est arrêtée
        if 'exp' in st.session_state:
            del st.session_state['exp']
            del st.session_state['X_train']
            del st.session_state['X_dev']
            del st.session_state['y_train']
            del st.session_state['y_dev']

        with st.spinner('Optimization in progress...'):
            exp, X_train, X_dev, y_train, y_dev = autohypothesis_utils.opti_loop(df_imputed, int(n_trials), optimize_obj="dual")
            st.session_state.exp = exp
            st.session_state.X_train = X_train
            st.session_state.X_dev = X_dev
            st.session_state.y_train = y_train
            st.session_state.y_dev = y_dev

    if 'exp' in st.session_state:
        exp = st.session_state.exp
        X_train = st.session_state.X_train
        X_dev = st.session_state.X_dev
        y_train = st.session_state.y_train
        y_dev = st.session_state.y_dev

        all_objectives = np.array([trial.values for trial in exp.best_trials])
        original_indices = np.array([trial.number for trial in exp.best_trials])
        models_df = pd.DataFrame(all_objectives, columns=['Accuracy', 'Simplicité'], index=original_indices).sort_values('Accuracy', ascending=False)

        st.write('Modèles générés')
        st.write(models_df)

        selected_model_index = st.selectbox('Sélectionner un modèle pour visualisation SHAP', models_df.index)
        if 'selected_model_index' not in st.session_state or st.session_state.selected_model_index != selected_model_index:
            st.session_state.selected_model_index = selected_model_index
            params = autohypothesis_utils.get_trial_hyperparams(exp, trial_number=selected_model_index)
            pipeline = autohypothesis_utils.rebuild_pipeline_with_hyperparams(params)
            pipeline.fit(X_train, y_train)
            st.session_state.pipeline = pipeline

        if 'pipeline' in st.session_state:
            pipeline = st.session_state.pipeline
            explainer = shap.Explainer(pipeline.named_steps['classifier'])
            shap_values = explainer.shap_values(X_dev)

            st.write('Visualisation SHAP - Graphique en nid d\'abeille')
            shap.summary_plot(shap_values[1], X_dev, show=False)
            st.pyplot(bbox_inches='tight')

            epsilon = st.number_input('Entrer la valeur de epsilon pour filtrer l\'importance des features', min_value=0.0, value=1.0, step=0.01)
            update_results = st.button('Update Results')

            tabs = st.tabs(["Toutes les instances", "Instance spécifique"])
            with tabs[0]:
                if update_results:
                    # Calculer les statistiques descriptives de X_dev
                    stats_describe = X_dev.describe()

                    def calculate_real_percentage_or_value(min_val, max_val, feature, stats):
                        if min_val == max_val:
                            return str(min_val)
                        else:
                            total_range = stats.at['max', feature] - stats.at['min', feature]
                            if total_range > 0:
                                min_percentage = ((min_val - stats.at['min', feature]) / total_range) * 100
                                max_percentage = ((max_val - stats.at['min', feature]) / total_range) * 100
                                return f"{min_percentage:.2f}%-{max_percentage:.2f}%"
                            else:
                                return "N/A"

                    all_data = []
                    for classe in range(len(shap_values)):
                        indices_of_class = [i for i, label in enumerate(y_dev) if label == classe]
                        global_mean_abs_shap = np.mean(np.abs(shap_values[classe][indices_of_class, :]), axis=0).sum()
                        data = []

                        for i, feature_name in enumerate(X_dev.columns):
                            positive_indices = [idx for idx in indices_of_class if shap_values[classe][idx, i] > 0]
                            negative_indices = [idx for idx in indices_of_class if shap_values[classe][idx, i] < 0]
                            original_values_positive = X_dev.iloc[positive_indices, i]
                            original_values_negative = X_dev.iloc[negative_indices, i]
                            positive_impact_percentage = len(positive_indices) / len(shap_values[classe][indices_of_class, i]) * 100
                            mean_shap_values = np.mean(shap_values[classe][indices_of_class, i], axis=0)
                            rule_for = f"{original_values_positive.min()}" if original_values_positive.min() == original_values_positive.max() else f"{original_values_positive.min()} and < {original_values_positive.max()}"
                            rule_against = f"{original_values_negative.min()}" if original_values_negative.min() == original_values_negative.max() else f"{original_values_negative.min()} and < {original_values_negative.max()}"
                            importance_sign = "Positive" if mean_shap_values > 0 else "Negative"
                            data.append({
                                'Class': classe,
                                'Feature': feature_name,
                                'Rule For': rule_for,
                                'Rule Against': rule_against,
                                'Couverture Pour': positive_impact_percentage,
                                'Couverture Contre': 100 - positive_impact_percentage,
                                'Intervalle Pour': calculate_real_percentage_or_value(original_values_positive.min(), original_values_positive.max(), feature_name, stats_describe),
                                'Intervalle Contre': calculate_real_percentage_or_value(original_values_negative.min(), original_values_negative.max(), feature_name, stats_describe),
                                'Importance': round((np.abs(shap_values[classe][:, i]).mean() / np.sum(np.abs(shap_values[classe]).mean(axis=0))) * 100, 2),
                                'Signe Importance': importance_sign
                            })
                        
                        all_data.extend(data)

                    results_df = pd.DataFrame(all_data).sort_values('Importance', ascending=False).query('`Couverture Pour` > 0')
                    selected_class = st.selectbox('Sélectionner une classe pour afficher les résultats', results_df['Class'].unique())
                    class_results_df = results_df.query(f'Class == {selected_class} and `Rule For` != "N/A" and Importance > {epsilon}')
                    st.write('Tableau des résultats par classe')
                    st.write(class_results_df)

            with tabs[1]:
                            instance_index = st.number_input('Entrer l\'index de l\'instance pour afficher les résultats SHAP', min_value=0, max_value=X_dev.shape[0]-1, step=1)

                            if st.button('Afficher les résultats pour l\'instance'):
                                # Accéder à l'instance spécifique dans shap_values
                                instance_shap_values = np.sum([shap_values[classe][instance_index, :] for classe in range(len(shap_values))], axis=0)

                                # Créer un DataFrame pour les features de l'instance sélectionnée
                                instance_data = []
                                for i, feature_name in enumerate(X_dev.columns):
                                    instance_data.append({
                                        'Feature': feature_name,
                                        'SHAP Value': instance_shap_values[i],
                                        'Instance Value': X_dev.iloc[instance_index][feature_name]
                                    })

                                instance_df = pd.DataFrame(instance_data).sort_values('SHAP Value', ascending=False)

                                # Tableau avec les features ayant un impact positif et négatif
                                positive_impact_df = instance_df.query('`SHAP Value` > 0')
                                negative_impact_df = instance_df.query('`SHAP Value` < 0')

                                # Normaliser les valeurs SHAP pour que la somme de chaque groupe soit égale à 100
                                def normalize_shap_values(df):
                                    total_sum = df['SHAP Value'].sum()
                                    df['Importance'] = 100 * df['SHAP Value'] / total_sum
                                    return df

                                positive_impact_df = normalize_shap_values(positive_impact_df)
                                negative_impact_df = normalize_shap_values(negative_impact_df)

                                # Filtrer les features avec un impact supérieur à epsilon ou inférieur à -epsilon
                                filtered_positive_impact_df = positive_impact_df.query('`Importance` > @epsilon')
                                filtered_negative_impact_df = negative_impact_df.query('`Importance` > @epsilon')

                                # Ajouter les valeurs des features de l'instance sélectionnée aux tableaux affichés
                                #filtered_positive_impact_df = filtered_positive_impact_df.assign(Instance_Value=X_dev.iloc[instance_index][filtered_positive_impact_df['Feature']].values)
                                #filtered_negative_impact_df = filtered_negative_impact_df.assign(Instance_Value=X_dev.iloc[instance_index][filtered_negative_impact_df['Feature']].values)
                                filtered_positive_impact_df.drop(columns='SHAP Value', inplace=True)
                                filtered_negative_impact_df.drop(columns='SHAP Value', inplace=True)
                                # Afficher les résultats
                                st.write("Features avec Impact Positif:")
                                st.write(filtered_positive_impact_df.sort_values('Importance', ascending=False))
                                fig_pos, ax_pos = plt.subplots()
                                sns.barplot(x='Importance', y='Feature', data=filtered_positive_impact_df, ax=ax_pos)
                                st.pyplot(fig_pos)

                                st.write("Features avec Impact Négatif:")
                                st.write(filtered_negative_impact_df.sort_values('Importance', ascending=False))
                                fig_neg, ax_neg = plt.subplots()
                                sns.barplot(x='Importance', y='Feature', data=filtered_negative_impact_df, ax=ax_neg)
                                st.pyplot(fig_neg)

                    #st.write("Features avec Impact Faible (<= epsilon):")
                    #st.write(low_impact_df.sort_values('Importance', ascending=False))
                    #fig_low, ax_low = plt.subplots()
                    #sns.barplot(x='Importance', y='Feature', data=low_impact_df, ax=ax_low)
                    #st.pyplot(fig_low)
